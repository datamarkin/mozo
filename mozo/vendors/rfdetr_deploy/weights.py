# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Checkpoint loading for RF-DETR deployment.

Upstream's loader is shared with training, so it tolerates partial loads and reshapes heads in both directions to
support fine-tuning and resume.  Deployment has the opposite requirement: a checkpoint that does not fully populate the
model is a bug, not a starting point.  This loader therefore aligns the model to the checkpoint (never the reverse) and
rejects any missing or unexpected key, so a mismatch surfaces immediately rather than silently leaving a head at its
random initialization.  The only exception is buffers the model derives from its own keypoint schema.

Checkpoints arrive as a path. Locating and fetching them is the host application's concern, not this package's --
see ``mozo.weights`` for how mozo does it.
"""

from __future__ import annotations

__all__ = ["load_checkpoint", "load_state_dict_into"]

import argparse
import math
import os
import pickle
import types
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from .utilities.logger import get_logger

logger = get_logger()

_PE_KEY_SUFFIX = "embeddings.position_embeddings"
_QUERY_PARAM_SUFFIXES: tuple[str, ...] = ("refpoint_embed.weight", "query_feat.weight")
# Buffers the model recomputes from its keypoint schema at construction time. They are not learned, so a
# checkpoint may carry a stale copy or omit them entirely; either way the model's own value is correct.
_DERIVED_BUFFER_KEYS: frozenset[str] = frozenset({"_kp_active_mask", "transformer.keypoint_class_mask"})
# Weights the published checkpoints still carry but no current model consumes. The preview keypoint checkpoint
# stores the old standalone MLP projection head that the GroupPose inference path replaced; ignoring it is safe
# precisely because it is *unexpected* rather than *missing* — no model parameter is left uninitialized.
_STALE_CHECKPOINT_PREFIXES: tuple[str, ...] = ("keypoint_head.keypoint_proj.",)
def load_checkpoint(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Load a checkpoint and normalize it to a ``{"model": state_dict, "args": ...}`` shape.

    Loading always uses ``weights_only=True``.  ``argparse.Namespace`` and ``types.SimpleNamespace`` are allowed through
    a scoped safe-globals context because pre-Lightning RF-DETR checkpoints embed an ``args`` namespace of primitives;
    no pickle fallback is offered, since a checkpoint that needs one is not one this package should execute.

    PyTorch Lightning ``.ckpt`` files store weights under ``state_dict`` with a ``model.`` prefix (and
    ``model._orig_mod.`` when the module was compiled); both prefixes are stripped here.

    Args:
        path: Path to a ``.pth`` / ``.pt`` / ``.ckpt`` checkpoint.

    Returns:
        A dict with a ``"model"`` key holding the state dict, plus whatever other keys the file carried.

    Raises:
        ValueError: If the file is Lightning-shaped but no key carries the ``model.`` prefix.

    Examples:
        >>> import tempfile, torch, pathlib
        >>> tmp = pathlib.Path(tempfile.mkdtemp()) / "ckpt.pth"
        >>> torch.save({"model": {"w": torch.zeros(1)}}, tmp)
        >>> sorted(load_checkpoint(tmp)["model"])
        ['w']
    """
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)
    except (RuntimeError, pickle.UnpicklingError):
        # Checkpoints written by the pre-Lightning training loop embed an `args` namespace. Allowing those two types
        # is safe: both hold only primitives, and the allow-list is scoped to this call rather than set process-wide.
        # `safe_globals` as a context manager is newer than `weights_only` itself, hence the attribute check.
        safe_globals = getattr(torch.serialization, "safe_globals", None)
        if safe_globals is None:
            raise
        with safe_globals([argparse.Namespace, types.SimpleNamespace]):
            checkpoint = torch.load(str(path), map_location="cpu", weights_only=True)

    if "model" not in checkpoint and "state_dict" in checkpoint:
        prefix, compile_prefix = "model.", "_orig_mod."
        model_state = {}
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith(prefix):
                continue
            stripped = key[len(prefix) :]
            if stripped.startswith(compile_prefix):
                stripped = stripped[len(compile_prefix) :]
            model_state[stripped] = value
        if not model_state:
            raise ValueError(
                f"{path!r} looks like a Lightning checkpoint ('state_dict' present, 'model' absent) but no key "
                "carries the expected 'model.' prefix. The file may be corrupt or in an unsupported format."
            )
        checkpoint["model"] = model_state
        if "args" not in checkpoint and "hyper_parameters" in checkpoint:
            checkpoint["args"] = checkpoint["hyper_parameters"]
    return checkpoint


def _args_get(args: Any, field: str, default: Any = None) -> Any:
    """Read *field* off a checkpoint ``args`` entry, which may be a dict or a namespace.

    Args:
        args: The checkpoint's ``args`` value.
        field: Attribute or key to read.
        default: Value returned when *field* is absent.

    Returns:
        The stored value, or *default*.

    Examples:
        >>> _args_get({"num_queries": 300}, "num_queries")
        300
        >>> _args_get(argparse.Namespace(group_detr=13), "group_detr")
        13
        >>> _args_get(None, "missing", "fallback")
        'fallback'
    """
    if args is None:
        return default
    if isinstance(args, dict):
        return args.get(field, default)
    return getattr(args, field, default)


def _slice_query_param_per_group(
    tensor: Tensor,
    ckpt_num_queries: int,
    ckpt_group_detr: int,
    target_num_queries: int,
    target_group_detr: int,
) -> Tensor:
    """Slice a ``refpoint_embed`` / ``query_feat`` weight preserving per-group structure.

    ``LWDETR`` packs query embeddings as ``nn.Embedding(num_queries * group_detr, ...)`` where group ``g`` occupies the
    contiguous slot range ``[g * num_queries, (g + 1) * num_queries)``.  A flat ``tensor[:target_rows]`` slice scrambles
    groups whenever ``num_queries`` shrinks and ``group_detr > 1``: the tail of group 0 lands in group 1's slots.
    Inference reads only group 0, so the damage is invisible until the checkpoint is used for anything else.

    Args:
        tensor: Checkpoint tensor for ``refpoint_embed.weight`` or ``query_feat.weight``.
        ckpt_num_queries: ``num_queries`` recorded in the checkpoint's args.
        ckpt_group_detr: ``group_detr`` recorded in the checkpoint's args.
        target_num_queries: ``num_queries`` the model was built with.
        target_group_detr: ``group_detr`` the model was built with.

    Returns:
        A tensor laid out for the model's packing.

    Raises:
        ValueError: If any dimension argument is not positive.

    Examples:
        >>> weight = torch.arange(12).float().unsqueeze(1)
        >>> _slice_query_param_per_group(weight, 4, 3, 2, 3).squeeze(1).tolist()
        [0.0, 1.0, 4.0, 5.0, 8.0, 9.0]
    """
    if min(ckpt_num_queries, ckpt_group_detr, target_num_queries, target_group_detr) <= 0:
        raise ValueError(
            "all dimension args must be positive; got "
            f"ckpt_num_queries={ckpt_num_queries}, ckpt_group_detr={ckpt_group_detr}, "
            f"target_num_queries={target_num_queries}, target_group_detr={target_group_detr}."
        )

    expected_total = ckpt_num_queries * ckpt_group_detr
    if tensor.shape[0] != expected_total:
        logger.warning(
            "checkpoint args claim %d x %d = %d query rows but the tensor has %d; falling back to a flat slice.",
            ckpt_num_queries,
            ckpt_group_detr,
            expected_total,
            tensor.shape[0],
        )
        return tensor[: target_num_queries * target_group_detr]

    if target_num_queries == ckpt_num_queries and target_group_detr == ckpt_group_detr:
        return tensor

    keep_groups = min(target_group_detr, ckpt_group_detr)
    keep_per_group = min(target_num_queries, ckpt_num_queries)
    pieces = [tensor[g * ckpt_num_queries : g * ckpt_num_queries + keep_per_group] for g in range(keep_groups)]
    return torch.cat(pieces, dim=0)


def _interpolate_position_embeddings(state: dict[str, Any], pe_size: int) -> None:
    """Bicubic-resize DINOv2 positional embeddings in *state* to a ``pe_size`` x ``pe_size`` grid, in place.

    ``load_state_dict`` raises on a shape mismatch rather than skipping it, so a model built at a resolution other than
    the checkpoint's must have its positional embeddings resized before loading.

    Args:
        state: The checkpoint's model state dict, mutated in place.
        pe_size: Target grid side length in patches.

    Examples:
        >>> state = {"backbone.embeddings.position_embeddings": torch.zeros(1, 17, 8)}
        >>> _interpolate_position_embeddings(state, 2)
        >>> tuple(state["backbone.embeddings.position_embeddings"].shape)
        (1, 5, 8)
    """
    n_target = pe_size * pe_size
    for key in [k for k in state if k.endswith(_PE_KEY_SUFFIX)]:
        ckpt_pe = state[key]
        n_source = ckpt_pe.shape[1] - 1
        if n_source == n_target:
            continue

        h_src, h_tgt = math.isqrt(n_source), math.isqrt(n_target)
        if h_src * h_src != n_source or h_tgt * h_tgt != n_target:
            logger.warning(
                "skipping positional-embedding interpolation for %s: grid is not square (source %d, target %d).",
                key,
                n_source,
                n_target,
            )
            continue

        dim = ckpt_pe.shape[-1]
        class_token, patch_pe = ckpt_pe[:, :1], ckpt_pe[:, 1:]
        patch_pe = patch_pe.reshape(1, h_src, h_src, dim).permute(0, 3, 1, 2)
        patch_pe = F.interpolate(
            patch_pe.float(),
            size=(h_tgt, h_tgt),
            mode="bicubic",
            align_corners=False,
            # antialias is unimplemented for bicubic on MPS; upstream makes the same exception.
            antialias=patch_pe.device.type != "mps",
        ).to(ckpt_pe.dtype)
        state[key] = torch.cat([class_token, patch_pe.permute(0, 2, 3, 1).reshape(1, n_target, dim)], dim=1)


def _seed_cross_attn_projector(state: dict[str, Any], model: Any) -> None:
    """Clone backbone projector weights into ``cross_attn_projector`` when a dual-projector model lacks them.

    Older dual-projector checkpoints carry only ``backbone.0.projector.*``.  The keypoint model expects a second
    branch; seeding it from the first matches upstream behaviour.

    Args:
        state: Checkpoint model state dict, mutated in place.
        model: The instantiated model, inspected to detect dual-projector mode.

    Examples:
        >>> _seed_cross_attn_projector({}, object())
    """
    backbone = getattr(model, "backbone", None)
    backbone = backbone[0] if backbone is not None else None
    if backbone is None:
        return
    has_branch = getattr(backbone, "cross_attn_projector", None) is not None
    if not (has_branch or bool(getattr(backbone, "dual_projector", False))):
        return
    if any(key.startswith("backbone.0.cross_attn_projector.") for key in state):
        return
    projector_keys = {k: v for k, v in state.items() if k.startswith("backbone.0.projector.")}
    if not projector_keys:
        return
    logger.info("Seeding %d cross_attn_projector key(s) from the backbone projector.", len(projector_keys))
    for key, value in projector_keys.items():
        state[key.replace("backbone.0.projector.", "backbone.0.cross_attn_projector.", 1)] = value.clone()


def load_state_dict_into(model: Any, checkpoint: dict[str, Any], *, positional_encoding_size: int) -> list[str]:
    """Load *checkpoint* into *model* strictly, adapting the checkpoint to the model's geometry first.

    The adaptations are the ones that can be derived unambiguously from the checkpoint itself: per-group query slicing
    (using the recorded ``num_queries`` / ``group_detr``), positional-embedding interpolation to the model's grid, and
    seeding a dual-projector branch.  Everything else must already agree — the final load is ``strict=True`` so that a
    head left at random initialization raises here instead of producing quietly wrong predictions.

    Args:
        model: The instantiated ``LWDETR``.
        checkpoint: A checkpoint dict as returned by :func:`load_checkpoint`.
        positional_encoding_size: The model's backbone positional grid side length.

    Returns:
        Class names recorded in the checkpoint, or an empty list when absent.

    Raises:
        RuntimeError: If the state dict does not match the model exactly.

    Examples:
        >>> load_state_dict_into(object(), {"model": {}}, positional_encoding_size=24)  # doctest: +SKIP
        []
    """
    state = dict(checkpoint["model"])
    args = checkpoint.get("args")

    ckpt_num_queries = _args_get(args, "num_queries")
    ckpt_group_detr = _args_get(args, "group_detr")
    target_num_queries = int(getattr(model, "num_queries", 0) or 0)
    target_group_detr = int(getattr(model, "group_detr", 1) or 1)
    target_query_rows = target_num_queries * target_group_detr
    if target_num_queries > 0:
        for name in [k for k in state if k.endswith(_QUERY_PARAM_SUFFIXES)]:
            if ckpt_num_queries is not None and ckpt_group_detr is not None:
                state[name] = _slice_query_param_per_group(
                    state[name],
                    ckpt_num_queries=int(ckpt_num_queries),
                    ckpt_group_detr=int(ckpt_group_detr),
                    target_num_queries=target_num_queries,
                    target_group_detr=target_group_detr,
                )
            else:
                # Several published checkpoints record no num_queries / group_detr (the segmentation releases among
                # them), so the per-group layout cannot be recovered. A flat head slice is what upstream falls back to,
                # and it is right for inference either way: only group 0 is ever read, and it occupies the leading rows
                # under both packings. Groups 1+ would be scrambled, which matters only for training resume.
                state[name] = state[name][:target_query_rows]

    _seed_cross_attn_projector(state, model)
    _interpolate_position_embeddings(state, positional_encoding_size)

    # Derived buffers are recomputed from `num_keypoints_per_class` during construction, so the model's own values are
    # authoritative and a checkpoint copy is redundant (detection checkpoints omit them entirely). Dropping them keeps
    # the load strict about everything that is actually learned.
    for key in _DERIVED_BUFFER_KEYS:
        state.pop(key, None)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [key for key in incompatible.missing_keys if key not in _DERIVED_BUFFER_KEYS]
    unexpected = [key for key in incompatible.unexpected_keys if not key.startswith(_STALE_CHECKPOINT_PREFIXES)]
    if unexpected or missing:
        raise RuntimeError(
            "checkpoint does not match the model. "
            f"Missing keys (would stay randomly initialized): {missing or 'none'}. "
            f"Unexpected keys: {unexpected or 'none'}."
        )

    raw_names = _args_get(args, "class_names")
    if isinstance(raw_names, str):
        return [raw_names]
    if raw_names:
        return [name for name in raw_names if isinstance(name, str)]
    return []
