# SPDX-License-Identifier: Apache-2.0
"""Mapping Meta's published ``sam3.pt`` onto this package's module names.

mozo consumes the checkpoint Meta ships, byte for byte -- no repacking, no pruning, no separate
mozo-format artifact. What it needs instead is a translation table, because the module names here
follow ``transformers`` (the Apache-2.0 implementation this package derives from) while the
checkpoint follows ``facebookresearch/sam3``.

The table below is adapted from ``transformers/models/sam3/convert_sam3_to_hf.py`` (Apache-2.0),
with two differences that follow from choices made in :mod:`.vision.vit`:

- ``qkv`` is *not* split into three projections. It stays as the checkpoint stores it.
- ``freqs_cis`` is *not* discarded and recomputed. It is loaded as shipped and used as shipped.

Ordering matters: the rules are applied in sequence, so a rule whose pattern is a prefix of an
earlier one must come after it.
"""

from __future__ import annotations

import re
from pathlib import Path

import torch

__all__ = [
    "DETECTOR_PREFIX",
    "TRACKER_PREFIX",
    "CLICK_RULES",
    "concept_state_dict",
    "load_state_dict",
    "text_state_dict",
    "click_state_dict",
    "vision_state_dict",
]

#: Where the image model lives inside the published checkpoint. Everything under here is what
#: upstream calls the "detector" -- the concept path plus the shared trunk.
DETECTOR_PREFIX = "detector."

#: Where the click path lives. Named for the video tracker because that is what owns it upstream,
#: but the prompt encoder and mask decoder under here are what answers a click on a still image.
TRACKER_PREFIX = "tracker."

#: Meta's key -> this package's key, applied in order. Values are regex replacements.
VISION_RULES: tuple[tuple[str, str], ...] = (
    # --- trunk -------------------------------------------------------------------------------
    (r"^trunk\.patch_embed\.proj\.", r"trunk.embeddings.patch.projection."),
    (r"^trunk\.pos_embed$", r"trunk.embeddings.position_embeddings"),
    (r"^trunk\.ln_pre\.", r"trunk.layer_norm."),
    (r"^trunk\.blocks\.(\d+)\.norm1\.", r"trunk.layers.\1.layer_norm1."),
    (r"^trunk\.blocks\.(\d+)\.norm2\.", r"trunk.layers.\1.layer_norm2."),
    (r"^trunk\.blocks\.(\d+)\.attn\.qkv\.", r"trunk.layers.\1.attention.qkv."),
    (r"^trunk\.blocks\.(\d+)\.attn\.proj\.", r"trunk.layers.\1.attention.o_proj."),
    (r"^trunk\.blocks\.(\d+)\.attn\.freqs_cis$", r"trunk.layers.\1.attention.freqs_cis"),
    (r"^trunk\.blocks\.(\d+)\.mlp\.fc1\.", r"trunk.layers.\1.mlp.fc1."),
    (r"^trunk\.blocks\.(\d+)\.mlp\.fc2\.", r"trunk.layers.\1.mlp.fc2."),
    # --- neck --------------------------------------------------------------------------------
    # ``dconv_2x2_0`` and ``dconv_2x2`` are told apart by the trailing dot in each pattern, not
    # by their order here.
    (r"^convs\.(\d+)\.dconv_2x2_0\.", r"neck.levels.\1.scale_layers.0."),
    (r"^convs\.(\d+)\.dconv_2x2_1\.", r"neck.levels.\1.scale_layers.2."),
    (r"^convs\.(\d+)\.dconv_2x2\.", r"neck.levels.\1.scale_layers.0."),
    (r"^convs\.(\d+)\.conv_1x1\.", r"neck.levels.\1.proj1."),
    (r"^convs\.(\d+)\.conv_3x3\.", r"neck.levels.\1.proj2."),
    (r"^sam2_convs\.(\d+)\.dconv_2x2_0\.", r"neck.click_levels.\1.scale_layers.0."),
    (r"^sam2_convs\.(\d+)\.dconv_2x2_1\.", r"neck.click_levels.\1.scale_layers.2."),
    (r"^sam2_convs\.(\d+)\.dconv_2x2\.", r"neck.click_levels.\1.scale_layers.0."),
    (r"^sam2_convs\.(\d+)\.conv_1x1\.", r"neck.click_levels.\1.proj1."),
    (r"^sam2_convs\.(\d+)\.conv_3x3\.", r"neck.click_levels.\1.proj2."),
)

#: Meta's key -> this package's key for the text tower, applied in order.
TEXT_RULES: tuple[tuple[str, str], ...] = (
    (r"^encoder\.token_embedding\.", r"tower.token_embedding."),
    (r"^encoder\.positional_embedding$", r"tower.position_embedding"),
    (r"^encoder\.ln_final\.", r"tower.final_layer_norm."),
    (r"^encoder\.transformer\.resblocks\.(\d+)\.attn\.", r"tower.layers.\1.attention."),
    (r"^encoder\.transformer\.resblocks\.(\d+)\.ln_1\.", r"tower.layers.\1.layer_norm1."),
    (r"^encoder\.transformer\.resblocks\.(\d+)\.ln_2\.", r"tower.layers.\1.layer_norm2."),
    (r"^encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_fc\.", r"tower.layers.\1.mlp.fc1."),
    (r"^encoder\.transformer\.resblocks\.(\d+)\.mlp\.c_proj\.", r"tower.layers.\1.mlp.fc2."),
)

#: Meta's key -> this package's key for the concept head, applied in order. The five sub-modules
#: live under three unrelated prefixes upstream, so most of the work is flattening them onto
#: :class:`~.grounding.concept.ConceptHead`'s attributes.
CONCEPT_RULES: tuple[tuple[str, str], ...] = (
    (r"^geometry_encoder\.", r"geometry."),
    (r"^transformer\.encoder\.", r"fusion."),
    (r"^transformer\.decoder\.", r"decoder."),
    (r"^dot_prod_scoring\.", r"scoring."),
    # ``mask_predictor`` wraps a single MLP and nothing else; the wrapper is dropped.
    (r"^segmentation_head\.mask_predictor\.mask_embed\.", r"mask.mask_embed."),
    (r"^segmentation_head\.", r"mask."),
)

#: Meta's key -> this package's key for the click path, applied in order.
#:
#: A longer table than the other three, because the click head derives from
#: ``transformers/models/sam3_tracker`` while the checkpoint follows ``facebookresearch/sam3``,
#: and the two disagree about names far more here than they do in the towers. Nothing below
#: changes a number: every rule renames a module or reindexes a ``Sequential`` whose members
#: this package gives names to.
CLICK_RULES: tuple[tuple[str, str], ...] = (
    # --- the two-way transformer, before anything generic ---------------------------------
    # Every attention here calls its output projection ``out_proj``; ours follows
    # ``transformers`` and calls it ``o_proj``. One rule covers the blocks and the final pass.
    (r"^sam_mask_decoder\.transformer\.(.+)\.out_proj\.",
     r"mask_decoder.transformer.\1.o_proj."),
    (r"^sam_mask_decoder\.transformer\.layers\.(\d+)\.norm(\d)\.",
     r"mask_decoder.transformer.layers.\1.layer_norm\2."),
    # The block's feed-forward is ``Mlp``, whose ``layers.N`` is the checkpoint's own naming
    # everywhere else; only these two keys spell it differently.
    (r"^sam_mask_decoder\.transformer\.layers\.(\d+)\.mlp\.lin1\.",
     r"mask_decoder.transformer.layers.\1.mlp.layers.0."),
    (r"^sam_mask_decoder\.transformer\.layers\.(\d+)\.mlp\.lin2\.",
     r"mask_decoder.transformer.layers.\1.mlp.layers.1."),
    (r"^sam_mask_decoder\.transformer\.norm_final_attn\.",
     r"mask_decoder.transformer.layer_norm_final_attn."),
    (r"^sam_mask_decoder\.transformer\.", r"mask_decoder.transformer."),
    # --- the upscaling Sequential, whose members are named here ----------------------------
    # The indices it skips are activations, which carry no weights.
    (r"^sam_mask_decoder\.output_upscaling\.0\.", r"mask_decoder.upscale_conv1."),
    (r"^sam_mask_decoder\.output_upscaling\.1\.", r"mask_decoder.upscale_layer_norm."),
    (r"^sam_mask_decoder\.output_upscaling\.3\.", r"mask_decoder.upscale_conv2."),
    (r"^sam_mask_decoder\.", r"mask_decoder."),
    # --- prompt encoder --------------------------------------------------------------------
    (r"^sam_prompt_encoder\.pe_layer\.positional_encoding_gaussian_matrix$",
     r"prompt_encoder.shared_embedding.positional_embedding"),
    (r"^sam_prompt_encoder\.mask_downscaling\.0\.", r"prompt_encoder.mask_embed.conv1."),
    (r"^sam_prompt_encoder\.mask_downscaling\.1\.", r"prompt_encoder.mask_embed.layer_norm1."),
    (r"^sam_prompt_encoder\.mask_downscaling\.3\.", r"prompt_encoder.mask_embed.conv2."),
    (r"^sam_prompt_encoder\.mask_downscaling\.4\.", r"prompt_encoder.mask_embed.layer_norm2."),
    (r"^sam_prompt_encoder\.mask_downscaling\.6\.", r"prompt_encoder.mask_embed.conv3."),
    (r"^sam_prompt_encoder\.", r"prompt_encoder."),
)



#: Keys this package does not build, matched by prefix. Two reasons appear here:
#:
#: - ``text_projection`` produces the pooled branch of the text tower, which SAM 3's caller drops
#:   on the floor -- see ``text/encoder.py``.
#: - the ``points_*`` projections encode point exemplars, which the geometry encoder supports but
#:   no public API reaches, and for which there is no Apache-licensed implementation to derive
#:   the behaviour from.
#: - everything under ``tracker.`` that is not the prompt encoder or the mask decoder is memory
#:   attention and mask-memory fusion. Those carry a segmentation from one video frame to the
#:   next and have nothing to attend to on a still image -- 160 tensors and 7.5 M parameters,
#:   against the 4.2 M the click path actually runs.
UNUSED: tuple[str, ...] = (
    "backbone.language_backbone.encoder.text_projection",
    "geometry_encoder.points_direct_project",
    "geometry_encoder.points_pool_project",
    "geometry_encoder.points_pos_enc_project",
    "tracker.transformer.",
    "tracker.maskmem_backbone.",
    "tracker.maskmem_tpos_enc",
    "tracker.mask_downsample.",
    "tracker.no_mem_pos_enc",
    "tracker.no_obj_ptr",
    "tracker.no_obj_embed_spatial",
    "tracker.obj_ptr_proj.",
    "tracker.obj_ptr_tpos_proj.",
)


def _skipped(key: str) -> bool:
    """Is this a weight the package deliberately does not build? See :data:`UNUSED`."""
    return key.startswith(UNUSED)


def rename(key: str, rules: tuple[tuple[str, str], ...]) -> str:
    """Apply ``rules`` to ``key`` in order, returning the first rewrite that matches."""
    for pattern, replacement in rules:
        renamed, count = re.subn(pattern, replacement, key)
        if count:
            return renamed
    return key


def _section(
    checkpoint: dict[str, torch.Tensor], prefix: str, what: str
) -> dict[str, torch.Tensor]:
    """Pull one tower out of the checkpoint, with its prefix stripped.

    Args:
        checkpoint: The raw state dict.
        prefix: Where the tower lives.
        what: Its name, for the error message.

    Returns:
        The tower's tensors, keyed relative to ``prefix``.

    Raises:
        KeyError: If nothing lives under ``prefix``. An empty tower means the file is not a SAM 3
            checkpoint at all, rather than that one layer is missing.
    """
    section = {k[len(prefix):]: v for k, v in checkpoint.items() if k.startswith(prefix)}
    if not section:
        raise KeyError(f"no {what} under {prefix!r} -- is this a SAM 3 checkpoint?")
    return section


def load_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    """Read a published checkpoint, unwrapping the ``model`` envelope if there is one.

    Mapped rather than materialised. Every caller wants a *subset* of this file -- the trunk, or
    the text tower, or the tracker -- and hands it straight to a module's ``load_state_dict``,
    which copies what it names into parameters the module already allocated. So a mapped tensor
    is a source to copy out of and never a tensor anything computes on, and a section nobody
    names is never faulted in.

    That matters most to the caller who wants least. A :class:`~.predictor.Segmenter` built with
    a graph vision encoder never asks for the trunk, and the trunk is 1.85 GB of this file:
    mapping takes its construction from 5.28 GB peak and 2.26 s to 3.37 GB and 1.76 s. Reading
    the whole checkpoint costs the same either way once every section *is* wanted -- 7.16 GB
    against 7.13 GB on the torch path -- so this is not a trade, it is the same read deferred.

    Mapping needs the zipfile serialisation ``torch.save`` has written by default since 1.6, and
    raises rather than falling back on a checkpoint written the old way. Since a caller may hand
    us one of their own -- ``Sam3Predictor`` takes a ``checkpoint_path`` -- the old way is read
    the old way. The tensors are the same either way; only what it costs to get them differs, so
    this is a performance path quietly declining, not a difference anything downstream can see.

    Args:
        path: Path to ``sam3.pt``.

    Returns:
        The raw state dict, still in Meta's key layout, backed by the mapped file where it could
        be mapped.
    """
    try:
        blob = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except RuntimeError:
        blob = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in blob and isinstance(blob["model"], dict):
        blob = blob["model"]
    return blob


def vision_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract and translate the weights :class:`~.vision.encoder.VisionEncoder` needs.

    Args:
        checkpoint: What :func:`load_state_dict` returned.

    Returns:
        A state dict keyed for this package, ready for ``load_state_dict(..., strict=True)``.

    Raises:
        KeyError: If the checkpoint carries no vision backbone at all, which means it is not a
            SAM 3 checkpoint rather than that some layer is missing.
    """
    source = _section(checkpoint, f"{DETECTOR_PREFIX}backbone.vision_backbone.", "vision backbone")

    out: dict[str, torch.Tensor] = {}
    for key, tensor in source.items():
        renamed = rename(key, VISION_RULES)

        if renamed.endswith("embeddings.position_embeddings"):
            # The checkpoint ships 577 positions: a leading class-token position, then 576
            # patches. This trunk is built without a class token, so index 0 is dropped -- the
            # same slice upstream's ``get_abs_pos`` takes.
            tensor = tensor[:, 1:]

        out[renamed] = tensor

    return out


def text_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract and translate the weights :class:`~.text.encoder.TextEncoder` needs.

    Args:
        checkpoint: What :func:`load_state_dict` returned.

    Returns:
        A state dict keyed for this package, ready for ``load_state_dict(..., strict=True)``.

    Raises:
        KeyError: If the checkpoint carries no language backbone.
    """
    source = _section(
        checkpoint, f"{DETECTOR_PREFIX}backbone.language_backbone.", "language backbone"
    )
    return {
        rename(key, TEXT_RULES): tensor
        for key, tensor in source.items()
        if not _skipped(f"backbone.language_backbone.{key}")
    }


def concept_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract and translate the weights :class:`~.grounding.concept.ConceptHead` needs.

    Args:
        checkpoint: What :func:`load_state_dict` returned.

    Returns:
        A state dict keyed for this package, ready for ``load_state_dict(..., strict=True)``.

    Raises:
        KeyError: If the checkpoint carries no detector at all.
    """
    source = _section(checkpoint, DETECTOR_PREFIX, "detector")
    # The rules' own patterns say which prefixes belong to this head; keeping a second list in
    # step with them by hand is how the two drift apart.
    wanted = tuple(pattern.lstrip("^").replace("\\.", ".") for pattern, _ in CONCEPT_RULES)
    return {
        rename(key, CONCEPT_RULES): tensor
        for key, tensor in source.items()
        if key.startswith(wanted) and not _skipped(key)
    }


def click_state_dict(checkpoint: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract and translate the weights :class:`~.click.ClickHead` needs.

    Args:
        checkpoint: What :func:`load_state_dict` returned.

    Returns:
        A state dict keyed for this package, ready for ``load_state_dict(..., strict=True)``.

    Raises:
        KeyError: If the checkpoint carries no tracker section.
    """
    source = _section(checkpoint, TRACKER_PREFIX, "click path")

    out: dict[str, torch.Tensor] = {}
    corners: dict[int, torch.Tensor] = {}
    for key, tensor in source.items():
        if _skipped(f"{TRACKER_PREFIX}{key}"):
            continue
        # The one structural difference. Upstream keeps four separate one-row embeddings, one
        # per label; ``transformers`` keeps a single table indexed by the label, which is what
        # lets the prompt encoder add them with an index rather than four ``where`` branches.
        # Stacking them in label order is the whole of the change -- no value moves.
        index = re.fullmatch(r"sam_prompt_encoder\.point_embeddings\.(\d+)\.weight", key)
        if index:
            corners[int(index.group(1))] = tensor
            continue
        out[rename(key, CLICK_RULES)] = tensor

    if corners:
        out["prompt_encoder.point_embed.weight"] = torch.cat(
            [corners[i] for i in sorted(corners)], dim=0
        )
    return out
