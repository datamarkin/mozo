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
    "concept_state_dict",
    "load_state_dict",
    "text_state_dict",
    "vision_state_dict",
]

#: Where the image model lives inside the published checkpoint. Everything under here is what
#: upstream calls the "detector" -- the concept path plus the shared trunk.
DETECTOR_PREFIX = "detector."

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

#: Keys this package does not build, matched by prefix. Two reasons appear here:
#:
#: - ``text_projection`` produces the pooled branch of the text tower, which SAM 3's caller drops
#:   on the floor -- see ``text/encoder.py``.
#: - the ``points_*`` projections encode point exemplars, which the geometry encoder supports but
#:   no public API reaches, and for which there is no Apache-licensed implementation to derive
#:   the behaviour from.
UNUSED: tuple[str, ...] = (
    "backbone.language_backbone.encoder.text_projection",
    "geometry_encoder.points_direct_project",
    "geometry_encoder.points_pool_project",
    "geometry_encoder.points_pos_enc_project",
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

    Args:
        path: Path to ``sam3.pt``.

    Returns:
        The raw state dict, still in Meta's key layout.
    """
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
