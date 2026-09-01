# ------------------------------------------------------------------------
# BEN2 -- Background Erase Network
# Copyright (c) 2025 Prama LLC. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""The geometry of the one published BEN2 model, written down rather than inferred.

Upstream has no config: ``BEN_Base.__init__`` hardcodes every number inline, and the Swin
constructor's defaults are Swin-T's, overridden at the one call site. Collecting them here is
what lets the strict load check them -- §5 of ``plans/vendoring.md``, "geometry that looks
derivable usually is not".

**The input resolution is not a preference, it is the architecture.** ``BEN_Base.forward`` splits
its input into four quadrants and pairs them with a half-scale copy of the whole frame, so the
backbone always sees five 512x512 images regardless of what came in. Changing ``INPUT`` changes
the pooling grids in every MCLM and MCRM block, and the published weights would no longer mean
anything. It is a constant so that a future change fails loudly here instead of silently there.
"""

from __future__ import annotations

__all__ = ["BACKBONE", "DECODER", "INPUT", "MEAN", "STD", "BackboneSpec", "DecoderSpec"]

from dataclasses import dataclass


#: Side length of the square the image is resized to before anything else. Aspect ratio is not
#: preserved -- upstream squashes, the model was trained squashed, and letterboxing instead moves
#: every pixel of the matte.
INPUT: int = 1024

#: ImageNet statistics, which is what upstream normalises with. Worth stating rather than
#: assuming: §5 warns that these are usually *not* ImageNet's, and here they are.
MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class BackboneSpec:
    """Swin-B, at window 12.

    ``embed_dim=128`` with ``depths=(2, 2, 18, 2)`` is Swin-**B**; Swin-S shares the depths at
    ``embed_dim=96`` and Swin-L uses 192 with different head counts. Naming it here because the
    upstream call site passes the numbers positionally and never says which Swin it is.

    Attributes:
        embed_dim: Width after patch embedding. Stage *i* is ``embed_dim * 2**i``.
        depths: Blocks per stage.
        num_heads: Attention heads per stage. Head dimension is 32 at every stage.
        window_size: Attention window. 12, not Swin's usual 7.
        patch_size: Patch-embedding stride.
        drop_path_rate: Stochastic depth, linearly ramped across all 24 blocks. Inert at eval --
            recorded because it decides which blocks get a ``DropPath`` and which get an
            ``Identity``, and therefore the module tree the checkpoint is loaded into.
        out_indices: Which stages emit a normed feature map.
    """

    embed_dim: int = 128
    depths: tuple[int, ...] = (2, 2, 18, 2)
    num_heads: tuple[int, ...] = (4, 8, 16, 32)
    window_size: int = 12
    patch_size: int = 4
    drop_path_rate: float = 0.2
    out_indices: tuple[int, ...] = (0, 1, 2, 3)

    @property
    def channels(self) -> tuple[int, ...]:
        """Channel width of each of the five feature maps the backbone returns.

        Five, not four: ``SwinTransformer.forward`` seeds its output list with the patch-embed
        tensor before the stages run, so ``features[0]`` is pre-stage and ``features[1:]`` are the
        four stage outputs. The decoder's ``output1`` and ``output2`` both take 128 for exactly
        that reason, and a reader counting stages would wire them wrongly.
        """
        return (self.embed_dim,) + tuple(self.embed_dim * 2 ** i for i in range(len(self.depths)))


@dataclass(frozen=True)
class DecoderSpec:
    """The cross-attention decoder that turns five feature maps into one matte.

    Attributes:
        emb_dim: Common width every backbone stage is projected to.
        num_heads: Attention heads in MCLM and MCRM. One -- not a typo, and not derived from the
            backbone's head counts.
        mclm_pools: Pooling ratios in the global block, applied at the deepest stage.
        mcrm_pools: Pooling ratios in each of the four refinement blocks.
        head_width: Width of the instance-mask head's hidden convolutions.
    """

    emb_dim: int = 128
    num_heads: int = 1
    mclm_pools: tuple[int, ...] = (1, 4, 8)
    mcrm_pools: tuple[int, ...] = (2, 4, 8)
    head_width: int = 384


BACKBONE = BackboneSpec()
DECODER = DecoderSpec()
