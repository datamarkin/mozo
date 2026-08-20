# SPDX-License-Identifier: Apache-2.0
"""Where the masks finally come out.

Derived from ``transformers/models/sam3/modeling_sam3.py`` (Apache-2.0).

The shape of this stage is MaskFormer's: instead of predicting a mask per pixel per class, it
predicts one *embedding* per object query and one embedding per pixel, and takes their dot
product. A query's mask is wherever that query's embedding agrees with the image.

Three things about the pyramid are worth stating, because each looks like a bug and is not:

**The coarsest backbone level is replaced, not merged.** The pixel decoder consumes the fused
image tokens from the grounding encoder in place of the 72x72 level -- the prompt-conditioned
features, not the raw ones. That is the only route by which the prompt reaches the mask.

**The pixel decoder runs in channels-last.** The feature maps reaching it are channels-last
views, because reshaping the fused tokens back to a grid produces one. ``F.interpolate`` does not
preserve that format, so without restoring it each stage the convolutions and group norms run on
a different layout than upstream -- and both of those are layout-sensitive, unlike interpolation
and addition, which are not. Measured 9.57e-06 on the pixel embeddings, growing to 2.7e-04 on the
masks. The same trap the trunk's output permutation set in :mod:`..vision.vit`.

**The third upsampling stage never runs.** Upstream allocates three conv/norm pairs but iterates
once per *gap* between levels, and there are three levels, so two gaps. ``conv_layers[2]`` and
``norms[2]`` are built because the checkpoint carries them and a strict load must find a home for
them; they are dead at this configuration. Verified by tracing the published model.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import MASK, MaskSpec
from .layers import Mlp

__all__ = ["MaskHead", "PixelDecoder"]


class PixelDecoder(nn.Module):
    """An FPN that walks the pyramid coarse-to-fine into one dense feature map.

    Args:
        spec: Mask-head geometry.
    """

    def __init__(self, spec: MaskSpec):
        super().__init__()
        self.conv_layers = nn.ModuleList(
            nn.Conv2d(spec.hidden, spec.hidden, kernel_size=3, padding=1)
            for _ in range(spec.upsampling_stages)
        )
        self.norms = nn.ModuleList(
            nn.GroupNorm(spec.groups, spec.hidden) for _ in range(spec.upsampling_stages)
        )

    def forward(self, levels: list[Tensor]) -> Tensor:
        """Fuse ``levels`` -- finest first, coarsest last -- into ``(B, hidden, H, W)``.

        Starts at the coarsest level and walks outwards, upsampling and adding each finer level
        as a skip connection.
        """
        fused = levels[-1]
        for stage, finer in enumerate(reversed(levels[:-1])):
            fused = F.interpolate(fused, size=finer.shape[-2:], mode="nearest") + finer
            # Restore channels-last before the convolution -- see the module docstring.
            fused = fused.contiguous(memory_format=torch.channels_last)
            fused = F.relu(self.norms[stage](self.conv_layers[stage](fused)))
        return fused


class MaskHead(nn.Module):
    """Object queries plus the image, into one mask per query.

    Args:
        spec: Mask-head geometry.
    """

    def __init__(self, spec: MaskSpec = MASK):
        super().__init__()
        # Sequence-first, matching the layout these weights were run under. See ``layers.py``.
        self.cross_attend_prompt = nn.MultiheadAttention(spec.hidden, spec.heads)
        self.cross_attn_norm = nn.LayerNorm(spec.hidden)
        self.pixel_decoder = PixelDecoder(spec)
        self.mask_embed = Mlp((spec.hidden,) * 4)
        self.instance_seg_head = nn.Conv2d(spec.hidden, spec.hidden, kernel_size=1)
        self.semantic_seg_head = nn.Conv2d(spec.hidden, 1, kernel_size=1)

    def forward(
        self,
        queries: Tensor,
        levels: list[Tensor],
        fused: Tensor,
        prompt: Tensor,
        prompt_padding: Tensor,
    ) -> dict[str, Tensor]:
        """Predict one mask per query.

        Args:
            queries: ``(B, Q, hidden)`` final-layer query features.
            levels: The pyramid, finest first, coarsest last.
            fused: ``(B, H*W, hidden)`` prompt-conditioned image tokens from the fusion encoder.
            prompt: ``(P, B, hidden)`` sequence-first prompt tokens.
            prompt_padding: ``(B, P)`` True where the slot is padding.

        Returns:
            ``masks`` ``(B, Q, h, w)`` logits at the pyramid's finest resolution, and
            ``semantic`` ``(B, 1, h, w)``.
        """
        normed = self.cross_attn_norm(fused).transpose(0, 1)
        # ``need_weights=False`` here, and the default in the decoder's prompt cross-attention --
        # despite both carrying a key-padding mask. Which of PyTorch's two paths reproduces
        # upstream is not derivable from the call; it was measured at each site.
        attended, _ = self.cross_attend_prompt(
            normed, prompt, prompt, need_weights=False, key_padding_mask=prompt_padding
        )
        fused = fused + attended.transpose(0, 1)

        # The coarsest level is swapped for the prompt-conditioned tokens -- see the module
        # docstring. Everything finer stays as the vision encoder produced it.
        coarse = levels[-1]
        batch, _, height, width = coarse.shape
        conditioned = list(levels)
        conditioned[-1] = fused[:, : height * width, :].transpose(1, 2).reshape(
            batch, -1, height, width
        )

        pixels = self.pixel_decoder(conditioned)
        return {
            "masks": torch.einsum(
                "bqc,bchw->bqhw", self.mask_embed(queries), self.instance_seg_head(pixels)
            ),
            "semantic": self.semantic_seg_head(pixels),
        }
