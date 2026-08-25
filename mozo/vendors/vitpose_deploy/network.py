# SPDX-License-Identifier: Apache-2.0
"""The trunk and the head: a crop in, one heatmap per joint out.

ViTPose is a plain ViT with two changes, both of which the upstream file says out loud: the patch
embedding pads by 2, and the MLP is a mixture of experts. There is no pose-specific architecture
below the head -- the paper's claim is that a plain transformer, given enough scale, needs none.

The head is the classic two-deconvolution decoder. Upstream also carries a simple one (ReLU,
bilinear upsample, 3x3 convolution), selected by ``use_simple_decoder``. No published variant here
sets it, so it is not built: a branch that no checkpoint can take is a branch nothing tests.
"""

from __future__ import annotations

import torch
from torch import nn

from .config import Spec, get_spec
from .layers import Layer

__all__ = ["ClassicDecoder", "Embeddings", "Encoder", "Backbone", "VitPose"]


class Embeddings(nn.Module):
    """Patches, plus a position embedding sized for the grid.

    Two oddities, both upstream's and both load-bearing. The patch convolution pads by **2**, so
    it is not the clean non-overlapping tiling a ViT usually has. And the position embedding
    carries an extra leading row for a class token this model does not have: the token is absent,
    but its position embedding is added to *every* patch, so the parameter cannot simply be
    trimmed to the grid.
    """

    def __init__(self, spec: Spec):
        super().__init__()
        rows, columns = spec.grid
        self.patch_embeddings = _PatchEmbeddings(spec)
        self.position_embeddings = nn.Parameter(torch.zeros(1, rows * columns + 1, spec.hidden))

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embeddings(pixel_values)
        return patches + self.position_embeddings[:, 1:] + self.position_embeddings[:, :1]


class _PatchEmbeddings(nn.Module):
    """The strided convolution that turns pixels into patch tokens."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.projection = nn.Conv2d(
            3, spec.hidden, kernel_size=spec.patch, stride=spec.patch, padding=spec.patch_padding
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.projection(pixel_values).flatten(2).transpose(1, 2)


class Encoder(nn.Module):
    """The stack of blocks."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.layer = nn.ModuleList(Layer(spec) for _ in range(spec.layers))

    def forward(self, hidden: torch.Tensor, expert: int) -> torch.Tensor:
        for block in self.layer:
            hidden = block(hidden, expert)
        return hidden


class Backbone(nn.Module):
    """Embeddings, blocks, and the final norm.

    Upstream is a general backbone that can hand back any stage; every published checkpoint asks
    for the last one only, so this returns that and nothing else. The stage machinery it drops
    carried no weights.
    """

    def __init__(self, spec: Spec):
        super().__init__()
        self.embeddings = Embeddings(spec)
        self.encoder = Encoder(spec)
        self.layernorm = nn.LayerNorm(spec.hidden, eps=spec.layer_norm_eps)

    def forward(self, pixel_values: torch.Tensor, expert: int) -> torch.Tensor:
        hidden = self.encoder(self.embeddings(pixel_values), expert)
        return self.layernorm(hidden)


class ClassicDecoder(nn.Module):
    """Two deconvolution blocks and a 1x1 convolution, turning features into heatmaps."""

    def __init__(self, spec: Spec):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(spec.hidden, 256, 4, stride=2, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.relu2 = nn.ReLU()
        self.conv = nn.Conv2d(256, spec.keypoints, kernel_size=1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        hidden = self.relu1(self.batchnorm1(self.deconv1(features)))
        hidden = self.relu2(self.batchnorm2(self.deconv2(hidden)))
        return self.conv(hidden)


class VitPose(nn.Module):
    """One ViTPose++ variant: a batch of person crops in, ``(N, K, H, W)`` heatmaps out.

    Args:
        variant: Which published geometry. See :data:`~.config.SPECS`.

    Attributes:
        spec: The geometry in use.
    """

    def __init__(self, variant: str = "base"):
        super().__init__()
        self.spec = get_spec(variant)
        self.backbone = Backbone(self.spec)
        self.head = ClassicDecoder(self.spec)

    def forward(self, pixel_values: torch.Tensor, expert: int = 0) -> torch.Tensor:
        """Run a batch of crops.

        Args:
            pixel_values: ``(N, 3, 256, 192)``, normalised.
            expert: Which dataset expert the mixture-of-experts blocks should use. Always 0 in
                mozo -- see :mod:`~.predictor`.
        """
        hidden = self.backbone(pixel_values, expert)
        rows, columns = self.spec.grid
        features = hidden.permute(0, 2, 1).reshape(hidden.shape[0], -1, rows, columns)
        return self.head(features.contiguous())
