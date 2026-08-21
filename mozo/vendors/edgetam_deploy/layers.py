# SPDX-License-Identifier: Apache-2.0
"""The two small modules the rest of this package is built out of.

Taken verbatim from ``sam2/modeling/sam2_utils.py`` in ``facebookresearch/EdgeTAM``, which is
itself SAM 2's file unchanged. Only ``LayerNorm2d`` and ``MLP`` are carried: everything else in
that module either belongs to the video tracker (``select_closest_cond_frames``,
``get_1d_sine_pe``) or samples training clicks (``sample_box_points``,
``sample_random_points_from_errors``, ``sample_one_point_from_error_center``, ``get_next_point``),
and the latter group was the only thing importing OpenCV and ``mask_to_box``.

``DropPath`` is left behind too, which is where this differs from SAM 2's own extraction. SAM 2's
Hiera trunk uses it; EdgeTAM's RepViT trunk does not, and nothing else on the image path ever did.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["LayerNorm2d", "MLP"]


class LayerNorm2d(nn.Module):
    """Layer norm over the channel axis of an ``NCHW`` tensor.

    Args:
        num_channels: Channels to normalise over.
        eps: Added to the variance before the square root.
    """

    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class MLP(nn.Module):
    """A stack of linear layers with the activation between them but not after.

    Args:
        input_dim: Width in.
        hidden_dim: Width of every layer but the last.
        output_dim: Width out.
        num_layers: How many linear layers, counting the output one.
        activation: Applied between layers.
        sigmoid_output: Squash the result into ``(0, 1)``. The IoU head uses this.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        activation: type[nn.Module] = nn.ReLU,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = self.act(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
