# SPDX-License-Identifier: Apache-2.0
"""The λ layers, and the two convolution blocks that surround them.

This is the part of Moebius that is Moebius. Where a latent diffusion UNet would put quadratic
self-attention and cross-attention, this puts a pair of **lambda layers**: the keys are softmaxed
over *positions* rather than the queries over keys, contracted with the values into one small
``(dim_k, dim_v)`` matrix, and that matrix is then applied to every query. Cost is linear in the
number of positions, which is what lets a 226M model behave like an 11.9B one.

Upstream writes both layers with ``einops`` and ``torch.einsum``. Neither survives here -- ``einops``
is not in mozo's dependency floor, and ``einsum`` is one of the operators that makes a graph refuse
to convert. Every contraction below is a plain ``matmul`` on an explicitly permuted tensor. That
rewrite is bit-exact and was measured to be, but only once the tensors are **materialised** where
upstream materialises them: ``einops.rearrange`` ends in ``.contiguous()``, a permuted view carries
the same values with a different layout, and torch picks a different vectorised path for each. Left
as views, this layer landed 4.8e-07 from upstream while every one of its contractions was
individually exact.

Four things were found by reading and are worth stating, because two of them delete code:

**The positional convolution is 2-D wearing a 3-D costume.** Upstream builds
``nn.Conv3d(dim_u, dim_k, (1, r, r), padding=(0, r // 2, r // 2))`` and runs it over a tensor whose
"depth" axis is the *value* dimension. The kernel is one deep, so every depth slice is convolved
independently -- which is a batched 2-D convolution, and :func:`fold_positional` performs that
rewrite. It matters because ExecuTorch and CoreML both refuse the 3-D form and accept the 2-D one.

**The fold is algebraically exact and numerically is not**, which is why it is not the default.
Measured on CPU at the published geometry, ``F.conv2d`` on the folded tensor differs from
``nn.Conv3d`` on the original by **2.1e-06** -- the same numbers summed in a different order,
because torch dispatches two- and three-dimensional convolutions to different kernels. So the torch
path runs the 3-D convolution and matches upstream tensor for tensor, and the fold belongs to the
export path, where it is a divergence with a number against its name rather than a free lunch.

**The cross-λ's positional gather is the identity.** Upstream indexes ``rel_pos_emb[n, m]`` where
``n, m`` come from ``meshgrid(arange(N), arange(M))`` -- so ``n[i, j] == i`` and ``m[i, j] == j``,
and the gather returns the tensor it was given. The machinery is inherited from the self-λ's global
branch, where the indices really are relative offsets. Here it is a no-op, and dropping it also
disposes of a latent bug: ``rel_pos`` is a plain attribute rather than a registered buffer, so it
never followed ``.to(device)``.

**The normalisations inside the attention are ``BatchNorm``.** In ``eval()`` they are an affine
scale per channel and perfectly deterministic; in ``train()`` they would rewrite their running
statistics from whatever inference data passed through. Nothing in this package ever calls
``.train()``, and :class:`SelfLambda` and :class:`CrossLambda` are the reason.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["CrossLambda", "DepthwiseSeparableConv", "MixFFN", "SelfLambda",
           "fold_for_export", "fold_positional"]


class DepthwiseSeparableConv(nn.Module):
    """timm's ``DepthwiseSeparableConv``, at the one configuration Moebius builds.

    A depthwise convolution, a norm with an activation, a pointwise convolution, and a norm
    without one. **The activation is ReLU** -- timm's default, which upstream never overrides. It
    is worth naming because every other nonlinearity in this network is SiLU, so ReLU here reads
    like a typo and is not.

    The skip is present exactly when the widths match, which is timm's ``has_skip`` rule and is
    why ``conv_in`` (9 channels to 320) has none while every resnet's pair has one.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.conv_dw = nn.Conv2d(in_channels, in_channels, kernel_size,
                                 padding=kernel_size // 2, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_pw = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.has_skip = in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.bn1(self.conv_dw(x)))
        hidden = self.bn2(self.conv_pw(hidden))
        return hidden + x if self.has_skip else hidden


class _Conv(nn.Module):
    """One convolution under a ``.conv`` attribute, which is how SANA's ``ConvLayer`` stores it.

    Kept as a wrapper purely so the parameter names match the published checkpoint. Upstream's
    ``ConvLayer`` also carries norm and activation slots; at Moebius's configuration all of them
    are ``None``, so what is left is the convolution and the name it lives under.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 1,
                 groups: int = 1, bias: bool = True, activation: bool = False) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              padding=kernel_size // 2, groups=groups, bias=bias)
        #: SiLU or nothing. It carries no parameters, which is exactly why it has to be written
        #: down here: a strict load cannot miss a tensor that was never going to exist, so an
        #: activation left out of a rewrite loads cleanly and answers wrongly.
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.conv(x)
        return F.silu(hidden) if self.activation else hidden


class MixFFN(nn.Module):
    """SANA's ``GLUMBConv``: the feed-forward network, with a gate and a depthwise convolution.

    Widen to ``2 x hidden``, mix spatially with a depthwise 3x3, split the result in half and use
    one half to gate the other, then project back. The gate is why ``inverted_conv`` writes twice
    the hidden width, and the depthwise convolution is why this FFN knows anything about
    neighbouring pixels at all.

    The final projection has **no bias** -- upstream's ``use_bias=(True, True, False)``.
    """

    def __init__(self, channels: int, ratio: float) -> None:
        super().__init__()
        hidden = int(channels * ratio)
        # ``act=("silu", "silu", None)`` upstream: the first convolution is followed by a SiLU,
        # the depthwise one is not, and the projection is not. The second "silu" in that tuple is
        # the GLU gate below, not an activation on ``depth_conv``.
        self.inverted_conv = _Conv(channels, hidden * 2, 1, activation=True)
        self.depth_conv = _Conv(hidden * 2, hidden * 2, 3, groups=hidden * 2)
        self.point_conv = _Conv(hidden, channels, 1, bias=False)

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """*x* is ``(B, H*W, C)``; the spatial shape is passed rather than inferred.

        Upstream recovers it with ``int(N ** 0.5)``, which is correct only while the latent is
        square. It is always square here, but a value that is *derived from an assumption* and a
        value that is *passed* fail differently when the assumption stops holding, and only one of
        them fails loudly.
        """
        batch, positions, channels = x.shape
        hidden = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2)
        hidden = self.depth_conv(self.inverted_conv(hidden))
        hidden, gate = torch.chunk(hidden, 2, dim=1)
        hidden = self.point_conv(hidden * F.silu(gate))
        return hidden.reshape(batch, channels, positions).permute(0, 2, 1)


class _Lambda(nn.Module):
    """What the self and cross layers share: the query side, and the content contraction.

    A **base class**, not a member, so that ``to_q`` and ``norm_q`` sit where the published
    checkpoint puts them -- ``attn1.to_q``, not ``attn1.core.to_q``. Composition would have been
    tidier and would have needed a key remap on load, which is exactly the layer of indirection
    that stops a strict load from being the thing that checks the geometry.

    Both project the query from the spatial map with a 1x1 convolution and normalise it with a
    ``BatchNorm2d``; both softmax the keys over their own length and contract keys with values
    into a single ``(dim_k, dim_v)`` matrix. They differ in where the keys and values come from
    and in how position is handled, which is the whole of each subclass.
    """

    def __init__(self, channels: int, dim_k: int, heads: int, dim_u: int = 1) -> None:
        super().__init__()
        self.heads = heads
        self.dim_k = dim_k
        self.dim_u = dim_u
        self.dim_v = channels // heads
        self.to_q = nn.Conv2d(channels, dim_k * heads, 1, bias=False)
        self.norm_q = nn.BatchNorm2d(dim_k * heads)

    def _queries(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, C, H, W)`` in, ``(B, heads, dim_k, H*W)`` out."""
        batch = x.shape[0]
        query = self.norm_q(self.to_q(x))
        return query.reshape(batch, self.heads, self.dim_k, -1)

    @staticmethod
    def _content(keys: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Contract keys and values into one ``(B, dim_k, dim_v)`` matrix.

        ``keys`` is ``(B, u, dim_k, L)`` and already softmaxed; ``values`` is ``(B, u, dim_v, L)``.
        Summing over ``u`` after the matmul is what upstream's ``b u k m, b u v m -> b k v`` does.
        """
        return torch.matmul(keys, values.transpose(-1, -2)).sum(dim=1)

    @staticmethod
    def _apply_content(queries: torch.Tensor, content: torch.Tensor) -> torch.Tensor:
        """``(B, heads, dim_v, L)`` -- every query read through the one content matrix."""
        return torch.matmul(content.transpose(-1, -2).unsqueeze(1), queries)

    def _pack(self, y: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """``(B, heads, dim_v, H*W)`` back to ``(B, H, W, heads*dim_v)``."""
        batch = y.shape[0]
        packed = y.reshape(batch, self.heads * self.dim_v, height, width)
        return packed.permute(0, 2, 3, 1).contiguous()


class SelfLambda(_Lambda):
    """Self-attention as a lambda layer, with a **local** positional term.

    Keys and values come from the same spatial map as the queries. Position is handled by
    convolving the values with a ``r x r`` kernel -- upstream's ``r = 15`` -- which is where this
    layer gets its locality, and is the only reason the model knows that neighbouring latent cells
    are neighbours.

    Args:
        channels: Width of the incoming map.
        dim_k: Key width. Per level: 40, 80, 160.
        heads: Query heads. Eight everywhere.
        kernel: Positional receptive side. Odd.
        dim_u: Upstream's "intra-depth". One in every published checkpoint.
    """

    def __init__(self, channels: int, dim_k: int, heads: int, kernel: int = 15,
                 dim_u: int = 1) -> None:
        if kernel % 2 == 0:
            raise ValueError(f"positional kernel must be odd, got {kernel}")
        super().__init__(channels, dim_k, heads, dim_u)
        self.to_k = nn.Conv2d(channels, dim_k * dim_u, 1, bias=False)
        self.to_v = nn.Conv2d(channels, self.dim_v * dim_u, 1, bias=False)
        self.norm_v = nn.BatchNorm2d(self.dim_v * dim_u)
        # Stored 3-D to match the published tensor -- ``(dim_k, dim_u, 1, r, r)`` -- and folded to
        # 2-D at every forward. See the module docstring: the length-one depth axis makes the two
        # forms the same function, and only the 2-D one converts to a mobile graph.
        self.pos_conv = nn.Conv3d(dim_u, dim_k, (1, kernel, kernel),
                                  padding=(0, kernel // 2, kernel // 2))

    def forward(self, x: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """*x* is ``(B, H*W, C)``. Returns the same shape."""
        batch, positions, channels = x.shape
        # ``.contiguous()`` is not cosmetic: upstream's ``_rearrange`` materialises here,
        # and a permuted view sends ``conv2d`` down a different BLAS path. 4.8e-07.
        spatial = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()

        queries = self._queries(spatial)
        keys = self.to_k(spatial).reshape(batch, self.dim_u, self.dim_k, positions)
        values = self.norm_v(self.to_v(spatial)).reshape(batch, self.dim_u, self.dim_v, positions)

        content = self._apply_content(queries, self._content(keys.softmax(dim=-1), values))

        # The 3-D convolution as published, unless :func:`fold_for_export` has swapped it for the
        # 2-D one. Branching on the weight's rank rather than on a flag: the fold *is* the state,
        # and a flag could disagree with the module it describes.
        if self.pos_conv.weight.dim() == 5:
            grid = values.reshape(batch, self.dim_u, self.dim_v, height, width)
            position = self.pos_conv(grid).reshape(batch, self.dim_k, self.dim_v, positions)
        else:
            grid = values.reshape(batch * self.dim_v, self.dim_u, height, width)
            position = self.pos_conv(grid).reshape(batch, self.dim_v, self.dim_k, positions)
            position = position.transpose(1, 2)

        # ``b h k n, b k v n -> b h v n``: a per-position matmul, batched over (B, n).
        positional = torch.matmul(queries.permute(0, 3, 1, 2), position.permute(0, 3, 1, 2))
        positional = positional.permute(0, 2, 3, 1)

        packed = self._pack(content + positional, height, width)
        return packed.reshape(batch, positions, -1)


def fold_for_export(module: nn.Module) -> nn.Module:
    """Replace every :class:`SelfLambda`'s 3-D positional convolution with the folded 2-D one.

    In place, and only for export: ExecuTorch, ONNX and CoreML all handle a ``Conv3d`` far worse
    than the ``Conv2d`` it algebraically is -- when they accept it at all. The cost is 2.1e-06 on
    CPU, which is why the torch path does not do this and why an exported graph is checked against
    the *unfolded* torch model with a tolerance rather than with ``torch.equal``.

    Returns the same module, so it reads as a conversion step at the call site.
    """
    for child in module.modules():
        if isinstance(child, SelfLambda) and child.pos_conv.weight.dim() == 5:
            child.pos_conv = fold_positional(child)
    return module


def fold_positional(layer: SelfLambda) -> nn.Conv2d:
    """Return *layer*'s positional convolution as the 2-D convolution it algebraically is.

    ``nn.Conv3d(u, k, (1, r, r), padding=(0, p, p))`` over ``(B, u, D, H, W)`` convolves each of
    the ``D`` depth slices independently, because the kernel is one deep. Folding ``D`` into the
    batch and running ``nn.Conv2d(u, k, (r, r), padding=(p, p))`` computes the same function, and
    the weight is the same numbers with a length-one axis dropped.

    Use it for export: ExecuTorch and CoreML both reject the 3-D operator. Do **not** use it on the
    torch path -- the two forms sum in a different order and land 2.1e-06 apart, which is a
    divergence worth taking deliberately for a mobile graph and not worth taking for nothing.

    Examples:
        >>> folded = fold_positional(SelfLambda(320, 40, 8))
        >>> folded.kernel_size, folded.padding
        ((15, 15), (7, 7))
        >>> tuple(folded.weight.shape)
        (40, 1, 15, 15)
    """
    source = layer.pos_conv
    folded = nn.Conv2d(source.in_channels, source.out_channels,
                       source.kernel_size[1:], padding=source.padding[1:])
    with torch.no_grad():
        folded.weight.copy_(source.weight.squeeze(2))
        folded.bias.copy_(source.bias)
    return folded


class CrossLambda(_Lambda):
    """Cross-attention as a lambda layer, with a **global** positional term.

    Keys and values come from the conditioning sequence -- ten Latent Categories Guidance
    embeddings, projected to ``cross_attention_dim`` -- so they are ``nn.Linear`` here rather than
    convolutions, and the value norm is ``BatchNorm1d`` over a sequence rather than a map.

    Position is a learned table, ``(H*W, sequence, dim_k, dim_u)``, and **that is what freezes this
    model to one input size**: the table has a row per latent position and no way to make more.

    Args:
        channels: Width of the incoming map.
        dim_k: Key width.
        heads: Query heads.
        cross_dim: Width of the conditioning after projection. 768.
        positions: ``H * W`` at this level. 4096, 1024 or 256.
        sequence: Conditioning length. Ten.
        dim_u: One.
    """

    def __init__(self, channels: int, dim_k: int, heads: int, cross_dim: int,
                 positions: int, sequence: int, dim_u: int = 1) -> None:
        super().__init__(channels, dim_k, heads, dim_u)
        self.to_k = nn.Linear(cross_dim, dim_k * dim_u, bias=False)
        self.to_v = nn.Linear(cross_dim, self.dim_v * dim_u, bias=False)
        self.norm_v = nn.BatchNorm1d(self.dim_v * dim_u)
        self.rel_pos_emb = nn.Parameter(torch.zeros(positions, sequence, dim_k, dim_u))

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor,
                height: int, width: int) -> torch.Tensor:
        """*x* is ``(B, H*W, C)``, *conditioning* is ``(B, sequence, cross_dim)``."""
        batch, positions, channels = x.shape
        # ``.contiguous()`` is not cosmetic: upstream's ``_rearrange`` materialises here,
        # and a permuted view sends ``conv2d`` down a different BLAS path. 4.8e-07.
        spatial = x.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()

        queries = self._queries(spatial)
        # Materialised before the norm, again: upstream's ``b l c -> b c l`` ends in
        # ``.contiguous()``, and ``BatchNorm1d`` reduces a transposed view along a different path.
        keys = self.to_k(conditioning).transpose(1, 2).contiguous()
        keys = keys.reshape(batch, self.dim_u, self.dim_k, -1)
        values = self.norm_v(self.to_v(conditioning).transpose(1, 2).contiguous())
        values = values.reshape(batch, self.dim_u, self.dim_v, -1)

        content = self._apply_content(queries, self._content(keys.softmax(dim=-1), values))

        # ``n m k u, b u v m -> b n k v``. The identity gather upstream performs here is dropped;
        # see the module docstring.
        table = self.rel_pos_emb.squeeze(-1).permute(0, 2, 1).unsqueeze(0)
        position = torch.matmul(table, values.squeeze(1).transpose(-1, -2).unsqueeze(1))

        # ``b h k n, b n k v -> b h v n``.
        positional = torch.matmul(queries.permute(0, 3, 1, 2), position)
        positional = positional.permute(0, 2, 3, 1)

        packed = self._pack(content + positional, height, width)
        return packed.reshape(batch, positions, -1)
