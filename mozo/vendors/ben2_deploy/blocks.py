# ------------------------------------------------------------------------
# BEN2 -- Background Erase Network
# Copyright (c) 2025 Prama LLC. All Rights Reserved.
# Licensed under the MIT License [see LICENSE for details]
# ------------------------------------------------------------------------
"""The decoder's two cross-attention blocks, and the tensor plumbing they sit on.

Extracted from ``BEN2.py`` lines 618-842. ``einops.rearrange`` is gone -- every call had a fixed
pattern with literal group sizes, so each becomes a ``view``/``permute``/``reshape``. That is
worth stating twice: the rewrites are the highest-risk change in this package, because a wrong
axis order reassembles the four quadrants into the wrong corners and produces a matte that still
looks like a matte. ``tools/verify/ben2.py`` checks all nine against ``einops`` directly.

**MCLM and MCRM are stateless here.** Upstream stores ``p_poses`` and ``g_pos`` on the module and
resets both to empty at the top of every ``forward``, so the cache never survives a call. Holding
them as locals is the same computation with the mutable module state removed -- §5's "mutable
module globals become spec fields when vendored", applied to attributes that were never read
across calls in the first place.
"""

from __future__ import annotations

__all__ = ["MCLM", "MCRM", "image2patches", "make_cbg", "make_cbr", "patches2image",
           "rescale_to", "resize_as"]

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_cbr(in_dim: int, out_dim: int) -> nn.Sequential:
    """Conv-InstanceNorm-GELU.

    ``nn.InstanceNorm2d`` defaults to ``affine=False, track_running_stats=False``, so it carries
    no parameters and the strict load has nothing to check here. That is why the widths are
    written down in ``config.py`` instead of being read back off the checkpoint.
    """
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         nn.InstanceNorm2d(out_dim), nn.GELU())


def make_cbg(in_dim: int, out_dim: int) -> nn.Sequential:
    """Identical to :func:`make_cbr`.

    Upstream defines both and uses each at different call sites. They are kept as two names
    because the checkpoint's keys are ``upsample1``/``upsample2`` versus ``conv1``..``output5``,
    and collapsing them would invite someone to collapse the call sites too.
    """
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1),
                         nn.InstanceNorm2d(out_dim), nn.GELU())


def rescale_to(x: torch.Tensor, scale_factor: float = 2, interpolation: str = "nearest") -> torch.Tensor:
    """Scale by a factor. Default ``nearest``, and the default is used -- see ``network.py``."""
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x: torch.Tensor, y: torch.Tensor, interpolation: str = "bilinear") -> torch.Tensor:
    """Resize *x* to *y*'s spatial size. Default ``bilinear``, unlike :func:`rescale_to`."""
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)


def image2patches(x: torch.Tensor) -> torch.Tensor:
    """``(b, c, 2h, 2w)`` -> ``(4b, c, h, w)``, ordered ``(hg wg b)``.

    Was ``rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)``.
    """
    b, c, H, W = x.shape
    return (x.view(b, c, 2, H // 2, 2, W // 2)
             .permute(2, 4, 0, 1, 3, 5).reshape(4 * b, c, H // 2, W // 2))


def patches2image(x: torch.Tensor) -> torch.Tensor:
    """``(4b, c, h, w)`` -> ``(b, c, 2h, 2w)``. The exact inverse of :func:`image2patches`.

    Was ``rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)``.
    """
    n, c, h, w = x.shape
    b = n // 4
    return (x.view(2, 2, b, c, h, w)
             .permute(2, 3, 0, 4, 1, 5).reshape(b, c, 2 * h, 2 * w))


def _bchw_to_hwbc(x: torch.Tensor) -> torch.Tensor:
    """``(b, c, h, w)`` -> ``(h*w, b, c)``. Was ``rearrange(x, 'b c h w -> (h w) b c')``."""
    b, c, h, w = x.shape
    return x.permute(2, 3, 0, 1).reshape(h * w, b, c)


def _pool(x: torch.Tensor, target: tuple[int, int]) -> torch.Tensor:
    """Average-pool *x* down to *target*, taking the exportable route when it is available.

    ``F.adaptive_avg_pool2d`` is what upstream writes and what this reduces to, but ONNX cannot
    lower it once the tracer has lost the input's static shape -- which happens here, because
    ``image2patches`` reaches the pooled tensor through a reshape with computed sizes. The error
    is ``Unsupported: ONNX export of operator adaptive_avg_pool2d, input size not accessible``.

    When the input divides evenly by the target, adaptive pooling *is* average pooling with
    ``kernel = stride = in // out``: the window for output *i* is ``[floor(i*in/out),
    ceil((i+1)*in/out))``, which collapses to ``[i*k, (i+1)*k)`` exactly. Every ratio in this
    model divides evenly -- ``config.py`` fixes the input at 1024 and the pooling grids follow
    from it -- so the substitution is exact rather than approximate, and
    ``tools/verify/ben2.py`` checks it at every one. The decoder makes fifteen pooling calls over
    twelve distinct ``(shape, target)`` pairs, the last rung repeating the one before it.

    This is §6 of ``plans/vendoring.md`` going the other way from EasyOCR, where the only
    substitution that traced was a mean over the same axis and *was* different in float (the pool
    divides by three, the mean multiplies by a reciprocal), so it was reverted. Here the two are
    bit-identical, so the fast path is taken whenever it applies and upstream's operator is kept
    for the case that would make it a guess.
    """
    # ``int()`` rather than using the shape entries directly: under a trace they are Tensors, and
    # the whole point here is to resolve them to constants. That is sound because the model has
    # exactly one input size -- see ``config.INPUT`` -- and unsound the moment that stops being
    # true, which is why ``INPUT`` is a constant rather than an argument.
    height, width = int(x.shape[-2]), int(x.shape[-1])
    if height % target[0] == 0 and width % target[1] == 0:
        kernel = (height // target[0], width // target[1])
        return F.avg_pool2d(x, kernel_size=kernel, stride=kernel)
    return F.adaptive_avg_pool2d(x, target)


class PositionEmbeddingSine:
    """Sine positional encoding over a feature grid.

    A plain class rather than an ``nn.Module``, as upstream has it -- it holds no parameters and
    nothing in the checkpoint corresponds to it. Unlike upstream it builds ``dim_t`` per call on
    the target device instead of caching a CPU tensor and moving it every time, because a bare
    tensor attribute on a non-Module is invisible to ``.to()`` and is exactly the kind of state
    that goes stale when a model is moved after construction.

    The encoding is always computed in float32, as upstream does, whatever dtype the features
    carry. Under an autocast that makes the subsequent add promote; that is upstream's behaviour.
    """

    def __init__(self, num_pos_feats: int = 64, temperature: int = 10000,
                 normalize: bool = False, scale: float | None = None) -> None:
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def __call__(self, b: int, h: int, w: int, device=None) -> torch.Tensor:
        dim_t_base = torch.arange(0, self.num_pos_feats, dtype=torch.float32, device=device)

        # The "mask" is all-zero, so not_mask is all-ones and the cumsums are just 1..h and 1..w
        # broadcast. Upstream builds it this way because the block descends from DETR, where the
        # mask is real. Kept, because collapsing it to an arange changes the float accumulation.
        mask = torch.zeros([b, h, w], dtype=torch.bool, device=device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = self.temperature ** (2 * (dim_t_base // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class MCLM(nn.Module):
    """Multi-scale cross-level module: the global view attends to the four quadrants, then back.

    Runs once, on the deepest feature map. Five ``nn.MultiheadAttention`` modules -- one for the
    global-to-local direction and one per quadrant coming back.

    **``need_weights`` is left at its default ``True``.** Every call here is
    ``self.attention[i](q, k, v)[0]``, discarding the weights. That default takes torch's unfused
    branch, where the query is scaled before the matmul and ``scaled_dot_product_attention`` is
    never reached -- so pinning an attention implementation does not govern this block, and
    "simplifying" it to SDPA is a real numerical change. Left exactly as upstream has it.
    """

    def __init__(self, d_model: int, num_heads: int, pool_ratios: tuple[int, ...] = (1, 4, 8)) -> None:
        super().__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1) for _ in range(5)])

        self.linear1 = nn.Linear(d_model, d_model * 2)
        self.linear2 = nn.Linear(d_model * 2, d_model)
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = F.gelu
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)

    def forward(self, l: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """*l* is ``(4, c, h, w)``, *g* is ``(1, c, h, w)``. Returns ``(5, c, h, w)``."""
        device = l.device
        b, c, h, w = l.size()
        concated_locs = patches2image(l)  # (1, c, 2h, 2w)

        pools, poses = [], []
        for pool_ratio in self.pool_ratios:
            # Note the asymmetry, faithfully reproduced: the target size is derived from the
            # *quadrant* height while the tensor being pooled is the reassembled 2h x 2w image.
            # So ratio 1 halves rather than preserving. Deriving tgt from concated_locs instead
            # is the obvious reading and pools the wrong grid at every ratio.
            tgt_hw = (round(int(h) / pool_ratio), round(int(w) / pool_ratio))
            pool = _pool(concated_locs, tgt_hw)
            pools.append(_bchw_to_hwbc(pool))
            poses.append(_bchw_to_hwbc(
                self.positional_encoding(pool.shape[0], pool.shape[2], pool.shape[3], device)))

        pools = torch.cat(pools, 0)
        p_poses = torch.cat(poses, dim=0)
        g_pos = _bchw_to_hwbc(self.positional_encoding(g.shape[0], g.shape[2], g.shape[3], device))

        # Global query attends to the pooled locals.
        g_hw_b_c = _bchw_to_hwbc(g)
        g_hw_b_c = g_hw_b_c + self.dropout1(
            self.attention[0](g_hw_b_c + g_pos, pools + p_poses, pools)[0])
        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = g_hw_b_c + self.dropout2(
            self.linear2(self.dropout(self.activation(self.linear1(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)

        # Each quadrant then attends to its own slice of the refreshed global.
        l_hw_b_c = _bchw_to_hwbc(l)
        gb = g_hw_b_c.shape[1]
        _g = g_hw_b_c.view(h, w, gb, c)
        # '(ng h) (nw w) b c -> (h w) (ng nw b) c', ng=2, nw=2
        _g = (_g.view(2, h // 2, 2, w // 2, gb, c)
                .permute(1, 3, 0, 2, 4, 5).reshape((h // 2) * (w // 2), 4 * gb, c))

        outputs_re = []
        for i, (_l, _gi) in enumerate(zip(l_hw_b_c.chunk(4, dim=1), _g.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _gi, _gi)[0])
        outputs_re = torch.cat(outputs_re, 1)

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(
            self.linear4(self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)

        out = torch.cat((l_hw_b_c, g_hw_b_c), 1)  # (h*w, 5, c)
        return out.view(h, w, out.shape[1], c).permute(2, 3, 0, 1).contiguous()


class MCRM(nn.Module):
    """Multi-scale cross-refinement module: one decoder rung.

    The global map produces a saliency gate that multiplies the quadrants, each quadrant attends
    to a pooled copy of its own region of the global, and the refreshed quadrants are added back
    into the global. Returns the stacked ``(5, c, h, w)`` and the gate, which upstream calls
    ``token_attention_map`` and then discards at every call site.
    """

    def __init__(self, d_model: int, num_heads: int, pool_ratios: tuple[int, ...] = (4, 8, 16)) -> None:
        super().__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1) for _ in range(4)])
        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = F.gelu
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """*x* is ``(5, c, h, w)`` -- four quadrants then the global."""
        b, c, h, w = x.size()
        loc, glb = x.split([4, 1], dim=0)

        patched_glb = image2patches(glb)

        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = F.interpolate(
            token_attention_map, size=patches2image(loc).shape[-2:], mode="nearest")
        loc = loc * image2patches(token_attention_map)

        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(int(h) / pool_ratio), round(int(w) / pool_ratio))
            pools.append(_pool(patched_glb, tgt_hw).flatten(2))  # 'nl c h w -> nl c (h w)'

        # 'nl c nphw -> nl nphw 1 c'
        pooled = torch.cat(pools, 2).permute(0, 2, 1).unsqueeze(2)
        # 'nl c h w -> nl (h w) 1 c'
        loc_ = loc.flatten(2).permute(0, 2, 1).unsqueeze(2)

        outputs = []
        for i, q in enumerate(loc_.unbind(dim=0)):
            v = pooled[i]
            outputs.append(self.attention[i](q, v, v)[0])
        outputs = torch.cat(outputs, 1)

        src = loc.view(4, c, -1).permute(2, 0, 1) + self.dropout1(outputs)
        src = self.norm1(src)
        src = src + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(src)).clone())))
        src = self.norm2(src)
        src = src.permute(1, 2, 0).reshape(4, c, h, w)

        glb = glb + F.interpolate(patches2image(src), size=glb.shape[-2:], mode="nearest")

        return torch.cat((src, glb), 0), token_attention_map
