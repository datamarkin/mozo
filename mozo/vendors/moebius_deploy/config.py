# SPDX-License-Identifier: Apache-2.0
"""Moebius's geometry, as frozen dataclasses.

Upstream reaches its architecture by ``eval()`` on a string read from a YAML file, then hands the
remaining forty-odd keys to ``diffusers``' ``UNet2DConditionModel``, which fills in whatever the
file omitted. That is fine for a training repository and wrong here: mozo publishes the checkpoint
alone, so the geometry has to be knowable without it, and :mod:`mozo.registry` has to answer "what
variants exist" with no download.

So every number is written out, and **every number below was read off the published checkpoint's
tensor shapes**, not transcribed from ``config/model_cfg/moebius.yaml``. Where the two disagree the
checkpoint wins, and one of them does disagree: the YAML sets
``projection_class_embeddings_input_dim: 2560``, but ``class_embed_type`` is never set, so
``diffusers`` builds no class embedding at all and the checkpoint contains no such tensor. It is
inert. It is recorded here as inert rather than silently dropped, so the next reader does not go
looking for it.

**This model has exactly one shape, and that is a property of the weights.** ``attn2.rel_pos_emb``
is stored with leading dimension ``n²`` for the latent side ``n`` at its level -- 4096, 1024, 256 --
so a differently sized input has nowhere to put its positions. There is no resize, no interpolation
and no flexible-shape mode: a 512x512 image is what the published tensors describe. See
:attr:`Spec.latent` and ``PROVENANCE.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["SPECS", "Spec", "VaeSpec", "get_spec"]


@dataclass(frozen=True)
class VaeSpec:
    """The autoencoder's geometry.

    Upstream names it ``sdvae_f8d4`` and ships it from a second repository
    (``hustvl/PixelHacker``). The config it travels with is Stable Diffusion XL's VAE config with
    ``sample_size`` changed from 1024 to 512; ``scaling_factor`` is SDXL's ``0.13025`` to five
    digits. ``PROVENANCE.md`` records what that means for the licence chain.

    Args:
        in_channels: Pixels in. RGB.
        out_channels: Pixels out.
        latent_channels: The ``d4`` in ``f8d4``. ``encoder.conv_out`` writes twice this -- a mean
            and a log-variance -- and ``quant_conv`` maps those eight channels to eight.
        block_out_channels: Encoder widths, in order. The decoder reverses them.
        layers_per_block: Residual blocks per level. The decoder uses this **plus one**, which is
            why 8 encoder resnets face 12 decoder resnets.
        norm_num_groups: Groups in every ``GroupNorm``.
        scaling_factor: Latents are multiplied by this after encoding and divided by it before
            decoding. Not a normalisation mozo chose -- it is baked into what the diffusion model
            was trained against.
        downsample: ``2 ** (len(block_out_channels) - 1)``. The ``f8``.
    """

    in_channels: int = 3
    out_channels: int = 3
    latent_channels: int = 4
    block_out_channels: tuple[int, ...] = (128, 256, 512, 512)
    layers_per_block: int = 2
    norm_num_groups: int = 32
    scaling_factor: float = 0.13025

    @property
    def downsample(self) -> int:
        """Pixels per latent cell, per side."""
        return 2 ** (len(self.block_out_channels) - 1)


@dataclass(frozen=True)
class Spec:
    """One published geometry.

    Every variant shares this. The four checkpoints upstream publishes differ only in what they
    were fine-tuned on -- their tensors are identically shaped, to the byte -- so ``SPECS`` maps
    every variant to the same object rather than pretending there is a choice to make.

    Args:
        latent: Latent side. 64, and not negotiable: ``attn2.rel_pos_emb`` is stored with leading
            dimension ``latent²`` at the top level.
        in_channels: Nine. Four noisy latent, **one mask**, four masked-image latent, concatenated
            in that order. Getting the order wrong conditions the model on its own noise.
        out_channels: Four. The predicted noise.
        block_out_channels: Level widths. Three, not four -- see ``mid_block``.
        layers_per_block: Residual blocks and attention blocks per level.
        heads: Attention heads, every level. Eight, and it arrives by way of a `documented
            diffusers naming bug <https://github.com/huggingface/diffusers/issues/2011>`_: the
            config says ``attention_head_dim: 8``, and because ``num_attention_heads`` is left
            unset, ``UNet2DConditionModel`` reads that field as the head *count*. The actual head
            dimension is then ``channels // 8`` -- see :meth:`head_dim`.
        cross_attention_dim: Width the conditioning is projected to before the cross-λ reads it.
        encoder_hid_dim: Width of the conditioning embeddings, before projection.
        num_embeddings: Latent Categories Guidance table size. The first half are the conditional
            tokens and the second half the unconditional ones -- see :meth:`conditioning_ids`.
        mix_mlp_ratio: MixFFN hidden width as a multiple of the level width. The gate doubles it.
        local_kernel: Receptive side of the self-λ's positional convolution. Odd, and stored as a
            ``(1, 15, 15)`` 3-D kernel over a depth of one -- which is a 2-D convolution wearing a
            costume. See ``network.py``.
        time_embed_dim: Width of the timestep embedding.
        norm_num_groups: Groups in the ResNet ``GroupNorm``s.
        norm_eps: ``1e-5``. PyTorch's ``GroupNorm`` default, and stated anyway because
            ``LayerNorm``'s is not.
    """

    latent: int = 64
    in_channels: int = 9
    out_channels: int = 4
    block_out_channels: tuple[int, ...] = (320, 640, 1280)
    layers_per_block: int = 2
    heads: int = 8
    cross_attention_dim: int = 768
    encoder_hid_dim: int = 3072
    num_embeddings: int = 20
    mix_mlp_ratio: float = 2.5
    local_kernel: int = 15
    time_embed_dim: int = 1280
    norm_num_groups: int = 32
    norm_eps: float = 1e-5
    vae: VaeSpec = field(default_factory=VaeSpec)

    @property
    def levels(self) -> int:
        """How many resolutions the UNet visits on the way down."""
        return len(self.block_out_channels)

    def head_dim(self, channels: int) -> int:
        """Per-head width at a level *channels* wide.

        40, 80 and 160 for the three published levels. This is the λ layers' ``dim_k``, and it is
        also the width of their key and value projections.
        """
        return channels // self.heads

    @property
    def image_size(self) -> int:
        """The only input side this model accepts. 512."""
        return self.latent * self.vae.downsample

    def latent_sides(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """Latent side at each down level and each up level, as ``(down, up)``.

        ``(64, 32, 16)`` and ``(16, 32, 64)``. The last down level does not downsample and the
        last up level does not upsample, so the walk is not simply ``latent >> i``.

        These are the sizes the cross-λ's ``rel_pos_emb`` is stored for, which is the whole reason
        they are written down rather than derived at construction: a mistake here loads cleanly
        against the wrong tensor only if the mistake happens to be symmetric, and this walk is not.
        """
        down = tuple(self.latent >> min(i, self.levels - 1) for i in range(self.levels))
        return down, tuple(reversed(down))

    @property
    def conditioning_ids(self) -> tuple[tuple[int, ...], tuple[int, ...]]:
        """The Latent Categories Guidance ids, as ``(unconditional, conditional)``.

        Ten each. **The unconditional half is not a null embedding** -- ids 10..19 are ten trained
        vectors, as much a part of the model as ids 0..9. There is no empty prompt to substitute
        and no way to skip the branch.

        Returned unconditional-first because that is the order the batch is built in and the order
        ``chunk(2)`` reads it back: reversed, guidance points the wrong way and still returns a
        plausible picture.
        """
        half = self.num_embeddings // 2
        return tuple(range(half, self.num_embeddings)), tuple(range(half))


#: The one geometry, under every variant name mozo publishes. ``general`` is upstream's
#: ``pretrained``; ``places2`` is its ``ft_places2``. Upstream also publishes ``ft_celebahq`` and
#: ``ft_ffhq``, which are face-specific, identically shaped, and deliberately not carried -- see
#: ``PROVENANCE.md``.
SPECS: dict[str, Spec] = {
    "general": Spec(),
    "places2": Spec(),
}

def get_spec(variant: str) -> Spec:
    """Return the geometry for *variant*.

    Raises:
        ValueError: If *variant* is not published, naming the ones that are.

    Examples:
        >>> get_spec("general").image_size
        512
        >>> get_spec("general").latent_sides()
        ((64, 32, 16), (16, 32, 64))
        >>> get_spec("general").conditioning_ids
        ((10, 11, 12, 13, 14, 15, 16, 17, 18, 19), (0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
        >>> [get_spec("general").head_dim(c) for c in (320, 640, 1280)]
        [40, 80, 160]
        >>> get_spec("xl")
        Traceback (most recent call last):
        ValueError: Unknown variant 'xl' for family 'moebius'. Available: general, places2
    """
    try:
        return SPECS[variant]
    except KeyError:
        available = ", ".join(sorted(SPECS))
        raise ValueError(
            f"Unknown variant {variant!r} for family 'moebius'. Available: {available}") from None
