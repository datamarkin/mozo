#!/usr/bin/env python3
"""Check that mozo's Moebius returns exactly what the published model does.

**The reference is the authors' own code**, ``hustvl/Moebius`` driven through
``removal/v1_2/pipeline.py``, standing on ``diffusers`` for its autoencoder, its scheduler and the
UNet scaffolding its model subclasses. There is no port and no second implementation, so the chain
is one link long: these checkpoints reproduce this code and no other.

**The comparison is exact.** ``torch.equal``, no tolerance, because a tolerance hides precisely the
drift a gate exists to catch. Two of the things this package had to get right would have been
swallowed by any sane tolerance: every ``GroupNorm`` in the autoencoder uses ``eps=1e-6`` rather
than PyTorch's ``1e-5``, and the encoder's downsample pads ``(0, 1, 0, 1)`` before a stride-2
convolution with no padding of its own -- which produces the same *shape* as
``Conv2d(3, stride=2, padding=1)`` and different numbers.

**What is pinned, and why.**

*The device is the CPU.* mozo's published fp32 artifact matches upstream tensor for tensor there.

*The checkpoints are cast to fp32 on both sides.* The autoencoder ships in half precision -- 167 MB
for 83.65M parameters -- and a comparison between an fp16 path and an fp32 one measures the cast.

*The reference's parameters are re-allocated before comparing.* ``from_config`` places tensors at
whatever offset the file had, and BLAS picks different vectorised paths for storage that happens to
be page-aligned; OWLv2's gate measured 1.2e-07 on that alone.

*The draw is not compared, the distribution is.* Upstream's encoder is stochastic --
``latent_dist.sample()``, not ``.mode()`` -- so what is held against the reference is the mean, the
log-variance and the standard deviation. Comparing samples would compare two random number
generators. See ``mozo/vendors/moebius_deploy/vae.py``.

**The gate is falsifiable and has been falsified.** ``--falsify`` perturbs one constant at a time
and confirms the run fails, and fails *at the stage that constant reaches*. A gate that has never
failed has not been shown to work.

    python tools/verify/moebius.py --weights path/to/weights
    python tools/verify/moebius.py --weights path/to/weights --falsify
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from mozo.vendors.moebius_deploy import vae as vae_module  # noqa: E402
from mozo.vendors.moebius_deploy.config import VaeSpec  # noqa: E402
from mozo.vendors.moebius_deploy.vae import AutoencoderKL  # noqa: E402

#: The environment this parity was established in. Printed on every run, because a reference is a
#: version as much as it is a repository -- ``diffusers`` has refactored its VAE blocks before.
PINNED = {"torch": "2.11.0", "diffusers": "0.38.0"}

#: Fixed, because a gate that draws a different image each run cannot be compared across runs.
SEED = 0


def versions() -> str:
    """The versions actually in use, flagged where they differ from :data:`PINNED`."""
    import diffusers

    actual = {"torch": torch.__version__, "diffusers": diffusers.__version__}
    return "  ".join(f"{name}={value}{'' if PINNED.get(name) == value else ' (pinned ' + PINNED[name] + ')'}"
                     for name, value in actual.items())


def load(weights: Path) -> tuple[dict, dict]:
    """The autoencoder's fp32 state dict and the config it travels with."""
    raw = torch.load(weights / "diffusion_pytorch_model.bin", map_location="cpu",
                     weights_only=True)
    return {k: v.float() for k, v in raw.items()}, json.loads((weights / "config.json").read_text())


def reference(state: dict, config: dict):
    """``diffusers``' own autoencoder, loaded strict and re-allocated."""
    from diffusers import AutoencoderKL as RefVAE

    model = RefVAE.from_config(config)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.data = parameter.data.clone()
    return model


def vae_stages(weights: Path) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """Every autoencoder stage, as ``(name, mine, reference)``.

    Structured as a list rather than printed in place so that :func:`falsify` can run the same
    comparisons under a perturbation and report *which* stage moved.
    """
    state, config = load(weights)
    ref = reference(state, config)
    mine = AutoencoderKL(VaeSpec())
    mine.load_state_dict(state, strict=True)
    mine.eval()

    torch.manual_seed(SEED)
    image = torch.randn(1, 3, 512, 512)

    stages: list[tuple[str, torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        stages.append(("encoder trunk", mine.encoder(image), ref.encoder(image)))

        got, want = mine.encode(image), ref.encode(image).latent_dist
        stages.append(("latent mean", got.mean, want.mean))
        stages.append(("latent logvar", got.logvar, want.logvar))
        stages.append(("latent std", got.std, want.std))

        latent = got.mode()
        # Each computed once and reused: both sides are fed the same tensor on purpose, so a
        # second evaluation would only cost a 512x512 pass and risk the two drifting.
        trunk = _trunk(mine.encoder, image)
        stages.append(("encoder mid block",
                       mine.encoder.mid_block(trunk), ref.encoder.mid_block(trunk)))
        mine_in, ref_in = mine.decoder.conv_in(latent), ref.decoder.conv_in(latent)
        stages.append(("decoder conv_in", mine_in, ref_in))
        stages.append(("decoder mid block",
                       mine.decoder.mid_block(mine_in), ref.decoder.mid_block(ref_in)))
        # The same operation on both sides. ``mine.decode`` divides by the scaling factor, so the
        # reference is handed a latent that has been through the same multiply and divide -- else
        # this measures ``z * s / s``, which is not ``z``, and calls it drift.
        scale = VaeSpec().scaling_factor
        stages.append(("decode", mine.decode(mine.scale(latent)),
                       ref.decode(latent * scale / scale).sample))
    return stages


def lambda_stages(source: Path) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """The two λ layers, against upstream's own modules, at random weights.

    Gated before any checkpoint exists, because a layer that is wrong at random weights is wrong
    at trained ones and this failure is far easier to read. *source* is a checkout of
    ``hustvl/Moebius``; it is not vendored and not shipped, and this is the only thing that needs
    it.

    **Upstream's package ``__init__`` imports the PixelHacker teacher**, which is the sole importer
    of ``flash-linear-attention`` -- a CUDA-only dependency that cannot install on this machine.
    The teacher is a distillation artifact and reaches none of Moebius's forward path, so the
    import is stubbed rather than satisfied. That the student then builds with neither ``fla`` nor
    ``transformers`` in ``sys.modules`` is the evidence for the claim in ``PROVENANCE.md``.
    """
    sys.path.insert(0, str(source))
    from model_lib.nets.layers.λ.vanillaλ import MQCλ, MQSλ  # noqa: E402

    from mozo.vendors.moebius_deploy.attention import CrossLambda, SelfLambda  # noqa: E402
    from mozo.vendors.moebius_deploy.config import get_spec  # noqa: E402

    spec = get_spec("general")
    channels = spec.block_out_channels[0]
    dim_k, side = spec.head_dim(channels), spec.latent
    sequence = spec.num_embeddings // 2

    torch.manual_seed(SEED)
    x = torch.randn(2, side * side, channels)
    conditioning = torch.randn(2, sequence, spec.cross_attention_dim)

    ref_self = MQSλ(dim=channels, dim_k=dim_k, n=side, r=spec.local_kernel, heads=spec.heads,
                    dim_out=channels, dim_u=1).eval()
    got_self = SelfLambda(channels, dim_k, spec.heads, kernel=spec.local_kernel).eval()
    for mine, theirs in ((got_self.to_q, ref_self.to_q), (got_self.to_k, ref_self.to_k),
                         (got_self.to_v, ref_self.to_v), (got_self.norm_q, ref_self.norm_q),
                         (got_self.norm_v, ref_self.norm_v),
                         (got_self.pos_conv, ref_self.pos_conv)):
        mine.load_state_dict(theirs.state_dict())

    ref_cross = MQCλ(dim=channels, dim_k=dim_k, dim_cross=spec.cross_attention_dim, n=side,
                     m=sequence, heads=spec.heads, dim_out=channels, dim_u=1).eval()
    got_cross = CrossLambda(channels, dim_k, spec.heads, spec.cross_attention_dim,
                            positions=side * side, sequence=sequence).eval()
    for mine, theirs in ((got_cross.to_q, ref_cross.to_q), (got_cross.to_k, ref_cross.to_k),
                         (got_cross.to_v, ref_cross.to_v),
                         (got_cross.norm_q, ref_cross.norm_q),
                         (got_cross.norm_v, ref_cross.norm_v)):
        mine.load_state_dict(theirs.state_dict())
    got_cross.rel_pos_emb.data.copy_(ref_cross.rel_pos_emb.data)

    with torch.no_grad():
        stages = [("self-λ (local r=15)", got_self(x, side, side), ref_self(x)),
                  ("cross-λ (global)", got_cross(x, conditioning, side, side),
                   ref_cross(x, conditioning))]
        # The export rewrite, measured rather than asserted. Not a failure: this is the number
        # that says what a mobile graph costs. See ``fold_positional``.
        from mozo.vendors.moebius_deploy.attention import fold_positional
        values = torch.randn(2, 1, got_self.dim_v, side, side)
        folded = fold_positional(got_self)
        as_2d = folded(values.reshape(2 * got_self.dim_v, 1, side, side))
        as_2d = as_2d.reshape(2, got_self.dim_v, dim_k, side, side).transpose(1, 2)
        stages.append(("Conv3d vs folded 2-D", as_2d, got_self.pos_conv(values)))
    return stages


def unet_stages(source: Path, checkpoint: Path) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    """The denoiser, whole, against upstream carrying the same 1203 tensors.

    One comparison, deliberately. The staged per-block comparisons that found this family's two
    real bugs live in the phase-0 harness, not here: once the whole forward is exact, a per-block
    breakdown of an exact result is 1203 tensors of zero. What this pins is that it *stays* exact.

    The bug worth remembering: this loaded 1203/1203 keys strict -- zero missing, zero unexpected,
    zero shape mismatches -- and was 4.46 wrong, because ``MixFFN.inverted_conv`` was missing a
    SiLU. A parameterless operation leaves no tensor for a strict load to miss.
    """
    import yaml

    sys.path.insert(0, str(source))
    from model_lib import (  # noqa: E402
        UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8 as RefUNet)

    from mozo.vendors.moebius_deploy.config import get_spec  # noqa: E402
    from mozo.vendors.moebius_deploy.network import UNet  # noqa: E402

    raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
    unwrapped = {(k[len("diff_model."):] if k.startswith("diff_model.") else k): v
                 for k, v in raw.items()}

    config = yaml.safe_load((source / "config" / "model_cfg" / "moebius.yaml").read_text())
    spec = get_spec("general")
    kwargs = config["model"]
    kwargs.pop("model_type")
    kwargs["sample_size"] = spec.latent
    kwargs["num_embeddings"] = spec.num_embeddings

    reference = RefUNet(**kwargs).eval()
    reference.load_state_dict({k: v for k, v in unwrapped.items()
                               if not k.startswith("embedding_layer")}, strict=True)
    for parameter in reference.parameters():
        parameter.data = parameter.data.clone()
    table = torch.nn.Embedding(spec.num_embeddings, spec.encoder_hid_dim)
    table.weight.data = raw["embedding_layer.weight"].clone()

    mine = UNet(spec).eval()
    mine.load_state_dict(unwrapped, strict=True)

    torch.manual_seed(SEED)
    sample = torch.randn(2, spec.in_channels, spec.latent, spec.latent)
    timestep = torch.tensor([500])
    uncond, cond = spec.conditioning_ids
    ids = torch.tensor([list(uncond), list(cond)], dtype=torch.long)

    with torch.no_grad():
        want = reference(sample, timestep=timestep, encoder_hidden_states=table(ids)).sample
        got = mine(sample, timestep, mine.conditioning(1))
    return [("UNet forward", got, want)]


def _trunk(encoder, image: torch.Tensor) -> torch.Tensor:
    """Everything in the encoder before the mid block."""
    hidden = encoder.conv_in(image)
    for block in encoder.down_blocks:
        hidden = block(hidden)
    return hidden


def report(stages) -> tuple[int, list[str]]:
    """Print one line per stage; return ``(compared, failures)``."""
    failures = []
    for name, got, want in stages:
        exact = torch.equal(got, want)
        delta = (got - want).abs().max().item()
        print(f"  {'PASS' if exact else 'FAIL'}  {name:<22} max|delta| = {delta:.3e}"
              f"   {tuple(got.shape)}")
        if not exact:
            failures.append(name)
    return len(stages), failures


def _gelu_resnet(self, x):
    """:class:`~.vae.Resnet.forward` with the activation swapped. A perturbation, not a variant."""
    import torch.nn.functional as F

    hidden = self.conv1(F.gelu(self.norm1(x)))
    hidden = self.conv2(F.gelu(self.norm2(hidden)))
    if self.conv_shortcut is not None:
        x = self.conv_shortcut(x)
    return x + hidden


def _symmetric_downsample(self, x):
    """The autoencoder's downsample with ordinary padding instead of ``(0, 1, 0, 1)``."""
    import torch.nn.functional as F

    return F.conv2d(x, self.conv.weight, self.conv.bias, stride=2, padding=1)


def _bilinear_upsample(self, x):
    """The autoencoder's upsample with a smooth filter instead of nearest-neighbour."""
    import torch.nn.functional as F

    return self.conv(F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False))


#: Each perturbation, as ``(what to break, which attribute, what to put there, the first stage it
#: must reach)``. One table rather than a branch per perturbation and a second dict of expected
#: stages: those were two structures keyed by the same strings, and two structures keyed by the
#: same strings drift. A fifth perturbation is now a row.
#:
#: The expected stage matters as much as the failure. A perturbation that breaks everything proves
#: less than one that breaks exactly where the constant it touched is used -- ``decoder conv_in``
#: still passing under an encoder-only perturbation is what says the stages are isolated.
PERTURBATIONS = {
    "groupnorm eps 1e-6 -> 1e-5": (vae_module, "NORM_EPS", 1e-5, "encoder trunk"),
    "downsample pad (0,1,0,1) -> symmetric":
        (vae_module.Downsample, "forward", _symmetric_downsample, "encoder trunk"),
    "upsample nearest -> bilinear":
        (vae_module.Upsample, "forward", _bilinear_upsample, "decode"),
    "silu -> gelu in resnets": (vae_module.Resnet, "forward", _gelu_resnet, "encoder trunk"),
}


@contextlib.contextmanager
def perturbed(what: str):
    """Break one thing, exactly one thing, and put it back afterwards."""
    target, attribute, replacement, _ = PERTURBATIONS[what]
    original = getattr(target, attribute)
    setattr(target, attribute, replacement)
    try:
        yield
    finally:
        setattr(target, attribute, original)


def falsify(weights: Path) -> int:
    """Confirm the gate fails when it should, and at the right stage."""
    print("\nFalsifying -- each perturbation must fail, at the stage it reaches:\n")
    wrong = 0
    for what, (_, _, _, expected) in PERTURBATIONS.items():
        with perturbed(what):
            _, failures = report(vae_stages(weights))
        if not failures:
            print(f"  !! {what}: gate did not notice\n")
            wrong += 1
        elif failures[0] != expected:
            print(f"  !! {what}: first failure {failures[0]!r}, expected {expected!r}\n")
            wrong += 1
        else:
            print(f"  ok  {what}: caught at {failures[0]!r}\n")
    return wrong


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True,
                        help="directory holding the autoencoder's .bin and config.json")
    parser.add_argument("--source", type=Path,
                        help="a checkout of hustvl/Moebius, to gate the λ layers against")
    parser.add_argument("--falsify", action="store_true",
                        help="also perturb one constant at a time and confirm the gate notices")
    args = parser.parse_args()

    print(f"versions: {versions()}   device: cpu   seed: {SEED}\n")
    print("Autoencoder, against diffusers:\n")
    compared, failures = report(vae_stages(args.weights))

    if args.source:
        print("\nλ layers, against hustvl/Moebius at random weights:\n")
        stages = lambda_stages(args.source)
        # The last row records the export rewrite's cost; it is a measurement, not a pass/fail.
        measured = stages.pop()
        extra, broken = report(stages)
        compared, failures = compared + extra, failures + broken
        name, got, want = measured
        print(f"  ----  {name:<22} max|delta| = {(got - want).abs().max().item():.3e}"
              "   (export rewrite, expected non-zero)")

    print(f"\n{compared - len(failures)}/{compared} exact")

    wrong = falsify(args.weights) if args.falsify else 0
    if failures:
        print(f"\nFAILED: {', '.join(failures)}")
    if wrong:
        print(f"\nFAILED: {wrong} perturbation(s) not caught where expected")
    return 1 if (failures or wrong) else 0


if __name__ == "__main__":
    raise SystemExit(main())
