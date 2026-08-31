#!/usr/bin/env python3
"""Export SAM 3's vision encoder to the CoreML package mozo publishes.

    python tools/export/sam3.py sam3
    python tools/export/sam3.py sam3 --revision 2026-08-20

This runs once, on a machine you control, and never ships. It needs macOS: the gate below runs
the package it just built, and nothing else can.

**One graph, and it is the image half.** SAM 3's cost is lopsided -- 4.8 s of vision encode
against 88 ms of text and 670 ms of everything that joins them -- and the vision half is also the
half that depends on nothing but the image, which is why ``Segmenter`` caches it. So it is the
half worth exporting and the only one exported here. The text tower and the concept head stay in
torch, and the checkpoint is loaded either way; what the package changes is that the trunk's
1.85 GB is then never loaded beside it.

The graph carries the trunk, the *concept* half of the dual FPN, and the coarsest level's
position encoding -- exactly what ``VisionEncoder.forward(batch, stacks=("concept",))`` returns,
so :class:`~mozo.adapters.sam3.GraphVision` can stand where the torch encoder stood without
assembling anything. The click stack is not in it: the click path reads a differently
preprocessed image and needs its own trunk pass, so a segmenter given this graph refuses clicks
rather than answering them from the wrong pixels.

**The rotation is rewritten in real arithmetic, and that is the only change to the model.**
``vision/vit.py`` keeps SAM 3's rotary table complex, as the checkpoint ships it and as upstream
multiplies it. coremltools cannot carry that: its torch frontend gives ``add`` a complex branch
and ``mul`` none, and ``view_as_complex`` is not registered at all -- so the multiply that *is*
the rotation has no path through the converter. Teaching it one would mean replacing the
converter's own ``mul`` handler, which every other multiply in the graph also goes through.

:class:`RealRotary` instead spells the same product out: ``(a + bi)(c + di) = (ac - bd) +
(ad + bc)i``, over the two halves of the shipped table rather than a rebuilt cosine. It
overrides ``RoPEAttention.rotate`` and inherits the rest of the block, so this is one step of
that attention written differently and not a second copy of it.

That rewrite is not an approximation, and it is worth being exact about which claim is being
made. The two forms disagree on about a fifth of the elements they touch, by one float32 ulp --
4.77e-07. But measured against a float64 evaluation of the same product, **both land 6.38e-07
away, and neither is nearer**: they are two valid roundings of one expression, not a good one
and a degraded one. What the trunk then does with a one-ulp difference is the real story --
32 blocks amplify it to about 1.35e-02, which is why the gate below reads masks and not tensors.

**fp32, not fp16, and not int8 -- every alternative was built and measured.** On an M1 Max,
encoding the fixture, against the torch encoder it replaces:

    variant                          size    encode   worst score d   mask px differ
    torch cpu                           -   4488 ms               -                -
    torch mps                           -   1098 ms               -                -
    coreml fp32                   1.70 GB    840 ms        1.4e-05                1
    coreml fp16, linear ops only  0.94 GB    822 ms        5.6e-04               16
    coreml fp16 weights, fp32 compute
                                  0.93 GB   7000 ms        5.5e-04               11
    coreml fp16, everything       0.88 GB    750 ms        5.7e-03              273
                                           (1826 ms if the ANE is offered)
    coreml int8, per channel      0.46 GB    869 ms        5.9e-03              490
    coreml int8, per block of 32  0.52 GB    876 ms        2.4e-02              244

Read the encode column first: **half precision buys no speed here.** 840 ms against 822 ms is
noise. The one unit where fp16 would be transformative is the ANE, and the ANE cannot run this
graph -- offered it, blanket fp16 gets *slower* (1826 ms), ANE-only did not finish a single pass
in two minutes, and the compiler logs ``ANECCompile() FAILED`` on the mixed model. A ViT-L at
1008 pixels, four of whose blocks attend over 5184 tokens at once, is not an ANE shape. What is
left for fp16 to buy is 45% of the download, and it costs an order of magnitude of fidelity.

So fp32 is both the fastest artifact and the most accurate one, which is the whole argument. It
also needs no compute-unit setting: the ANE cannot execute fp32 at all, so it runs identically
however CoreML is asked to schedule it, and :class:`~mozo.runtimes.CoreMLRunner` needs no knob
it does not have.

Two of the rejected rows are worth keeping rather than just discarding:

- **int8 is disqualified, not merely worse.** Per-block int8 moved one instance's score by
  2.4e-02 where the closest instance on this fixture clears the 0.5 cut by 0.045. That is
  within a factor of two of adding or dropping a detection outright, on the one image this has
  been measured on. A third of the size is not worth a coin toss at the threshold.
- **``constexpr_cast`` does cleanly separate fp16 storage from fp32 arithmetic**, and it is the
  best fidelity-per-byte of everything here -- 0.93 GB at 11 differing pixels. CoreML does not
  fuse the decompression, so it runs at 7 s, eight times slower than doing no compression at
  all. It is the right op and the wrong trade, and it is written down so nobody rediscovers it.

**The gate checks two different things, because one of them is not enough.**

What must not change is what the model *finds*, so every prompt is checked on its instance count,
then its scores, then its mask pixels. That is the claim a user cares about, and the way a vision
encoder could quietly break it is the threshold: ``instances`` cuts at 0.5, so a query at 0.4999
appearing or vanishing is a regression no mask comparison would see.

But that check is not *sensitive*. Setting every trunk ``LayerNorm``'s epsilon to 1e-6 -- the
divergence ``config.py`` warns about, since ``transformers`` defaults to it and Meta pins 1e-5 --
changes nothing it can see: the same six people, scores moved 4.3e-05 where an honest conversion
moves 1.4e-05, not one mask pixel over the allowance. A gate that publishes that graph is not a
gate. So the encoder's own output is checked too, on relative L2, where the same perturbation
reads 7.2e-05 against an honest 2.8e-05.

Tensor equality is not on the table and is not the goal: the trunk amplifies one ulp into
1.35e-02, so a graph behaving perfectly still disagrees in the fourth decimal. That amplification
is also why the margin here is 2.6x rather than orders of magnitude -- a real mistake and honest
rounding come out the same size once 32 blocks have multiplied both. This gate is not asked to
prove SAM 3 correct; ``tools/verify/sam3.py`` does that exactly, on 73 stages, against the
reference. It is asked to prove the conversion did not change it, and a mistake subtler than an
epsilon may pass. That is the limit, stated rather than papered over.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, variant_parser  # noqa: E402
from mozo.adapters.sam3 import LEVELS, GraphVision  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import CoreMLRunner  # noqa: E402
from mozo.vendors.sam3_deploy import Segmenter  # noqa: E402
from mozo.vendors.sam3_deploy.checkpoint import load_state_dict, vision_state_dict  # noqa: E402
from mozo.vendors.sam3_deploy.config import SPEC  # noqa: E402
from mozo.vendors.sam3_deploy.vision.encoder import VisionEncoder  # noqa: E402
from mozo.vendors.sam3_deploy.vision.vit import RoPEAttention  # noqa: E402

#: The revision these weights were published under.
REVISION = "2026-08-20"

#: The artifact this writes. No ``-vision`` suffix, though that is what it holds: a suffix in an
#: artifact key means "one part of a runtime split across files", and SAM 3's CoreML runtime is
#: one file. ``tools/generate_manifest.py`` refuses a lone suffixed part for exactly that reason
#: -- it cannot be told apart from an unsplit runtime. What the key does *not* say, and what the
#: module docstring above and :class:`~mozo.adapters.sam3.GraphVision` do, is that this runtime
#: covers the image half and the checkpoint is still needed for the rest.
ARTIFACT = "coreml-fp32"

#: Minimum deployment target. Not a default worth taking: below iOS 18 / macOS 15 coremltools has
#: no fused attention op and decomposes ``scaled_dot_product_attention`` into matmul and softmax.
#: Four of the trunk's 32 blocks attend over the whole 72x72 grid, so that decomposition would
#: materialise a 16 x 5184 x 5184 score matrix -- 1.7 GB, per block, that torch's own kernel
#: never forms.
TARGET = "iOS18"

#: Prompts the export is gated on. Chosen for what they do to the *threshold*, which is what a
#: vision encoder can quietly move: a common concept, a multi-word phrase, and one that is absent
#: -- the last being the only case where the right answer is that nothing survives the cut.
PROMPTS = ["person", "coffee mug", "a person holding a coffee mug", "cow"]

#: Confidence a published graph may move an instance's score by. Deliberately looser than
#: :data:`FEATURE_TOLERANCE`, because it is guarding something else: not drift, but a detection
#: crossing the 0.5 cut. Seven times the worst measured (1.4e-05), and still three orders below
#: the 0.045 margin the closest instance on the fixture has. Drift is the feature check's job,
#: and scores are a handful of numbers where that is millions.
SCORE_TOLERANCE = 1e-4

#: Relative L2 -- ``||want - got|| / ||want||`` -- a published graph may move a pyramid level by.
#: This is the sensitive half of the gate, and the number is set between two measurements rather
#: than chosen: an honest conversion reads at most 2.8e-05 across the three levels, and a trunk
#: whose ``LayerNorm`` epsilon has been moved to 1e-6 reads at least 7.2e-05. Relative L2 rather
#: than a maximum, which is one element's bad luck: over 27 M of them the norm is what holds
#: still between images.
FEATURE_TOLERANCE = 4e-5

#: Mask pixels that may differ, of the 2.46 M in this fixture. The allowance SAM 2's export
#: takes, and sixteen times the worst measured here -- a mask boundary lands on whichever side
#: of the threshold a runtime's own convolutions put it, and one pixel of edge is that.
MASK_TOLERANCE_PX = 16


class RealRotary(RoPEAttention):
    """One block's attention with the rotary step in real arithmetic. See the module docstring.

    Adopts the block's own projections rather than building and reloading its own -- there is
    nothing here to construct, which is why this calls ``nn.Module.__init__`` rather than
    ``super().__init__``. Everything but :meth:`rotate` is inherited, so the graph is built from
    the same attention the torch path runs.
    """

    def __init__(self, source: RoPEAttention) -> None:
        nn.Module.__init__(self)
        self.heads, self.head_dim = source.heads, source.head_dim
        self.qkv, self.o_proj = source.qkv, source.o_proj
        # The shipped table's two halves, not a rebuilt cosine: rebuilding lands one ulp from
        # the values the weights were trained under, and this rewrite already spends the only
        # ulp it can afford to.
        self.register_buffer("cos", source.freqs_cis.real.contiguous())
        self.register_buffer("sin", source.freqs_cis.imag.contiguous())

    def rotate(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``(a + bi)(c + di) = (ac - bd) + (ad + bc)i``, over adjacent channel pairs."""
        def turn(x: torch.Tensor) -> torch.Tensor:
            pairs = x.float().reshape(*x.shape[:-1], -1, 2)
            a, b = pairs[..., 0], pairs[..., 1]
            rotated = (a * self.cos - b * self.sin, a * self.sin + b * self.cos)
            return torch.stack(rotated, dim=-1).flatten(3).type_as(x)

        return turn(query), turn(key)


class Graph(nn.Module):
    """The vision encoder as one traceable function: a batch in, four tensors out.

    ``VisionEncoder`` answers with a dict holding a list, which a graph cannot return, and takes
    ``stacks`` to choose a pyramid, which a graph cannot branch on. Both are settled here rather
    than in the vendor, where they are the right shape for a caller that has a choice to make.
    """

    def __init__(self, encoder: VisionEncoder) -> None:
        super().__init__()
        for layer in encoder.trunk.layers:
            layer.attention = RealRotary(layer.attention)
        self.encoder = encoder

    def forward(self, batch: torch.Tensor) -> tuple[torch.Tensor, ...]:
        got = self.encoder(batch, stacks=("concept",))
        return (*got["concept"], got["positions"])


def _instances(segmenter: Segmenter, pixels: np.ndarray) -> dict[str, dict]:
    """What this segmenter finds for every gated prompt, keyed by prompt."""
    return {prompt: segmenter.predict(pixels, prompt) for prompt in PROMPTS}


def _reject(reason: str) -> None:
    """Refuse to publish, saying which artifact and why. Every gate below ends here."""
    raise SystemExit(f"{ARTIFACT} {reason}; not published")


def _features(want: list, got: list, image: str) -> None:
    """Raise unless every pyramid level the graph produced matches the trunk's closely enough."""
    for level, (a, b) in enumerate(zip(want, got)):
        drift = (torch.linalg.vector_norm(a - b) / torch.linalg.vector_norm(a)).item()
        print(f"    {image} level{level:<29} relative L2 {drift:.2e}")
        if drift > FEATURE_TOLERANCE:
            _reject(f"moves level{level} on {image} by {drift:.2e}, "
                    f"over {FEATURE_TOLERANCE:.0e}")


def _compare(want: dict, got: dict, prompt: str, image: str) -> None:
    """Raise unless the two agree on what is there, how sure they are, and which pixels."""
    if len(want["scores"]) != len(got["scores"]):
        _reject(f"finds {len(got['scores'])} instances of {prompt!r} on {image} "
                f"where torch finds {len(want['scores'])}")
    if not len(want["scores"]):
        print(f"    {image} {prompt!r:34} nothing found, as in torch")
        return

    score = (want["scores"] - got["scores"]).abs().max().item()
    differing = int((want["masks"] ^ got["masks"]).flatten(1).sum(1).max())
    closest = (want["scores"] - 0.5).abs().min().item()
    print(f"    {image} {prompt!r:34} {len(want['scores'])} found, "
          f"dscore {score:.1e}, {differing:>3} mask px differ, "
          f"nearest the cut {closest:+.3f}")
    if score > SCORE_TOLERANCE:
        _reject(f"moves a score for {prompt!r} on {image} by {score:.2e}, "
                f"over {SCORE_TOLERANCE:.0e}")
    if differing > MASK_TOLERANCE_PX:
        _reject(f"disagrees with torch by {differing} mask px for {prompt!r} "
                f"on {image}, over {MASK_TOLERANCE_PX}")


def export_variant(checkpoint: Path, destination: Path) -> None:
    """Convert the vision encoder, and publish it only if it finds what torch finds.

    Written to a scratch directory and moved into place only once every prompt on every fixture
    has agreed. A rejected package left in the revision directory would be hashed and published
    by ``tools/generate_manifest.py`` on the next run, which reads the directory as the source
    of truth and has no way to know this gate turned it down.
    """
    import coremltools as ct

    size = SPEC.trunk.image_size
    encoder = VisionEncoder()
    encoder.load_state_dict(vision_state_dict(load_state_dict(str(checkpoint))), strict=True)
    graph = Graph(encoder).eval()

    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch) / "package.mlpackage"
        with torch.no_grad():
            traced = torch.jit.trace(graph, torch.zeros(1, 3, size, size))
        # Freed before conversion: coremltools holds the whole program in memory while it
        # builds it, and the eager modules are no longer anybody's reference once traced.
        del graph, encoder

        print(f"    converting at {TARGET}", flush=True)
        ct.convert(
            traced,
            inputs=[ct.TensorType(name="image", shape=(1, 3, size, size), dtype=np.float32)],
            outputs=[ct.TensorType(name=name, dtype=np.float32)
                     for name in (*LEVELS, "positions")],
            compute_precision=ct.precision.FLOAT32,
            minimum_deployment_target=getattr(ct.target, TARGET),
        ).save(str(staged))
        del traced

        torch_segmenter = Segmenter(checkpoint)
        graph_segmenter = Segmenter(
            checkpoint, vision=GraphVision(CoreMLRunner(staged), "cpu"))
        for image in fixtures():
            pixels = load_image(image)
            _features(torch_segmenter.encode_image(pixels)["concept"],
                      graph_segmenter.encode_image(pixels)["concept"], image.name)
            want = _instances(torch_segmenter, pixels)
            got = _instances(graph_segmenter, pixels)
            for prompt in PROMPTS:
                _compare(want[prompt], got[prompt], prompt, image.name)
        del torch_segmenter, graph_segmenter

        # Only now, with every prompt agreed, does anything land where the manifest can see it.
        destination.mkdir(parents=True, exist_ok=True)
        shutil.make_archive(str(destination / ARTIFACT), "zip", root_dir=staged)


def main() -> int:
    """Export the variants named on the command line."""
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=REVISION)
    args = parser.parse_args()
    unknown = [name for name in args.variants if name != "sam3"]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; SAM 3 publishes one, named 'sam3'")
    for name in args.variants:
        revision = args.weights_dir / "sam3" / name / args.revision
        checkpoint = revision / "torch-fp32.pth"
        if not checkpoint.is_file():
            raise SystemExit(f"{checkpoint} is missing. Run tools/fetch/sam3.py first.")
        print(f"  {name} ({args.revision})")
        export_variant(checkpoint, revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
