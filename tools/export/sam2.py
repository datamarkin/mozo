#!/usr/bin/env python3
"""Export SAM 2 to the graph artifacts mozo publishes.

    python tools/export/sam2.py tiny base_plus
    python tools/export/sam2.py large --revision 2026-08-20

This runs once, on a machine you control, and never ships.

SAM 2 exports as **two** graphs rather than one, which is the same split the vendor draws between
``Sam2.encode`` and ``Sam2.decode``. The encoder is fixed-shape and expensive; the decoder takes a
variable number of prompt points and is cheap. Fusing them would force a re-encode on every click,
which is the one thing the whole design exists to avoid.

**The decoder graph returns all four mask tokens.** Upstream slices to one or three inside
``MaskDecoder.forward`` depending on ``multimask_output``, which is a Python bool and would bake
into the graph. Exporting ``predict_masks`` instead -- the unsliced internal -- lets one artifact
serve both settings, with the slice done in Python where the bool lives.

**There is no CoreML here on purpose.** Apple publishes converted SAM 2.1 packages for tiny, small
and large under Apache-2.0, and mozo redistributes those rather than producing its own. They are
fp16 and split three ways rather than two -- image encoder, prompt encoder, mask decoder -- so
they are labelled ``coreml-fp16-*`` and driven through their own interface. ``base_plus`` has no
CoreML because Apple did not publish one. Converting SAM 2 ourselves is possible (Hiera's bicubic
position-embedding interpolation is the one op coremltools lacks, and at a fixed 1024 input it
folds to a constant), but writing an exporter to duplicate artifacts that already exist is work
without a result.

Every artifact is checked against the torch model it came from before it is written, on masks
rather than on raw tensors -- what matters is that the two artifacts segment the same pixels.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from common import fixtures, variant_parser  # noqa: E402
from mozo.image import load_image  # noqa: E402
from mozo.runtimes import OnnxRunner  # noqa: E402
from mozo.vendors.sam2_deploy import Segmenter  # noqa: E402
from mozo.vendors.sam2_deploy.config import SPECS  # noqa: E402
from mozo.vendors.sam2_deploy.image import (  # noqa: E402
    MASK_THRESHOLD, preprocess, to_model_coords, to_original)

#: The revision these weights were published under.
REVISION = "2026-08-20"

#: Opset the graphs are written at. A constant rather than a flag, and shared by both halves:
#: which opset mozo publishes is a property of the artifact, and the encoder and decoder of one
#: variant drifting apart on it is not a thing anyone should be able to do in one line.
OPSET = 17

#: Agreement required of an exported graph, in mask pixels that may differ from torch. Not zero:
#: a graph runtime's convolutions are its own, and a mask boundary lands on whichever side of the
#: threshold that arithmetic puts it. Four times the worst ever measured here (0-4 px of 2.5M),
#: which leaves room for another photograph without leaving room for a broken graph.
MASK_TOLERANCE_PX = 16


class _Encoder(torch.nn.Module):
    """The expensive half as a graph: an image in, the three cached feature maps out."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, image):
        features = self.network.encode(image)
        return features["image_embed"], *features["high_res_feats"]


class _Decoder(torch.nn.Module):
    """The cheap half as a graph: cached features and a prompt in, all four mask tokens out."""

    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, image_embed, high_res_0, high_res_1, point_coords, point_labels):
        prompt = self.network.sam_prompt_encoder
        sparse, dense = prompt(points=(point_coords, point_labels), boxes=None, masks=None)
        masks, iou, _, _ = self.network.sam_mask_decoder.predict_masks(
            image_embeddings=image_embed,
            image_pe=prompt.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            repeat_image=False,
            high_res_features=[high_res_0, high_res_1],
        )
        return masks, iou


def _prompts(image: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Prompt cases an export is checked on: a click, a box, and a mixed sign set.

    A box is given as the two corner points it is spelled as, rather than as four numbers, so the
    verification loop below can feed every case through one path instead of recovering which kind
    it is from the label values.
    """
    height, width = image.shape[:2]
    return [
        ("one positive point", np.array([[width // 2, height // 2]], float), np.array([1])),
        ("box", np.array([[width * 0.2, height * 0.2], [width * 0.8, height * 0.8]], float),
         np.array([2, 3])),
        ("positive and negative", np.array([[width // 2, height // 2],
                                            [width // 4, height // 4]], float), np.array([1, 0])),
    ]


def _masks_from(low_res: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Take a graph's raw mask tokens to booleans in the source image, as the vendor would."""
    return (to_original(torch.from_numpy(low_res[:, 1:]), shape) > MASK_THRESHOLD).numpy()


def export_variant(checkpoint: Path, destination: Path) -> None:
    """Export one checkpoint's two graphs, refusing to publish either if it disagrees with torch.

    Both graphs are written to a scratch directory and moved into place only once every prompt
    case has agreed. A rejected artifact left in the revision directory would be hashed and
    published by ``tools/generate_manifest.py`` on the next run, which reads the directory as the
    source of truth and has no way to know this gate turned it down.
    """
    segmenter = Segmenter(checkpoint)
    size = segmenter.image_size
    graphs = {}

    with tempfile.TemporaryDirectory() as scratch:
        staged = Path(scratch)
        for image in fixtures():
            image_pixels = load_image(image)
            shape = image_pixels.shape[:2]
            # Through the cache, so the reference predictions below reuse this encode rather than
            # running the encoder a second time on the identical batch.
            features = segmenter.encode(image_pixels)
            batch = preprocess(image_pixels, size)
            embed = features["image_embed"]
            high = list(features["high_res_feats"])

            if not graphs:
                torch.onnx.export(
                    _Encoder(segmenter.network), (batch,), str(staged / "onnx-fp32-encoder.onnx"),
                    input_names=["image"],
                    output_names=["image_embed", "high_res_0", "high_res_1"],
                    opset_version=OPSET, dynamo=False,
                )
                sample = to_model_coords(np.array([[0.0, 0.0]]), shape, size)[None]
                torch.onnx.export(
                    _Decoder(segmenter.network),
                    (embed, high[0], high[1], sample, torch.ones(1, 1, dtype=torch.int32)),
                    str(staged / "onnx-fp32-decoder.onnx"),
                    input_names=["image_embed", "high_res_0", "high_res_1",
                                 "point_coords", "point_labels"],
                    output_names=["low_res_masks", "iou_predictions"],
                    dynamic_axes={"point_coords": {1: "points"}, "point_labels": {1: "points"}},
                    opset_version=OPSET, dynamo=False,
                )
                graphs = {name: OnnxRunner(staged / f"onnx-fp32-{name}.onnx")
                          for name in ("encoder", "decoder")}

            got = graphs["encoder"](batch.numpy())
            for name, points, labels in _prompts(image_pixels):
                coords = to_model_coords(points, shape, size).numpy()[None].astype(np.float32)
                low, _ = graphs["decoder"](
                    got[0], high_res_0=got[1], high_res_1=got[2],
                    point_coords=coords, point_labels=labels[None].astype(np.int32))
                want = segmenter.predict(image_pixels, points=points, labels=labels)
                differing = int((_masks_from(low, shape)[0] != want.masks[0]).sum())
                print(f"    onnx  {image.name} {name:24} {differing:>6} mask px differ")
                if differing > MASK_TOLERANCE_PX:
                    raise SystemExit(
                        f"onnx export disagrees with torch by {differing} px on {image.name} "
                        f"({name}); not published")

        # Only now, with every case agreed, does anything land where the manifest can see it.
        destination.mkdir(parents=True, exist_ok=True)
        for name in ("encoder", "decoder"):
            shutil.move(str(staged / f"onnx-fp32-{name}.onnx"),
                        str(destination / f"onnx-fp32-{name}.onnx"))


def main() -> int:
    """Export the variants named on the command line."""
    parser = variant_parser(__doc__, ROOT / "weights", required=True, revision=REVISION)
    args = parser.parse_args()
    unknown = [name for name in args.variants if name not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}; choose from {list(SPECS)}")
    for name in args.variants:
        revision = args.weights_dir / "sam2" / name / args.revision
        checkpoint = revision / "torch-fp32.pth"
        if not checkpoint.is_file():
            raise SystemExit(f"{checkpoint} is missing; place the checkpoint there first")
        print(f"  {name} ({args.revision})")
        export_variant(checkpoint, revision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
