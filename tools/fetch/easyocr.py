#!/usr/bin/env python3
"""Fetch EasyOCR's published weights and fuse them into mozo's checkpoints.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/easyocr.py                     # everything
    python tools/fetch/easyocr.py english japanese

**Two graphs in, one file out.** Upstream ships CRAFT once and a recogniser per script, and
selecting a language means downloading both. mozo publishes a variant, and a variant is one
download: each ``torch-fp32.pth`` holds ``{"detector": ..., "recogniser": ...}``. CRAFT's 83 MB
is therefore repeated across all five, which is the price of a variant that is self-contained
the way every other one in mozo is.

**MD5, not SHA-256.** JaidedAI publishes an md5 per file in ``easyocr/config.py`` and nothing
else, so that is what there is to check against. It is checked against the *unzipped* ``.pth``,
because that is the file upstream hashed -- the release asset is a zip around it.

**Five variants of seventeen.** Upstream publishes eleven first-generation recognisers and eight
second. All five here are second generation, and together they are 88% of upstream's own
download counts. The first-generation network is a different feature extractor and is not
vendored; adding one would be a research question, not a line in this table.

Licensing is uniform: Apache-2.0 on the code, and the weights are release assets of that same
Apache-2.0 repository. The detector's weights originate from CLOVA's CRAFT, whose code is MIT and
whose checkpoint was published without terms of its own -- see the vendor's PROVENANCE.md, which
says so rather than claiming a cleaner chain than exists.

The output is ``weights/easyocr/<variant>/<revision>/torch-fp32.pth`` and a NOTICE naming the
authors and sources. The LICENSE beside them is not written by this script: a licence is part of
what is published, so it lives in the weights tree like any other artifact.
"""

from __future__ import annotations

import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import digest, require_licence, variant_parser  # noqa: E402

from mozo.vendors.easyocr_deploy.checkpoint import DETECTOR, RECOGNISER  # noqa: E402
from mozo.vendors.easyocr_deploy.config import SPECS, VARIANTS  # noqa: E402

_RELEASES = "https://github.com/JaidedAI/EasyOCR/releases/download"

#: The revision this family was published under.
REVISION = "2026-08-21"

#: Where Apache-2.0's canonical text lives. JaidedAI ships a LICENSE in the repository but not
#: beside the release assets, so the text is placed once per revision.
LICENCE_SOURCE = "https://www.apache.org/licenses/LICENSE-2.0.txt"

@dataclass(frozen=True)
class Asset:
    """One published checkpoint: which release it is in, and what it should hash to."""

    tag: str
    name: str
    md5: str

    @property
    def url(self) -> str:
        return f"{_RELEASES}/{self.tag}/{self.name}.zip"


#: The detector, shared by every variant. Trained by CLOVA on MLT and republished by JaidedAI.
#: Named for the network rather than its role, because ``DETECTOR`` is already the checkpoint
#: key imported above and shadowing it writes an ``Asset`` where a string belongs.
CRAFT = Asset("pre-v1.1.6", "craft_mlt_25k", "2f8227d2def4037cdb3b34389dcf9ec1")

#: One recogniser per variant. All second generation; see the module docstring.
RECOGNISERS = {
    "english": Asset("v1.3", "english_g2", "5864788e1821be9e454ec108d61b887d"),
    "latin": Asset("v1.3", "latin_g2", "469869130aad1a34e8f9086f4262bc59"),
    "chinese-simplified": Asset("v1.3", "zh_sim_g2", "b601ce7143293387d3ec4f41a66edc07"),
    "japanese": Asset("v1.3", "japanese_g2", "bad5146990ccb1272cb0908440fbe15e"),
    "korean": Asset("v1.3", "korean_g2", "befecf7b1ca2fffb5af814a51443682d"),
}

NOTICE = """\
EasyOCR
=======

Text detection and recognition, published by Jaided AI.

Code:    https://github.com/JaidedAI/EasyOCR  (Apache License 2.0)
Weights: release assets of that repository, under the same licence.

This artifact fuses two of upstream's published checkpoints into one file.

  detector    craft_mlt_25k.pth   {detector_url}
  recogniser  {recogniser_name}   {recogniser_url}

The detector is CRAFT -- "Character Region Awareness for Text Detection", Baek et al., 2019 --
originally from CLOVA AI Research (https://github.com/clovaai/CRAFT-pytorch, MIT licence). CLOVA
published the trained weights without stating terms for them separately from that code; the file
redistributed here is the copy Jaided AI publishes as a release asset of its Apache-2.0
repository, and that is the chain relied on.

The recogniser is a CRNN in the shape of clovaai/deep-text-recognition-benchmark (Apache License
2.0), retrained by Jaided AI on the {variant} script.

Neither network is modified. The tensors are upstream's, byte for byte; only the two state
dictionaries' arrangement into a single file is mozo's.
"""


def _unzip(archive: Path, member: str, target: Path) -> Path:
    """Extract one member, returning where it landed. Upstream zips a single ``.pth``."""
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if member not in names:
            raise SystemExit(f"{archive} holds {names}, expected {member}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with bundle.open(member) as source, target.open("wb") as out:
            while chunk := source.read(1 << 20):
                out.write(chunk)
    return target


def _download_pth(asset: Asset, staging: Path) -> Path:
    """Fetch one release zip and unpack the checkpoint inside it, md5-checked.

    ``common.download_verified`` cannot be used directly here: JaidedAI publishes an md5 for the
    ``.pth`` *inside* the zip and none for the zip, so the bytes that arrive are not the bytes
    with a digest. The check is therefore on the extracted file, which is what the published md5
    describes. A zip that arrived corrupt fails at the unzip or at the md5, either way before
    anything is published.
    """
    archive = staging / f"{asset.name}.zip"
    checkpoint = staging / f"{asset.name}.pth"
    if not (checkpoint.is_file() and digest(checkpoint, "md5") == asset.md5):
        if not archive.is_file():
            _fetch(asset.url, archive, asset.name)
        _unzip(archive, f"{asset.name}.pth", checkpoint)
        actual = digest(checkpoint, "md5")
        if actual != asset.md5:
            raise SystemExit(
                f"{asset.name}: md5 mismatch on {asset.name}.pth. Expected {asset.md5}, got "
                f"{actual}. Upstream may have re-released this file; do not publish it until "
                "the change is understood."
            )
    print(f"  {asset.name:<20} ok, {checkpoint.stat().st_size / 1e6:.1f} MB", flush=True)
    return checkpoint


def _fetch(url: str, target: Path, label: str) -> None:
    """Plain download. The digest is checked after unzipping, not here."""
    import urllib.request

    target.parent.mkdir(parents=True, exist_ok=True)
    staged = target.with_name(target.name + ".part")
    print(f"  {label:<20} downloading {url}", flush=True)
    try:
        with urllib.request.urlopen(url, timeout=600) as response, staged.open("wb") as out:
            while chunk := response.read(1 << 20):
                out.write(chunk)
        staged.replace(target)
    finally:
        staged.unlink(missing_ok=True)


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", revision=REVISION)
    args = parser.parse_args()
    wanted = args.variants or VARIANTS
    unknown = [v for v in wanted if v not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variant(s) {unknown}. Known: {VARIANTS}")

    # Outside the weights tree, not inside it. Everything under ``weights/`` is published --
    # that is what the manifest generator scans and what gets synced to the CDN -- and a
    # half-gigabyte of upstream archives is not an artifact. This family needs scratch because
    # it fuses two of them into one file; every other fetch tool downloads straight to its
    # target and has none.
    staging = ROOT / ".fetch-cache" / "easyocr"
    print("detector (shared by every variant)")
    detector_state = torch.load(_download_pth(CRAFT, staging), map_location="cpu",
                                weights_only=True)

    for variant in (v for v in VARIANTS if v in wanted):
        recogniser = RECOGNISERS[variant]
        print(f"\n{variant}")
        state = torch.load(_download_pth(recogniser, staging), map_location="cpu",
                           weights_only=True)

        revision_dir = args.weights_dir / "easyocr" / variant / args.revision
        revision_dir.mkdir(parents=True, exist_ok=True)
        torch.save({DETECTOR: detector_state, RECOGNISER: state},
                   revision_dir / "torch-fp32.pth")
        (revision_dir / "NOTICE").write_text(NOTICE.format(
            detector_url=CRAFT.url,
            recogniser_name=f"{recogniser.name}.pth",
            recogniser_url=recogniser.url,
            variant=variant,
        ))
        size = (revision_dir / "torch-fp32.pth").stat().st_size / 1e6
        print(f"  {'fused':<20} {revision_dir / 'torch-fp32.pth'}  {size:.1f} MB")
        require_licence(revision_dir, "Apache-2.0", LICENCE_SOURCE)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
