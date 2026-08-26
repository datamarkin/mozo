#!/usr/bin/env python3
"""Fetch Moebius's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- where a checkpoint came from is a fact about publishing, not about inference.

    python tools/fetch/moebius.py            # every variant
    python tools/fetch/moebius.py general

**Two artifacts per variant, from two repositories.** Moebius is a denoiser and an autoencoder, and
the autoencoder is not its own: upstream's config names ``sdvae_f8d4`` and its README points at
``hustvl/PixelHacker``. mozo publishes the pair as ``torch-fp32-unet`` and ``torch-fp32-vae``
rather than folding one into the other, because they are separate works with separate provenance
and a reader who is handed one file cannot tell that.

The autoencoder is byte-identical across variants and is fetched once per revision.

**The download hash comes from Hugging Face's API, not from a README.** Each blob's sha256 is read
from ``/api/models/<repo>?blobs=true`` at fetch time, so a mismatch means the bytes changed under us
rather than that someone transcribed a digest wrongly.

**The checkpoints are pickles, not safetensors**, so unlike SigLIP 2 and ViTPose there is nothing to
repack -- the ``.bin`` is placed as a ``.pth`` unchanged, as OWLv2 does. They are loaded everywhere
in mozo with ``weights_only=True``.

**Four checkpoints are published; mozo carries two.** ``ft_celebahq`` and ``ft_ffhq`` are
face-specific, identically shaped, and exercise no code path the other two do not -- 1.7 GB for a
task mozo has no other model for. They are named below so that adding one is a decision rather than
a discovery.

Licensing is permissive and the authors state it twice, differently: the GitHub README says
Apache-2.0 on code *and weights* with commercial use explicitly permitted, while the Hugging Face
model card's tag says MIT. Both are recorded in the NOTICE; neither is chosen over the other. The
autoencoder's own card says MIT, and its config is Stable Diffusion XL's VAE config with one value
changed -- ``stabilityai/sdxl-vae`` is itself an MIT repository, so the chain terminates in MIT
whichever link you follow.
"""

from __future__ import annotations

import json
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import download_verified, require_licence, variant_parser  # noqa: E402

_HF = "https://huggingface.co"

#: The revision this family was published under.
REVISION = "2026-08-26"

#: Apache's canonical text. Upstream ships a LICENSE file; this is the fallback used when placing
#: terms beside the autoencoder, whose repository states MIT on the card and ships no file.
LICENCE_SOURCE = "https://www.apache.org/licenses/LICENSE-2.0.txt"

#: mozo's variant name -> the subdirectory upstream publishes it under in ``hustvl/Moebius``.
#: ``general`` is upstream's ``pretrained``; mozo names by what it is for, as in every other family.
VARIANTS: dict[str, str] = {
    "general": "pretrained",
    "places2": "ft_places2",
}

#: Deliberately not carried. Face-specific, identically shaped, 863 MB each.
SKIPPED = ("ft_celebahq", "ft_ffhq")

#: The autoencoder. A different repository, a different paper, a different licence claim.
VAE_REPO = "hustvl/PixelHacker"
VAE_FILE = "vae/diffusion_pytorch_model.bin"


def hub_json(url: str, timeout: int = 60) -> dict:
    """Read one JSON document from the Hub."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def blob_sha256(repo: str, filename: str) -> str:
    """The sha256 Hugging Face records for *filename* in *repo*.

    Only LFS-tracked files carry a content hash; the small ones carry a git blob id, which hashes
    different bytes and is useless for verifying a download. Every checkpoint here is LFS, so a
    file missing from this listing raises rather than being waved through unverified.
    """
    listing = hub_json(f"{_HF}/api/models/{repo}?blobs=true")
    for entry in listing["siblings"]:
        if entry["rfilename"] == filename and entry.get("lfs"):
            return entry["lfs"]["sha256"]
    raise SystemExit(f"{repo} records no content hash for {filename}; refusing to publish it")


def _unet_notice(variant: str, upstream: str, sha256: str) -> str:
    """Attribution that travels with the denoiser."""
    return (
        f"Moebius -- {variant}\n"
        "lightweight image inpainting: an image and a mask in, the masked thing gone\n\n"
        "Copyright 2026 Huazhong University of Science and Technology (HUST Vision Lab)\n"
        "and vivo AI Lab.\n\n"
        f"Source:  {_HF}/hustvl/Moebius  ({upstream}/diffusion_pytorch_model.bin)\n"
        "Project: https://github.com/hustvl/Moebius\n"
        "Paper:   Moebius: 0.2B Lightweight Image Inpainting Framework with 10B-Level\n"
        "         Performance (ECCV 2026), arXiv:2606.19195\n\n"
        "Licence: the authors state terms in two places and the two differ. Both are recorded\n"
        "         here and neither is preferred:\n"
        "           - github.com/hustvl/Moebius README: \"Both the code and the pretrained\n"
        "             model weights of Moebius are released under the Apache License 2.0 ...\n"
        "             Commercial use of the weights and the images produced with them is\n"
        "             permitted.\"\n"
        "           - huggingface.co/hustvl/Moebius model card metadata: license: mit\n"
        "         Both are permissive and both permit commercial use.\n\n"
        "This is upstream's byte stream, verified against the sha256 the Hub records and placed\n"
        "unchanged. No tensor is altered, renamed, cast or dropped.\n"
        f"           {upstream}/diffusion_pytorch_model.bin  sha256 {sha256}\n\n"
        "This checkpoint is the denoiser only. It cannot run without the autoencoder published\n"
        "beside it as torch-fp32-vae, which is a separate work under separate terms -- see the\n"
        "NOTICE that travels with it.\n"
    )


def _vae_notice(sha256: str) -> str:
    """Attribution that travels with the autoencoder, whose chain is the longer of the two."""
    return (
        "PixelHacker VAE -- the autoencoder Moebius denoises inside\n"
        "512x512 pixels <-> a 64x64x4 latent\n\n"
        "Copyright Huazhong University of Science and Technology (HUST Vision Lab).\n\n"
        f"Source:  {_HF}/{VAE_REPO}  ({VAE_FILE})\n"
        "Project: https://github.com/hustvl/PixelHacker\n"
        "Paper:   PixelHacker: Image Inpainting with Structural and Semantic Consistency,\n"
        "         arXiv:2504.20438\n"
        "Licence: MIT, stated on the Hugging Face model card.\n\n"
        "Provenance worth stating plainly: the config this ships with is Stable Diffusion XL's\n"
        "VAE config with sample_size changed from 1024 to 512. Every other value matches,\n"
        "including scaling_factor 0.13025 to five digits, so this is SDXL's autoencoder,\n"
        "fine-tuned or not. That matters only for terms, and it does not change them:\n"
        "stabilityai/sdxl-vae is published as its own MIT repository, separate from SDXL base's\n"
        "OpenRAIL++. The chain terminates in MIT whichever link is followed.\n\n"
        "Upstream stores these tensors in half precision; they are placed here unchanged, and\n"
        "mozo casts to fp32 at load.\n"
        f"           {VAE_FILE}  sha256 {sha256}\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's denoiser, verify it, and place it with its notice."""
    upstream = VARIANTS[variant]
    target = weights_dir / "moebius" / variant / revision / "torch-fp32-unet.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    name = f"{upstream}/diffusion_pytorch_model.bin"
    sha256 = download_verified(f"{_HF}/hustvl/Moebius/resolve/main/{name}", target,
                               blob_sha256("hustvl/Moebius", name),
                               label=variant, width=8, detail="Apache-2.0 / MIT")

    (target.parent / "NOTICE").write_text(_unet_notice(variant, upstream, sha256))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)
    fetch_vae(variant, revision, weights_dir)


def fetch_vae(variant: str, revision: str, weights_dir: Path) -> None:
    """Place the autoencoder beside *variant*, fetching it once per revision.

    Byte-identical across variants, so it is downloaded once into the first variant that asks and
    copied afterwards -- 160 MB per variant is a waste, and a second download is a second chance
    for the bytes to differ.
    """
    target = weights_dir / "moebius" / variant / revision / "torch-fp32-vae.pth"
    if target.is_file():
        return

    cached = next((path for path in
                   (weights_dir / "moebius").glob(f"*/{revision}/torch-fp32-vae.pth")
                   if path.is_file()), None)
    if cached is not None:
        print(f"  {variant:8s} autoencoder already fetched for this revision; copying")
        shutil.copy2(cached, target)
        shutil.copy2(cached.with_name("NOTICE-vae"), target.with_name("NOTICE-vae"))
        return

    sha256 = download_verified(f"{_HF}/{VAE_REPO}/resolve/main/{VAE_FILE}", target,
                               blob_sha256(VAE_REPO, VAE_FILE),
                               label="vae", width=8, detail="MIT")
    # A second NOTICE beside the first: two works, two sets of terms, and folding them into one
    # file would make the shorter chain look as well-attested as the longer one.
    target.with_name("NOTICE-vae").write_text(_vae_notice(sha256))


def main() -> int:
    args = variant_parser(__doc__, ROOT / "weights", revision=REVISION).parse_args()
    wanted = args.variants or list(VARIANTS)

    unknown = [name for name in wanted if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variant(s) {unknown}. Published: {list(VARIANTS)}. "
                         f"Upstream also publishes {list(SKIPPED)}, which mozo does not carry.")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)
    print(f"\n{len(wanted)} variant(s) in {args.weights_dir / 'moebius'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
