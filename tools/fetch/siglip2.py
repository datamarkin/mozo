#!/usr/bin/env python3
"""Fetch SigLIP 2's published checkpoints into the local ``weights/`` tree.

Bootstrap tooling. It runs on a machine you control, never ships, and is the only place the
upstream URLs live -- the vendored package carries none of them, because where a checkpoint came
from is a fact about publishing, not about inference.

    python tools/fetch/siglip2.py                  # every variant
    python tools/fetch/siglip2.py base-224
    python tools/fetch/siglip2.py --derive-vocab   # rebuild the tokenizer asset

**The download hash comes from Hugging Face's API, not from a README.** Each blob's sha256 is read
from ``/api/models/<repo>?blobs=true`` at fetch time, so a mismatch means the bytes changed under
us rather than that someone transcribed a digest wrongly.

**Safetensors only, so mozo repacks.** Not one of the fifteen repositories publishes a
``pytorch_model.bin`` -- OWLv2's trick of placing Google's ``.bin`` unchanged is simply not
available here. This reads the safetensors and writes an ordinary ``.pth``, which is the same
repack ``tools/fetch/clip.py`` does for a TorchScript archive and for the same reason: the version
risk is taken here, once, on a machine we control. ``safetensors`` is a dependency of *this script*
and not of mozo; nothing in ``mozo/`` imports it.

**Two variants are sharded.** ``giant-256`` and ``giant-384`` ship as two safetensors files plus an
index; the other thirteen are one file. The index is read when present and the shards are merged,
so what mozo publishes has the same shape for all fifteen. That is a packaging detail of Google's,
not a difference between the models.

**Two ``-naflex`` checkpoints are not carried.** They run at variable resolution through a
different image tower -- ``Siglip2Model`` rather than ``SiglipModel`` -- which ``siglip2_deploy``
does not build. They are named here so adding them is a decision rather than a discovery.

The output is ``weights/siglip2/<variant>/<revision>/torch-fp32.pth`` and a NOTICE. The LICENSE
beside them is not written by this script: a licence is part of what is published, so it lives in
the weights tree like any other artifact.
"""

from __future__ import annotations

import gzip
import json
import sys
import urllib.request
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

from common import digest, download_verified, require_licence, variant_parser  # noqa: E402

from mozo.vendors.siglip2_deploy.config import SPECS  # noqa: E402

_HF = "https://huggingface.co"

#: The revision this family was published under.
REVISION = "2026-08-23"

#: Where Apache-2.0's canonical text lives. Google ships no LICENSE file in these repositories,
#: so the text is placed once per revision rather than copied from upstream.
LICENCE_SOURCE = "https://www.apache.org/licenses/LICENSE-2.0.txt"

#: Where the tokenizer asset lands, and which repository it is derived from. One vocabulary serves
#: all fifteen -- verified identical by sha256 -- so it is taken from the smallest.
VOCAB_ASSET = ROOT / "mozo" / "vendors" / "siglip2_deploy" / "assets" / "gemma_bpe.json.gz"
VOCAB_SOURCE = "google/siglip2-base-patch16-224"


def repository(spec) -> str:
    """Where Google publishes *spec*.

    Composed here rather than on ``Spec`` because an address is a fact about publishing, not about
    inference, and the vendored package ships to users who never fetch anything.
    """
    return f"google/siglip2-{spec.upstream}"


def hub_json(url: str, timeout: int = 60):
    """GET one JSON document from the Hub.

    A timeout because the download beside it has one: a metadata call that hangs forever stalls a
    publishing run exactly as a stalled download would.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def blobs(repo: str) -> dict[str, str]:
    """Filename -> sha256, for the files in *repo* that Hugging Face records one for.

    Only LFS-tracked files carry a content hash; the small ones carry a git blob id, which is a
    hash of different bytes and useless for verifying a download. Every safetensors file is LFS, so
    every file this script actually verifies is in here -- and one that is not will raise a
    ``KeyError`` naming itself rather than being waved through unverified.
    """
    listing = hub_json(f"{_HF}/api/models/{repo}?blobs=true")
    return {
        entry["rfilename"]: entry["lfs"]["sha256"]
        for entry in listing["siblings"] if entry.get("lfs")
    }


def shard_names(repo: str, available: dict[str, str]) -> list[str]:
    """Which safetensors files make up *repo*, in the order its index names them.

    One file for thirteen variants; for the two ``giant`` ones, whatever
    ``model.safetensors.index.json`` lists. Read from the index rather than guessed from the
    ``-of-`` suffix, so a repository that re-shards does not silently lose a tensor.
    """
    if "model.safetensors" in available:
        return ["model.safetensors"]
    index = hub_json(f"{_HF}/{repo}/resolve/main/model.safetensors.index.json")
    return sorted(set(index["weight_map"].values()))


def _notice(variant: str, repo: str, digests: dict[str, str]) -> str:
    """Attribution that travels with the weights, as Apache-2.0 asks."""
    listed = "\n".join(f"           {name}  sha256 {digest}" for name, digest in digests.items())
    return (
        f"SigLIP 2 -- {variant}\n"
        "sigmoid-loss image-text pretraining, two towers in one shared space\n\n"
        "Copyright 2025 The Google Research Authors.\n"
        "Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem,\n"
        "Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer,\n"
        "Ye Xia, Basil Mustafa, Olivier Henaff, Jeremiah Harmsen, Andreas Steiner,\n"
        "Xiaohua Zhai.\n\n"
        f"Source:  {_HF}/{repo}\n"
        "Project: https://github.com/google-research/big_vision\n"
        "Paper:   SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic\n"
        "         Understanding, Localization, and Dense Features (2025), arXiv:2502.14786\n"
        "Licence: Apache-2.0 (full text in the LICENSE file beside this one)\n\n"
        "The authors state of their SigLIP releases: these models are not official Google\n"
        "products and were trained and released for research purposes.\n\n"
        "Google's own release is JAX/Flax. This file derives from the PyTorch conversion Google\n"
        "publishes beside it on Hugging Face, under the same terms. It is not the byte stream\n"
        "Google serves: tools/fetch/siglip2.py verifies the safetensors against the sha256 the\n"
        "Hub records, then writes the same tensors back out as an ordinary checkpoint. No tensor\n"
        "is altered, renamed, cast or dropped. The source files are:\n"
        f"{listed}\n\n"
        "The tokenizer vocabulary shipped inside mozo derives from tokenizer.model in this same\n"
        "repository. It is Gemma's vocabulary, and Gemma's own weights are published under the\n"
        "Gemma Terms of Use behind a gate -- but Google publishes this copy here, ungated, in a\n"
        "repository licensed Apache-2.0. That is an affirmative grant by the rights holder on the\n"
        "artifact actually taken, not an inference from silence.\n"
    )


def fetch(variant: str, revision: str, weights_dir: Path) -> None:
    """Download one variant's checkpoint, verify it, repack it, and place it with its notice."""
    repo = repository(SPECS[variant])
    target = weights_dir / "siglip2" / variant / revision / "torch-fp32.pth"
    target.parent.mkdir(parents=True, exist_ok=True)

    available = blobs(repo)
    names = shard_names(repo, available)
    digests, parts = {}, []
    for name in names:
        part = target.with_name(name)
        digests[name] = download_verified(
            f"{_HF}/{repo}/resolve/main/{name}", part, available[name],
            label=variant, width=13, detail="Apache-2.0")
        parts.append(part)

    if not target.is_file():
        from safetensors.torch import load_file

        print(f"  {variant:13s} repacking {len(parts)} safetensors -> state dict")
        state: dict[str, torch.Tensor] = {}
        for part in parts:
            state.update(load_file(part))
        torch.save(state, target)
    for part in parts:
        part.unlink(missing_ok=True)

    (target.parent / "NOTICE").write_text(_notice(variant, repo, digests))
    require_licence(target.parent, "Apache-2.0", LICENCE_SOURCE)


def case_folding() -> dict[str, str]:
    """Every codepoint the reference lowercases, and what it lowercases to.

    Carried rather than deferred to ``str.lower()`` because Python's case tables are a property of
    the interpreter, not of this package. Python 3.10 ships Unicode 13.0.0 and the reference's Rust
    normaliser ships a later one, and they disagree about **95 codepoints** -- the case mappings
    added for Vithkuqi, Latin Extended-D and Cyrillic Extended-C in Unicode 14 and 15. Every one is
    a character Python leaves alone and the reference folds.

    That would make a prompt's token ids depend on which Python is running, silently, which for an
    embedding model is worse than it sounds: two callers on different interpreters would write
    different vectors for the same phrase into the same index. A table costs 20 KB and makes the
    answer a property of the package.

    Needs ``tokenizers`` installed, which is bootstrap-only -- nothing under ``mozo/`` imports it.
    """
    from tokenizers import normalizers

    lowercase = normalizers.Lowercase()
    folding = {}
    for codepoint in range(0x110000):
        character = chr(codepoint)
        if 0xD800 <= codepoint <= 0xDFFF:            # lone surrogates are not encodable text
            continue
        folded = lowercase.normalize_str(character)
        if folded != character:
            folding[str(codepoint)] = folded
    return folding


def derive_vocab() -> None:
    """Rebuild the tokenizer asset from the published ``tokenizer.json``.

    Committed to the repository rather than fetched, because it has to be in the wheel before a
    user runs anything -- the same arrangement as CLIP's ``bpe_simple_vocab_16e6.txt.gz``.

    Merges are stored as pairs of ids rather than pairs of pieces: it is what the merge loop wants
    at runtime, and it avoids escaping questions in an asset whose pieces contain newlines and
    tabs. The added tokens are carried too. They cannot be recovered from the vocabulary alone --
    they are a *subset* of it, singled out as matching before normalisation, and which 249 of the
    256,000 those are is not something any rule recovers.
    """
    published = hub_json(f"{_HF}/{VOCAB_SOURCE}/resolve/main/tokenizer.json", timeout=300)

    model = published["model"]
    vocab, merges = model["vocab"], model["merges"]
    pieces = [piece for piece, _ in sorted(vocab.items(), key=lambda item: item[1])]

    added = published["added_tokens"]
    if any(token["normalized"] or token["lstrip"] or token["rstrip"] for token in added):
        raise SystemExit("an added token wants normalising or stripping; the tokenizer assumes not")

    folding = case_folding()
    payload = json.dumps({
        "pieces": pieces,
        "merges": [[vocab[left], vocab[right]] for left, right in merges],
        "added": [token["content"] for token in added],
        "lower": folding,
    }, ensure_ascii=False).encode()

    VOCAB_ASSET.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0: gzip stamps the current time into its header by default, which would make the
    # committed asset differ on every rebuild and its recorded sha256 meaningless.
    VOCAB_ASSET.write_bytes(gzip.compress(payload, 9, mtime=0))
    print(f"{VOCAB_ASSET.relative_to(ROOT)}  {VOCAB_ASSET.stat().st_size:,} bytes")
    print(f"  {len(pieces):,} pieces, {len(merges):,} merges, {len(added)} added tokens, "
          f"{len(folding):,} case mappings")
    print(f"  sha256 {digest(VOCAB_ASSET)}")


def main() -> int:
    parser = variant_parser(__doc__, ROOT / "weights", revision=REVISION)
    parser.add_argument("--derive-vocab", action="store_true",
                        help="rebuild the tokenizer asset and exit")
    args = parser.parse_args()

    if args.derive_vocab:
        # The shared parser's other arguments describe fetching weights, and this mode writes into
        # the package instead. Saying so beats accepting them and doing something else.
        if args.variants:
            raise SystemExit("--derive-vocab writes the tokenizer asset and takes no variants")
        derive_vocab()
        return 0

    wanted = args.variants or list(SPECS)
    unknown = [name for name in wanted if name not in SPECS]
    if unknown:
        raise SystemExit(f"unknown variants: {unknown}. Known: {list(SPECS)}")

    for variant in wanted:
        fetch(variant, args.revision, args.weights_dir)

    print(f"\n{len(wanted)} checkpoints in {args.weights_dir}, all Apache-2.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
