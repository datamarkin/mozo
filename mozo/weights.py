"""Turn a model name into a verified local file.

Every published artifact mozo can run is listed in ``manifest.json``, which ships inside the
package. Resolution therefore needs no network and no configuration: the manifest that came with
this version of mozo is the definitive answer to what bytes a given model refers to.

    >>> from mozo.weights import resolve
    >>> resolve("rfdetr", "small")                      # doctest: +SKIP
    PosixPath('/Users/you/.cache/mozo/rfdetr/small/2026-08-18/torch-fp32.pth')

A model absent from the manifest has no weights mozo publishes -- either because its licence
forbids redistribution, or because it is only ever run against checkpoints you bring yourself.
That is not an error condition to work around; pass the file to the adapter directly.

Environment:
    ``MOZO_CACHE``     Where downloads live. Default ``~/.cache/mozo``.
    ``MOZO_BASE_URL``  Serve artifacts from a mirror instead of the manifest's ``base_url``.
                       A ``file://`` URL pointing at a ``weights/`` tree works, which is how an
                       air-gapped host can be fed from removable media.
    ``MOZO_OFFLINE``   Set to ``1`` to refuse downloads. Missing files raise an error naming the
                       exact path, URL and hash, so they can be placed by hand.
"""

from __future__ import annotations

__all__ = ["WeightsError", "artifacts", "cache_dir", "companions", "manifest", "resolve"]

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path
from typing import Any

#: Sits next to this module and ships in the wheel.
_MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"

_DOWNLOAD_CHUNK_BYTES = 1 << 16
_TIMEOUT_SECONDS = 60.0

#: Fetched alongside every artifact, so a cached model is never separated from its terms.
_LICENCE_KEY = "LICENSE"
_NOTICE_KEY = "NOTICE"
# Artifacts that ship *with* whichever one you asked for, rather than being a choice. The licence
# is always published; a NOTICE only where the upstream terms ask for attribution to travel with
# the copy, which for CC-BY-NC weights they do.
_ACCOMPANYING = (_LICENCE_KEY, _NOTICE_KEY)
#: Of those, the ones every revision must publish. A model may have nothing to attribute, but it
#: always has terms.
_REQUIRED = frozenset({_LICENCE_KEY})

_manifest: dict[str, Any] | None = None


class WeightsError(RuntimeError):
    """Raised when a model cannot be resolved, downloaded, or verified."""


def manifest() -> dict[str, Any]:
    """Return the bundled manifest, read once and kept for the life of the process."""
    global _manifest
    if _manifest is None:
        _manifest = json.loads(_MANIFEST_PATH.read_text())
    return _manifest


def cache_dir() -> Path:
    """Return the download cache root: ``$MOZO_CACHE`` if set, else ``~/.cache/mozo``."""
    override = os.environ.get("MOZO_CACHE", "").strip()
    base = Path(override) if override else Path.home() / ".cache" / "mozo"
    return base.expanduser()


def _base_url() -> str:
    """Return the URL artifacts are served from, honouring a ``MOZO_BASE_URL`` mirror."""
    override = os.environ.get("MOZO_BASE_URL", "").strip()
    return (override or manifest()["base_url"]).rstrip("/")


def _lookup(family: str, variant: str, revision: str | None) -> tuple[str, dict[str, Any]]:
    """Return the resolved revision name and its manifest entry.

    Raises:
        WeightsError: If the model is not published, or the requested revision does not exist.
    """
    model_id = f"{family}/{variant}"
    model = manifest()["models"].get(model_id)
    if model is None:
        raise WeightsError(
            f"mozo publishes no weights for {model_id!r}. Either its licence does not permit "
            f"redistribution, or it runs only against checkpoints you supply -- pass the "
            f"checkpoint path to the adapter instead."
        )

    name = revision or model["latest"]
    entry = model["revisions"].get(name)
    if entry is None:
        available = ", ".join(sorted(model["revisions"]))
        raise WeightsError(f"{model_id}: no revision {name!r}. Published revisions: {available}")
    return name, entry


def _artifact(entry: dict[str, Any], model_id: str, revision: str, key: str) -> dict[str, Any]:
    """Return one artifact record from a revision.

    Raises:
        WeightsError: If the revision does not publish that artifact.
    """
    artifact = entry["artifacts"].get(key)
    if artifact is None:
        available = ", ".join(sorted(k for k in entry["artifacts"] if k not in _ACCOMPANYING))
        raise WeightsError(
            f"{model_id} revision {revision} does not publish {key!r}. Available: {available}"
        )
    return artifact


def _fetch(url: str, target: Path, *, size: int, sha256: str) -> None:
    """Download *url* to *target*, verifying size and hash before it takes its final name.

    The transfer lands on a sibling ``.part`` file and is hashed as it streams, so an interrupted
    or corrupted download can never be mistaken for a complete cache entry.

    Raises:
        WeightsError: If the transfer fails, or the bytes do not match the manifest.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    written = 0
    last_report = time.monotonic()

    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT_SECONDS) as response, partial.open("wb") as out:
            for chunk in iter(lambda: response.read(_DOWNLOAD_CHUNK_BYTES), b""):
                out.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                now = time.monotonic()
                if size and now - last_report > 1.0:
                    print(f"  {target.name}  {100 * written / size:5.1f}%", flush=True)
                    last_report = now
    except OSError as error:
        partial.unlink(missing_ok=True)
        raise WeightsError(f"failed to download {url}: {error}") from error

    if written != size or digest.hexdigest() != sha256:
        partial.unlink(missing_ok=True)
        raise WeightsError(
            f"{target.name} does not match the manifest "
            f"(expected {size} bytes / {sha256}, got {written} bytes / {digest.hexdigest()}). "
            f"The mirror at {url} may be stale or corrupt."
        )
    partial.replace(target)


def _obtain(artifact: dict[str, Any], revision_dir: Path) -> Path:
    """Return the local path for *artifact*, downloading it if the cache does not already hold it.

    Raises:
        WeightsError: If the file is absent and ``MOZO_OFFLINE`` forbids fetching it.
    """
    target = revision_dir / Path(artifact["path"]).name
    if target.is_file():
        return target

    url = f"{_base_url()}/{artifact['path']}"
    if os.environ.get("MOZO_OFFLINE", "").strip() not in ("", "0"):
        raise WeightsError(
            f"MOZO_OFFLINE is set and {target} is not cached.\n"
            f"  fetch:  {url}\n"
            f"  place:  {target}\n"
            f"  sha256: {artifact['sha256']}"
        )

    print(f"[mozo] downloading {target.name} ({artifact['size'] / 1e6:.1f} MB)", flush=True)
    _fetch(url, target, size=artifact["size"], sha256=artifact["sha256"])
    return target


def artifacts(family: str, variant: str, *, revision: str | None = None) -> list[str]:
    """Return the artifact keys a revision publishes, e.g. ``["onnx-fp32", "torch-fp32"]``.

    ``LICENSE`` and ``NOTICE`` are omitted -- they ship with every artifact rather than being
    ones you choose.
    Callers use this to find out what a model can actually be run as before asking for it.

    Raises:
        WeightsError: If the model or revision is not published.

    Examples:
        >>> artifacts("rfdetr", "small")  # doctest: +SKIP
        ['torch-fp32']
    """
    _, entry = _lookup(family, variant, revision)
    return sorted(k for k in entry["artifacts"] if k not in _ACCOMPANYING)


def companions(family: str, variant: str, *, revision: str | None = None) -> list[str]:
    """Return the accompanying artifact keys a revision publishes.

    The complement of :func:`artifacts`, which filters these out because they ship with whatever
    you asked for rather than being a thing you can run. A caller that needs to check a model's
    terms travelled with it asks here, instead of walking the manifest -- its layout is this
    module's to know.

    Raises:
        WeightsError: If the model or revision is not published.

    Examples:
        >>> companions("rfdetr", "small")  # doctest: +SKIP
        ['LICENSE']
    """
    _, entry = _lookup(family, variant, revision)
    return sorted(k for k in entry["artifacts"] if k in _ACCOMPANYING)


def resolve(
    family: str,
    variant: str,
    key: str = "torch-fp32",
    *,
    revision: str | None = None,
) -> Path:
    """Return a local path to one published artifact, downloading and verifying it if needed.

    The artifact's licence is fetched into the same directory, along with its NOTICE where one
    is published, so a cached model always sits beside the terms it was published under and the
    attribution those terms require.

    Args:
        family: Model family, e.g. ``"rfdetr"``.
        variant: Variant within that family, e.g. ``"small"``.
        key: Which artifact, as :func:`artifacts` reports it. Runnable ones are named
            ``<runtime>-<precision>``; ``labels`` and ``LICENSE`` are not, which is why this is
            one opaque string rather than two halves that would have to be rejoined.
        revision: Published revision to pin. Defaults to the manifest's ``latest``.

    Returns:
        Path to the artifact inside the cache. Its revision is part of the path, so a pinned
        revision is honoured even when another is already cached.

    Raises:
        WeightsError: If the model, revision, or artifact is not published; if the download
            fails or does not match the manifest; or if it is absent while ``MOZO_OFFLINE``
            is set.

    Examples:
        >>> resolve("rfdetr", "small")                            # doctest: +SKIP
        >>> resolve("rfdetr", "small", "onnx-fp32")               # doctest: +SKIP
        >>> resolve("rfdetr", "small", revision="2026-08-18")     # doctest: +SKIP
    """
    revision_name, entry = _lookup(family, variant, revision)
    model_id = f"{family}/{variant}"

    artifact = _artifact(entry, model_id, revision_name, key)

    revision_dir = cache_dir() / family / variant / revision_name
    path = _obtain(artifact, revision_dir)
    for companion in _ACCOMPANYING:
        # The licence is required of every revision and _artifact says so if it is missing; a
        # NOTICE is published only where the terms ask for attribution to travel with the copy.
        published = entry["artifacts"].get(companion)
        if published is None and companion not in _REQUIRED:
            continue
        _obtain(_artifact(entry, model_id, revision_name, companion), revision_dir)
    return path
