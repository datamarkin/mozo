"""The workflow runtime over HTTP.

Four endpoints: what nodes there are, whether a document is valid, run it, run it and watch. They
are mounted by ``mozo.server``, which is the only place in mozo that names this package.

Handlers are ``def`` rather than ``async def``, deliberately and for the same reason the model
endpoints are: a workflow is seconds of blocking CPU or GPU work, so FastAPI runs each in its
threadpool instead of stalling the event loop. The implementation this replaces got that wrong --
it ran the whole graph on the loop -- and it is the kind of wrong that only shows up under a second
concurrent request.

Validation is construction. :class:`~mozo.workflow.graph.Workflow` refuses a document that names a
node that does not exist, wires ports whose types disagree, leaves an input unfed or contains a
cycle, so ``/validate`` builds one and reports what building it said. Nothing here re-checks any of
that, which is why there is no second opinion to keep in step.

**What travels is not this module's to decide.** Turning a node's output into JSON is
:mod:`mozo.workflow.wire`'s, because it is a property of the port types rather than of HTTP. What
is left here is requests: which endpoint, what it accepts, and how long an uploaded file lives.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Literal, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from .graph import Workflow
from .registry import catalogue
from .wire import serialise

router = APIRouter(prefix="/workflow", tags=["workflow"])

#: The built editor. Its source is ``ui/`` at the repository root, which never ships -- the same
#: arrangement as ``tools/``, which produces ``weights/``. npm is needed to change the editor and
#: never to install mozo.
_EDITOR = Path(__file__).parent / "static"


@router.get("", summary="The workflow editor", include_in_schema=False)
def editor():
    """Serve the editor.

    Registered without a trailing slash so that ``/workflow`` is the address, and the assets it
    asks for are relative to it -- which is what lets the whole thing be mounted under a prefix
    without the page knowing.
    """
    page = _EDITOR / "index.html"
    if not page.is_file():
        raise HTTPException(
            status_code=404,
            detail="The editor is not built. Run `npm install && npm run build` in ui/.")
    return FileResponse(page, media_type="text/html")


@router.get("/assets/{name}", summary="The editor's own files", include_in_schema=False)
def asset(name: str):
    """Serve one of the editor's built files.

    A route rather than a mounted ``StaticFiles``, because mounting is the application's to do and
    ``mozo/server.py`` has exactly one line about this package. Vite emits flat names, so one
    segment is the whole of it -- and the resolved path is checked to be inside the directory, so a
    name full of ``..`` reaches nothing.
    """
    path = (_EDITOR / "assets" / name).resolve()
    if not path.is_file() or (_EDITOR / "assets").resolve() != path.parent:
        raise HTTPException(status_code=404, detail=f"no such file: {name}")
    return FileResponse(path)


@router.get("/nodes", summary="List the nodes a workflow can be built from")
def list_nodes():
    """Every node, with its ports and parameters.

    Derived from the functions that implement them, so what this returns and what runs cannot
    disagree -- see :mod:`mozo.workflow.node`.
    """
    return {"nodes": catalogue()}


@router.post("/validate", summary="Check a workflow without running it")
def validate(workflow: str = Form(..., description="The workflow document, as JSON")):
    """Say whether a document is a workflow, and if not, why not."""
    try:
        built = _build(workflow)
    except HTTPException as refusal:
        return {"valid": False, "error": refusal.detail}
    return {"valid": True, "order": list(built.order), "terminals": list(built.terminals)}


@router.post("/run", summary="Run a workflow")
def run(
    workflow: str = Form(..., description="The workflow document, as JSON"),
    inputs: str = Form("{}", description="Parameter overrides, as a JSON object"),
    include: Literal["terminals", "all"] = Form(
        "terminals", description="Which nodes' outputs to send back"),
    file: Optional[UploadFile] = File(None, description="An image or video to run on, if it is "
                                                        "not already on the server"),
):
    """Run a workflow and send back what it produced.

    Args:
        include: ``"terminals"``, the ends of the graph, or ``"all"``, every node.

    Sending everything is what :meth:`Workflow.run` does, because in Python an intermediate result
    costs a reference. Over HTTP it costs an encode: on one 1920x1281 photograph a five-node
    workflow takes 27 ms to run and 347 ms to encode, and answers with 21.7 MB of which four fifths
    is intermediates nobody asked for. At 4K that is 52 MB and half a gigabyte of resident memory
    per request, in a threadpool forty deep.

    So the default is the ends, and looking inside is a thing you ask for. The editor asks; a batch
    job over full-resolution frames should not have to.
    """
    built = _build(workflow)
    terminals = built.terminals
    # Asked before the file is written down, so a workflow with nowhere to put one is refused
    # having copied nothing. Deriving the name costs microseconds; spooling costs the whole upload.
    into = _destination(built) if file is not None else None
    upload = _spool(file)

    results = {}
    try:
        for event in _events(built, inputs, upload, into):
            if event.status == "failed":
                # A run that failed is not a run that returned nothing. Answering 200 with an empty
                # results dict is indistinguishable from success to any client that does not go
                # looking, and the reason the node gave would be thrown away.
                raise HTTPException(status_code=422, detail=event.error)
            if event.status == "completed" and (include == "all" or event.node in terminals):
                # Serialised as it arrives, so a node's output is not held past the one it feeds.
                # Combined with the engine dropping a wire once its last reader has run, a
                # five-node chain over a 4K photograph peaks at 70 MB instead of 182 MB.
                results[event.node] = serialise(built.steps[event.node].spec, event.output)
    finally:
        _discard(upload)

    return {"results": results, "terminals": list(terminals)}


@router.post("/stream", summary="Run a workflow, reporting each node as it goes")
def stream(
    workflow: str = Form(..., description="The workflow document, as JSON"),
    inputs: str = Form("{}", description="Parameter overrides, as a JSON object"),
    include: Literal["terminals", "all"] = Form(
        "terminals", description="Which nodes' outputs to send back"),
    file: Optional[UploadFile] = File(None, description="An image or video to run on, if it is "
                                                        "not already on the server"),
):
    """Run a workflow as server-sent events, one per node starting and finishing.

    The document is built and the overrides are settled before the response begins, so a document
    that is not a workflow -- or an override naming no parameter -- is a 400 rather than an error
    delivered inside a stream that has already claimed success.

    Args:
        include: What ``/run`` means by it, and for the same measured reason. Watching a workflow
            go by is not a reason to be sent every intermediate at full resolution.
    """
    built = _build(workflow)
    terminals = built.terminals

    # Settled here rather than inside ``events`` so that a refused override is still a 400 before
    # the response begins, which is what this endpoint promises. Everything that can refuse runs
    # before the file is written down, so the only spooled file that exists is one a started run
    # owns -- and it outlives this function, which is why the generator is what removes it.
    into = _destination(built) if file is not None else None
    upload = _spool(file)
    started = _events(built, inputs, upload, into)

    def events():
        try:
            for event in started:
                reported = {"node": event.node, "status": event.status}
                if event.status == "completed" and (include == "all" or event.node in terminals):
                    reported["output"] = serialise(built.steps[event.node].spec, event.output)
                if event.error:
                    reported["error"] = event.error
                yield _event(reported)

                if event.status == "failed":
                    # The failure is the end. Saying "done" after it would leave a consumer to
                    # decide which of the two the server meant.
                    return
        except Exception as error:  # a bad override, or something a node could not recover from
            yield _event({"status": "failed", "error": str(error)})
        else:
            yield _event({"done": True})
        finally:
            _discard(upload)      # however the response ended, including a client hanging up

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


def _events(built: Workflow, inputs: str, upload: Optional[Path], into: Optional[str]):
    """Start *built* running, turning a refused override into a 400 before anything executes.

    ``stream`` settles the overrides before it returns its iterator, so this catches a bad one here
    rather than part-way through a response that has already claimed success.
    """
    try:
        return built.stream(**_overrides(inputs, upload, into))
    except KeyError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _build(document: str) -> Workflow:
    """Build a workflow from its JSON, turning every refusal into a 400 that says which."""
    try:
        return Workflow.from_dict(json.loads(document))
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail=f"not JSON: {error}") from error
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _overrides(inputs: str, upload: Optional[Path], into: Optional[str]) -> dict:
    """Parameter overrides for this run, with an uploaded file folded in as one of them.

    *into* is the parameter the file is the value of, which
    :attr:`~mozo.workflow.graph.Workflow.file_parameter` derived from the annotations rather than
    from a name written down here. This used to write to a parameter literally called ``image``,
    which meant the transport knew one node's vocabulary and broke when that node was renamed.
    """
    try:
        overrides = json.loads(inputs)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=400, detail=f"inputs is not JSON: {error}") from error
    if not isinstance(overrides, dict):
        raise HTTPException(status_code=400, detail="inputs must be a JSON object")

    if upload is not None:
        overrides[into] = str(upload)
    return overrides


def _destination(built: Workflow) -> str:
    """Where an upload goes, as a 400 rather than a traceback where there is no such place."""
    try:
        return built.file_parameter
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _spool(upload: Optional[UploadFile]) -> Optional[Path]:
    """*upload* as a file on disk, or None if there was none.

    **A path rather than the bytes, because a video cannot be bytes.** ``cv2.VideoCapture`` takes a
    filename, a device index or a URL and has no memory-buffer form at all, so an upload that never
    touches the disk can only ever have been an image -- which is exactly the limit that made the
    editor's picker refuse an ``.mp4``. Writing it down first is what lets one input node read
    either kind from either place.

    The suffix is kept, because it is what selects the decoder. A file arriving with no name is
    decoded as an image, which is what the picker's own default would have done.
    """
    if upload is None:
        return None
    suffix = Path(upload.filename or "").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        shutil.copyfileobj(upload.file, handle)
    return Path(handle.name)


def _discard(upload: Optional[Path]) -> None:
    """Remove a spooled upload once the run that reads it is over.

    ``missing_ok`` so that a file already gone cannot replace the run's own error with this one.
    """
    if upload is not None:
        upload.unlink(missing_ok=True)


def _event(payload: dict) -> str:
    """One server-sent event."""
    return f"data: {json.dumps(payload)}\n\n"
