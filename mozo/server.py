"""HTTP surface: upload an image, get the model's answer back.

Two kinds of endpoint. ``/predict`` runs a model; ``/models`` says what there is. Nothing here
manages memory -- a model that is loaded stays loaded, so there is no cleanup call for anyone
to forget to make.

Handlers are ``def`` rather than ``async def`` on purpose: inference is seconds of blocking CPU
or GPU work, so FastAPI runs each in its threadpool instead of stalling the event loop.
"""

from __future__ import annotations

import os

# Set before torch is imported -- adapters load lazily, so this is always in time. Some ops have
# no MPS kernel, and without the fallback they raise instead of running on the CPU.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response

from . import __version__
from .image import load_image
from .manager import ModelManager
from .registry import ENCODES, MODEL_REGISTRY, PROMPTED, get_model_info
from .weights import NotPublished, published

app = FastAPI(
    title="Mozo Model Server",
    description="Computer vision models served from a pip install.",
    version=__version__,
)

# Built at import: a manager is an empty dict and a lock, and loads nothing until asked. There is
# no startup work to defer, and no window where a request can arrive before it exists.
app.state.model_manager = ModelManager()

_STATIC = Path(__file__).parent / "static"



@app.get("/", summary="Health check")
def health_check():
    """Report that the server is up, and which models are resident."""
    return {
        "status": "ok",
        "version": __version__,
        "loaded_models": app.state.model_manager.loaded(),
    }


@app.get("/test-ui", summary="Interactive test page")
def serve_test_ui():
    """Serve the browser page for trying models by hand."""
    return FileResponse(_STATIC / "test_ui.html", media_type="text/html")


@app.get("/static/example.jpg", summary="Example image")
def serve_example_image():
    """Serve the image the test page starts with."""
    image_path = _STATIC / "example.jpg"
    if not image_path.is_file():
        raise HTTPException(status_code=404, detail="Example image not found at mozo/static/example.jpg")
    return FileResponse(image_path, media_type="image/jpeg")


# --- Prediction ---

def _coordinates(value: str, count: int, what: str) -> List[float]:
    """Parse a comma-separated pixel coordinate from a query parameter.

    Args:
        value: The raw parameter, e.g. ``"820,640"``.
        count: How many numbers it must carry.
        what: What to call it in the error, e.g. ``"point"``.

    Returns:
        The parsed numbers.

    Raises:
        ValueError: If it is not *count* numbers. The message says what was given, because a
            caller who wrote ``?point=820, 640`` and got "expected 2 numbers" learns nothing
            about which of their several points was the problem.
    """
    try:
        numbers = [float(part) for part in value.split(",")]
    except ValueError:
        raise ValueError(f"{what} {value!r} is not numbers; give {count} separated by commas")
    if len(numbers) != count:
        raise ValueError(f"{what} {value!r} has {len(numbers)} numbers; {what} takes {count}")
    return numbers


def _depth_response(depth: np.ndarray, unit: Optional[str]) -> Response:
    """Encode a depth map as a 16-bit PNG, with what is needed to read it back in the headers.

    An 8-bit PNG is what the old adapter returned, and it is the wrong answer here: six of the
    nine Depth Anything V2 variants predict metres, and metres are the entire point of choosing
    one. Quantising them to 256 levels and calling it an image discards the measurement.

    16-bit is lossless enough to be honest -- over an 80 m range one step is 1.2 mm -- and PNG
    stays viewable in any tool. The values are min-max normalised into the full 16-bit range and
    the endpoints travel in the headers, so a client recovers the original with

        depth = X-Depth-Min + png / 65535 * (X-Depth-Max - X-Depth-Min)

    ``unit`` is ``"metres"`` or ``None``; ``None`` means inverse depth on an arbitrary per-image
    scale, where larger is nearer. The server does not decide that a unitless map is metres any
    more than mozo decides a class id is a name.
    """
    low, high = float(depth.min()), float(depth.max())
    # One pass into one buffer. The arithmetic spelling of this allocates a full-size float32
    # temporary per operator -- four of them, ~34 MB of churn on a 1920x1281 map -- for the same
    # rounding, and needs a special case for a flat map that NORM_MINMAX handles itself.
    scaled = cv2.normalize(depth, None, 0, 65535, cv2.NORM_MINMAX, dtype=cv2.CV_16U)

    success, encoded = cv2.imencode(".png", scaled)
    if not success:
        raise HTTPException(status_code=500, detail="Could not encode the depth map.")

    # Response rather than StreamingResponse: the bytes are already in hand, so streaming only
    # buys an extra copy and a chunked transfer with no Content-Length.
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={
            "X-Depth-Unit": unit or "none",
            "X-Depth-Min": repr(low),
            "X-Depth-Max": repr(high),
        },
    )


@app.post("/predict/{family}/{variant}", summary="Run a model")
def predict(
    family: str,
    variant: str,
    file: UploadFile = File(..., description="Image file to process."),
    threshold: Optional[float] = Query(
        None,
        description="Confidence floor. Omitted, the family's own published default applies -- "
                    "these differ, and naming them here would be a second copy of a number each "
                    "adapter already owns.",
    ),
    labels: Optional[str] = None,
    text: Optional[List[str]] = Query(
        None,
        description="Concept to look for, for prompted models. Repeat it to ask for several: "
                    "?text=car&text=person. Not comma-separated -- a prompt is free text and "
                    "may contain a comma of its own.",
    ),
    point: Optional[List[str]] = Query(
        None,
        description="A click, as x,y in the image's own pixels, for promptable models. Repeat "
                    "it to give several: ?point=820,640&point=900,700. Each needs a ?label=.",
    ),
    label: Optional[List[int]] = Query(
        None,
        description="1 to include the matching ?point=, 0 to exclude it. One per point, in the "
                    "same order. Required with ?point= -- guessing between include and exclude "
                    "returns a confident mask of the wrong thing.",
    ),
    box: Optional[str] = Query(
        None,
        description="A box, as x1,y1,x2,y2 in the image's own pixels, for promptable models.",
    ),
    name: Optional[str] = Query(
        None,
        description="What to call what you pointed at, for promptable models. Omitted, "
                    "detections carry class_name=null -- the model does not know what it "
                    "segmented and mozo will not invent it.",
    ),
    multimask: bool = Query(
        True,
        description="For promptable models, return three candidate masks ranked by the model's "
                    "predicted IoU rather than one. Worth keeping on for a single click, which "
                    "is genuinely ambiguous about whether you meant the part or the whole.",
    ),
):
    """Run one model over one image.

    Args:
        family: Model family, e.g. ``rfdetr``.
        variant: Variant within it, e.g. ``nano``.
        file: The image.
        threshold: Confidence floor, for detection models. Omitted, each family's own default
            applies -- these differ, and this endpoint does not have an opinion about which is
            right for a model it is only routing to.
        labels: Comma-separated class names overriding the model's own, e.g. ``hardhat,vest``.
        text: The concept to look for, for prompted models -- ``cow``, ``yellow school bus``.
            Repeat the parameter to ask for several in one request; they share the image encode.
            Required by every task in :data:`~mozo.registry.PROMPTED`, ignored by the rest. Deliberately *not*
            comma-separated the way ``labels`` is: a prompt is free text, and ``"a person,
            holding a mug"`` is one concept rather than two.
        point: A click, as ``x,y`` in the image's own pixels, for promptable models. Repeatable.
        label: ``1`` to include the matching *point*, ``0`` to exclude it. One per point, in the
            same order, and required with them.
        box: A box, as ``x1,y1,x2,y2`` in the image's own pixels, for promptable models.
        name: What to call what was pointed at. Omitted, detections carry ``class_name=null``.
        multimask: Return three candidate masks ranked by predicted IoU rather than one.

    Returns:
        Detections as JSON, or a depth map as a 16-bit PNG -- see :func:`_depth_response`.
    """
    # Whether a model exists is answerable from the registry alone -- no adapter import, no
    # torch, no weights, no image decode. Answering it first makes an unknown name free, and
    # keeps it from being confused with a model that exists and failed to load. The registry
    # raises with the available names, so its message is the answer rather than a second copy
    # of one.
    try:
        task = get_model_info(family, variant)["task_type"]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Settled here rather than beside the call: nothing about it needs the image or the model,
    # and asking first is what keeps a forgotten parameter from costing a decode and a
    # multi-gigabyte load before the caller is told they forgot it.
    # Deliberately the adapter's own rule, not a looser one. A guard that accepted
    # ``?text=car&text=`` would hand the caller the same 400 from the adapter, but only after
    # decoding the image and loading the weights -- which is the cost this exists to avoid.
    if task in PROMPTED and (not text or any(not t.strip() for t in text)):
        raise HTTPException(
            status_code=400,
            detail=f"{family} is prompted: pass ?text=... naming what to look for, and give "
                   "every one a concept. Repeat it for several: ?text=car&text=person.",
        )

    # Same reasoning as above: settled before the image is decoded and the weights are loaded,
    # because a forgotten prompt should not cost a multi-gigabyte load to be told about. Parsed
    # here as well as checked, so the adapter is handed numbers rather than strings.
    points = marks = corners = None
    if task == "promptable_segmentation":
        if not point and box is None:
            raise HTTPException(
                status_code=400,
                detail=f"{family} is promptable: point at something. Pass ?point=x,y with a "
                       "?label=1, or ?box=x1,y1,x2,y2, or both.",
            )
        if len(point or []) != len(label or []):
            raise HTTPException(
                status_code=400,
                detail=f"{len(point or [])} point(s) and {len(label or [])} label(s): give one "
                       "?label= per ?point=, 1 to include it and 0 to exclude it.",
            )
        try:
            points = [_coordinates(p, 2, "point") for p in point] if point else None
            corners = _coordinates(box, 4, "box") if box is not None else None
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        marks = list(label) if label else None

    # Decode through the same function the Python API uses, so both entry points agree on
    # channel order. A second decoder here is how the two drift apart without anyone noticing.
    try:
        image = load_image(file.file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Could not read or decode the image file: {e}")

    # Existence in the registry was settled above; publication was not. A registered variant
    # mozo ships no weights for still cannot load, and that is the caller's answer rather than
    # a genuine failure -- so it is separated out below.
    try:
        model = app.state.model_manager.get_model(family, variant)
    except NotPublished as e:
        # The catalogue does not offer it: no such variant, revision or runtime. A permanent fact
        # rather than a failure of this request -- no retry helps -- so it is the caller's answer,
        # the same 404 an unknown name gets, and not a 500 blaming the server for its own
        # catalogue. Deliberately not its parent ``WeightsError``, which also covers a download
        # that failed, a mirror serving the wrong bytes and an offline cache miss: those are the
        # server's problem, a retry may fix them, and they keep the 500 below.
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # How a task is called and how its result is encoded are the same decision, so each task
    # is named exactly once. The registry declares which one applies; a task with no branch
    # here is a family the registry knows and this endpoint has not been taught to serve.
    # Forwarded only when the caller said one, so an adapter's own default is what applies
    # otherwise. Restating 0.5 here would silently override OWLv2's 0.1 and return nothing.
    floor = {} if threshold is None else {"threshold": threshold}
    try:
        if task == "object_detection":
            parsed = [n.strip() for n in labels.split(",") if n.strip()] if labels else None
            return JSONResponse(
                content=model.predict(image, labels=parsed or None, **floor).to_dict())
        if task in PROMPTED:
            return JSONResponse(content=model.predict(image, text, **floor).to_dict())
        if task == "promptable_segmentation":
            return JSONResponse(content=model.predict(
                image, points=points, labels=marks, boxes=corners,
                multimask_output=multimask, name=name,
            ).to_dict())
        if task == "text_recognition":
            return JSONResponse(content=model.predict(image).to_dict())
        if task == "depth_estimation":
            return _depth_response(model.predict(image), model.unit)
    except HTTPException:
        raise  # already carries the status it wants; do not re-wrap it as a 500
    except ValueError as e:
        # An adapter raising ValueError is rejecting the caller's arguments, which is a 400 --
        # for every family, rather than something each one arranges separately.
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    raise HTTPException(
        status_code=501, detail=f"{family} performs {task!r}, which this endpoint cannot encode.")


# --- Encoding ---

@app.post("/encode/{family}/{variant}", summary="Embed an image or a phrase")
def encode(
    family: str,
    variant: str,
    file: Optional[List[UploadFile]] = File(
        None, description="Image file(s) to embed. Repeat for a batch."),
    text: Optional[List[str]] = Query(
        None, description="Phrase(s) to embed. Repeat for a batch: ?text=a+cat&text=a+dog."),
):
    """Return the vectors a model works from, rather than an answer.

    Some models represent an image and a phrase in one shared space, so that a dot product between
    two vectors says how well they match. That is what makes a corpus embedded once searchable by
    words afterwards -- but only through a vector database, which is the caller's. mozo produces
    the vectors and stops there.

    Send **either** images or phrases, not both: they are two different towers and one call runs
    one of them.

    Args:
        family: A family that embeds. ``GET /models`` reports which, and what each accepts.
        variant: Variant within it.
        file: Image(s). Repeat the part for a batch.
        text: Phrase(s). Repeat the parameter for a batch.

    Returns:
        ``{"model", "revision", "dim", "embeddings"}``. The vectors are L2-normalised, so a dot
        product between any two is a cosine similarity.

        ``model`` and ``revision`` name the weights that produced them, and are not decoration: a
        vector is only comparable against others from the same weights, so a stored index is tied
        to them. Recording which is what makes an index re-embeddable later.
    """
    # Answered from the registry, so an unknown or non-embedding family costs no image decode and
    # no download. /predict can afford its 501 at the bottom because every registered task has an
    # arm there; here the reverse holds, and a late refusal would download a checkpoint to say no.
    try:
        get_model_info(family, variant)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    kinds = ENCODES.get(family)
    if kinds is None:
        raise HTTPException(
            status_code=501,
            detail=f"{family} does not produce embeddings. Families that do: {sorted(ENCODES)}.")

    # FastAPI can require a parameter but not "exactly one of these two", so this is the one check
    # the route has to make itself.
    if bool(file) == bool(text):
        raise HTTPException(
            status_code=400,
            detail=f"send either images or ?text=, not both and not neither. "
                   f"{family} accepts: {sorted(kinds)}.")

    wanted = "image" if file else "text"
    if wanted not in kinds:
        raise HTTPException(
            status_code=400,
            detail=f"{family} does not embed {wanted}; it accepts {sorted(kinds)}.")

    try:
        model = app.state.model_manager.get_model(family, variant)
    except NotPublished as e:
        # The catalogue does not offer it: no such variant, revision or runtime. A permanent fact
        # rather than a failure of this request -- no retry helps -- so it is the caller's answer,
        # the same 404 an unknown name gets, and not a 500 blaming the server for its own
        # catalogue. Deliberately not its parent ``WeightsError``, which also covers a download
        # that failed, a mirror serving the wrong bytes and an offline cache miss: those are the
        # server's problem, a retry may fix them, and they keep the 500 below.
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # ``encode_<kind>`` is the convention ENCODES is written in and the one the adapter test
    # asserts against, so the route reads it rather than keeping a second copy as two branches.
    payload = [load_image(part.file.read()) for part in file] if wanted == "image" else list(text)
    try:
        vectors = getattr(model, f"encode_{wanted}")(payload)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")

    return JSONResponse(content={
        "model": f"{family}/{variant}",
        # Read off the model rather than re-derived from the manifest. The two cannot disagree
        # today, because this route always loads the newest; they would the moment it takes a
        # ?revision=, and a vector stamped with the wrong revision is undetectable afterwards.
        "revision": model.revision,
        "dim": int(vectors.shape[1]),
        "embeddings": vectors.tolist(),
    })


# --- Discovery ---

@app.get("/models", summary="List every model")
def list_models():
    """Every family and its variants.

    Answered from the registry, so it costs no imports and no weights. This is the whole
    catalogue: a per-family or per-variant endpoint would only be this response, filtered.

    Residency is not mixed in. It belongs to ``/models/loaded``, and reporting it here would
    mean taking a model id apart to recover the variant -- which puts the id format in a second
    module, where it can drift from the one that composes it.
    """
    return {
        family: {
            "task_type": entry["task_type"],
            "description": entry["description"],
            "variants": entry["variants"],
            # Which of those can actually run. A family may register a variant mozo publishes no
            # weights for -- a licence that forbids redistribution, or an architecture that only
            # runs against a checkpoint you supply -- and a catalogue that does not say so sends
            # every caller to a 404 to find out. Read off the manifest that ships in the wheel, so
            # this still costs no network and no weights.
            "published": [v for v in entry["variants"] if published(family, v)],
            # The two capability flags a caller cannot infer from the task name alone. Served
            # rather than restated: the browser page needs both and cannot import the registry,
            # and a copy it keeps in step by hand is a copy that drifts.
            "prompted": entry["task_type"] in PROMPTED,
            # What /encode will accept, if anything. Empty for almost every family, and the only
            # way to discover the second route without calling it and reading a 501 back.
            "encodes": sorted(ENCODES.get(family, ())),
        }
        for family, entry in MODEL_REGISTRY.items()
    }


@app.get("/models/loaded", summary="List resident models")
def list_loaded_models():
    """Model ids currently in memory, in the order they were first asked for.

    Nothing is evicted, so this is everything this process has served since it started.
    """
    return {"models": app.state.model_manager.loaded()}
