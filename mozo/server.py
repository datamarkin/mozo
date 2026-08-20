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
from .registry import MODEL_REGISTRY, get_model_info

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
    threshold: float = 0.5,
    labels: Optional[str] = None,
    text: Optional[List[str]] = Query(
        None,
        description="Concept to look for, for prompted models. Repeat it to ask for several: "
                    "?text=car&text=person. Not comma-separated -- a prompt is free text and "
                    "may contain a comma of its own.",
    ),
):
    """Run one model over one image.

    Args:
        family: Model family, e.g. ``rfdetr``.
        variant: Variant within it, e.g. ``nano``.
        file: The image.
        threshold: Confidence floor, for detection models.
        labels: Comma-separated class names overriding the model's own, e.g. ``hardhat,vest``.
        text: The concept to look for, for prompted models -- ``cow``, ``yellow school bus``.
            Repeat the parameter to ask for several in one request; they share the image encode.
            Required by prompted models, ignored by the rest. Deliberately *not* comma-separated
            the way ``labels`` is: a prompt is free text, and ``"a person, holding a mug"`` is
            one concept rather than two.

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
    if task == "concept_segmentation" and (not text or any(not t.strip() for t in text)):
        raise HTTPException(
            status_code=400,
            detail=f"{family} is prompted: pass ?text=... naming what to look for, and give "
                   "every one a concept. Repeat it for several: ?text=car&text=person.",
        )

    # Decode through the same function the Python API uses, so both entry points agree on
    # channel order. A second decoder here is how the two drift apart without anyone noticing.
    try:
        image = load_image(file.file.read())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Could not read or decode the image file: {e}")

    # Existence was settled above, so anything raised here is a genuine failure to load a
    # model that does exist.
    try:
        model = app.state.model_manager.get_model(family, variant)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # How a task is called and how its result is encoded are the same decision, so each task
    # is named exactly once. The registry declares which one applies; a task with no branch
    # here is a family the registry knows and this endpoint has not been taught to serve.
    try:
        if task == "object_detection":
            parsed = [n.strip() for n in labels.split(",") if n.strip()] if labels else None
            return JSONResponse(
                content=model.predict(image, threshold=threshold, labels=parsed or None).to_dict())
        if task == "concept_segmentation":
            return JSONResponse(
                content=model.predict(image, text, threshold=threshold).to_dict())
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
        }
        for family, entry in MODEL_REGISTRY.items()
    }


@app.get("/models/loaded", summary="List resident models")
def list_loaded_models():
    """Model ids currently in memory, in the order they were first asked for.

    Nothing is evicted, so this is everything this process has served since it started.
    """
    return {"models": app.state.model_manager.loaded()}
