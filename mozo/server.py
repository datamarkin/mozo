import cv2
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response, FileResponse
from typing import Optional

# Import model manager, factory, and registry utilities
from . import __version__
from .manager import ModelManager
from .registry import get_available_families, get_available_variants, get_model_info
from .utils import load_image

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- FastAPI App ---
app = FastAPI(
    title="Mozo Model Server",
    description="Dynamic model serving API with lazy loading and lifecycle management.",
    version=__version__
)

# --- Model Manager Setup ---
@app.on_event("startup")
def setup_manager():
    """
    Initialize the model manager (no models loaded yet - they load on-demand).

    This is much faster than the old approach which loaded all models at startup.
    Models will be loaded automatically when first requested.
    """
    print("[Server] Initializing model manager...")
    app.state.model_manager = ModelManager()
    print("[Server] Model manager ready. Models will be loaded on-demand.")

# --- API Endpoints ---
@app.get("/", summary="Health Check", description="Check if the API server is ready.")
def health_check():
    """
    Health check endpoint.

    Note: Models are loaded on-demand, so this just checks if the manager is initialized.
    """
    manager_ready = hasattr(app.state, "model_manager")
    if not manager_ready:
        return {"status": "error", "message": "Server is starting up, model manager not yet initialized."}
    return {
        "status": "ok",
        "message": "Server is running with dynamic model management.",
        "loaded_models": app.state.model_manager.list_loaded_models()
    }


# --- Test UI ---

@app.get("/test-ui", summary="Test UI", description="Serve interactive testing interface.")
def serve_test_ui():
    """
    Serve the interactive test UI for model testing.

    This provides a user-friendly web interface to:
    - Upload images
    - Select models dynamically
    - View prediction results
    """
    html_path = Path(__file__).parent / "static" / "test_ui.html"
    return FileResponse(html_path, media_type="text/html")


@app.get("/static/example.jpg", summary="Example Image", description="Serve example test image.")
def serve_example_image():
    """Serve the default example image for testing."""
    image_path = Path(__file__).parent / "static" / "example.jpg"

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Example image not found at mozo/static/example.jpg")

    return FileResponse(image_path, media_type="image/jpeg")


# --- Prediction Endpoints ---

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


@app.post("/predict/{family}/{variant}",
          summary="Run Model Prediction",
          description="Upload an image and get predictions from any available model variant.")
def predict(
    family: str,
    variant: str,
    file: UploadFile = File(..., description="Image file to process."),
    threshold: float = 0.5,
    labels: Optional[str] = None,
):
    """
    Universal prediction endpoint supporting all model families and variants.

    Args:
        family: Model family (e.g., 'rfdetr', 'depth_anything_v2')
        variant: Model variant (e.g., 'nano', 'indoor-small')
        file: Image file to process
        threshold: Confidence threshold for detection models
        labels: Comma-separated class labels for detection models (e.g., "hardhat,vest,person").
                Overrides the model's default labels when provided.

    Returns:
        Detections as JSON, or a depth map as a 16-bit PNG -- see :func:`_depth_response`.

    Examples:
        POST /predict/rfdetr/nano?threshold=0.5
        POST /predict/depth_anything_v2/indoor-small
    """
    if not hasattr(app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Server is starting up, model manager not initialized.")

    # Whether a model exists is answerable from the registry alone -- no adapter import, no
    # torch, no weights, no image decode. Answering it first makes an unknown name free, and
    # keeps it from being confused with a model that exists and failed to load. The registry
    # raises with the available names, so its message is the answer rather than a second copy
    # of one.
    try:
        task = get_model_info(family, variant)["task_type"]
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

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
        if task == "depth_estimation":
            return _depth_response(model.predict(image), model.unit)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    raise HTTPException(
        status_code=501, detail=f"{family} performs {task!r}, which this endpoint cannot encode.")


# --- Model Management Endpoints ---

@app.get("/models",
         summary="List Available Models",
         description="Get all available model families with their variants and loaded status.")
def list_available_models():
    """
    List all available model families with their variants.

    Also returns which variants are currently loaded in memory.

    Returns:
        dict: Available models organized by family, with variant lists, descriptions, and loaded status
    """
    loaded_models = set(
        app.state.model_manager.list_loaded_models() if hasattr(app.state, "model_manager") else []
    )
    result = {}

    for family in get_available_families():
        try:
            info = get_model_info(family)
            variants = info['variants']

            result[family] = {
                'task_type': info['task_type'],
                'description': info['description'],
                'num_variants': len(variants),
                'variants': variants,
                'loaded': [v for v in variants if f"{family}/{v}" in loaded_models],
            }
        except Exception as e:
            # If adapter fails to load, return error state
            result[family] = {
                'error': str(e),
                'variants': [],
                'loaded': [],
            }

    return result


@app.get("/models/{family}/variants",
         summary="Get Model Variants",
         description="Get available variants for a specific model family.")
def get_family_variants(family: str):
    """
    Get available variants for a specific model family.

    Args:
        family: Model family name (e.g., 'rfdetr', 'depth_anything_v2')

    Returns:
        dict: Family name and list of available variants

    Example:
        GET /models/rfdetr/variants
        Returns: {"family": "rfdetr", "variants": ["nano", "small", ...]}
    """
    try:
        variants = get_available_variants(family)
        return {
            "family": family,
            "variants": variants,
            "num_variants": len(variants)
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/models/loaded",
         summary="List Loaded Models",
         description="Get currently loaded models in memory.")
def list_loaded_models():
    """
    List currently loaded models.

    Returns:
        dict: Loaded model IDs and their usage information
    """
    if not hasattr(app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Model manager not initialized.")

    loaded = app.state.model_manager.list_loaded_models()
    info = app.state.model_manager.get_model_info()

    return {
        "loaded_count": len(loaded),
        "models": info
    }


@app.get("/models/{family}/{variant}/info",
         summary="Get Model Info",
         description="Get detailed information about a specific model variant.")
def get_model_details(family: str, variant: str):
    """
    Get detailed information about a specific model variant.

    Args:
        family: Model family name
        variant: Model variant name

    Returns:
        dict: Model information including parameters and load status
    """
    try:
        info = get_model_info(family, variant)

        # Add load status
        if hasattr(app.state, "model_manager"):
            model_id = f"{family}/{variant}"
            load_info = app.state.model_manager.get_model_info(model_id)
            info['load_status'] = load_info
        else:
            info['load_status'] = {'loaded': False}

        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/models/{family}/{variant}/unload",
          summary="Unload Model",
          description="Manually unload a model to free memory.")
def unload_model(family: str, variant: str):
    """
    Manually unload a specific model to free memory.

    Args:
        family: Model family name
        variant: Model variant name

    Returns:
        dict: Unload status
    """
    if not hasattr(app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Model manager not initialized.")

    success = app.state.model_manager.unload_model(family, variant)

    if success:
        return {
            "status": "unloaded",
            "family": family,
            "variant": variant,
            "model_id": f"{family}/{variant}"
        }
    else:
        return {
            "status": "not_loaded",
            "family": family,
            "variant": variant,
            "message": "Model was not loaded, nothing to unload."
        }


@app.post("/models/cleanup",
          summary="Cleanup Inactive Models",
          description="Unload models that haven't been used recently.")
def cleanup_inactive_models(inactive_seconds: int = 600):
    """
    Cleanup models that haven't been used in the specified time period.

    Args:
        inactive_seconds: Time threshold in seconds (default: 600 = 10 minutes)

    Returns:
        dict: Cleanup results
    """
    if not hasattr(app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Model manager not initialized.")

    count = app.state.model_manager.cleanup_inactive_models(inactive_seconds)

    return {
        "status": "completed",
        "models_unloaded": count,
        "inactive_threshold_seconds": inactive_seconds
    }