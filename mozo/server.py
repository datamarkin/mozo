import io
import cv2
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from typing import Optional
from PIL import Image

# Import model manager, factory, and registry utilities
from .manager import ModelManager
from .factory import ModelFactory
from .registry import get_available_families, get_model_info

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# --- FastAPI App ---
app = FastAPI(
    title="Mozo Model Server",
    description="Dynamic model serving API with lazy loading and lifecycle management.",
    version="0.2.0"
)

# --- Model Manager Setup ---
@app.on_event("startup")
def setup_manager():
    """
    Initialize model manager and factory (no models loaded yet - they load on-demand).

    This is much faster than the old approach which loaded all models at startup.
    Models will be loaded automatically when first requested.
    """
    print("[Server] Initializing model manager and factory...")
    app.state.model_manager = ModelManager()
    app.state.model_factory = ModelFactory()
    print("[Server] Model manager and factory ready. Models will be loaded on-demand.")

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

@app.post("/predict/{family}/{variant}",
          summary="Run Model Prediction",
          description="Upload an image and get predictions from any available model variant.")
async def predict(
    family: str,
    variant: str,
    file: UploadFile = File(..., description="Image file to process."),
    mask: Optional[UploadFile] = File(None, description="Mask file for inpainting models."),
    prompt: str = "Describe this image in detail.",
    bearer_token: Optional[str] = None
):
    """
    Universal prediction endpoint supporting all model families and variants.

    Args:
        family: Model family (e.g., 'detectron2', 'datamarkin', 'stability_inpainting')
        variant: Model variant (e.g., 'mask_rcnn_R_50_FPN_3x', 'wings-v4')
                For datamarkin, variant is the training_id
        file: Image file to process
        mask: Mask file for inpainting models
        prompt: Text prompt for generative models
        bearer_token: Authentication token for datamarkin models (optional)

    Returns:
        JSON response with predictions or an image

    Examples:
        POST /predict/detectron2/mask_rcnn_R_50_FPN_3x
        POST /predict/datamarkin/wings-v4?bearer_token=xxx
        POST /predict/qwen3_vl/2b-thinking?prompt=What is in this image?
    """
    if not hasattr(app.state, "model_manager"):
        raise HTTPException(status_code=503, detail="Server is starting up, model manager not initialized.")

    # Read and decode image
    try:
        contents = await file.read()
        image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read or decode the image file: {e}")

    # Read and decode mask if provided
    mask_image = None
    if mask:
        try:
            mask_contents = await mask.read()
            mask_image = Image.open(io.BytesIO(mask_contents))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read or decode the mask file: {e}")

    # Get or load model (lazy loading happens here)
    try:
        if family == 'datamarkin':
            # Pass bearer_token for datamarkin models
            model = app.state.model_manager.get_model(family, variant, bearer_token=bearer_token)
        else:
            model = app.state.model_manager.get_model(family, variant)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Run prediction
    try:
        if family in ['qwen2.5_vl', 'qwen3_vl']:
            results = model.predict(image, prompt=prompt)
        elif family == 'stability_inpainting':
            if mask_image is None:
                raise HTTPException(status_code=400, detail="Inpainting model requires a mask file.")
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            results = model.predict(image=pil_image, mask=mask_image, prompt=prompt)
        else:
            results = model.predict(image)

        # Handle different return types
        if hasattr(results, 'save'):  # It's a PIL Image
            buffer = io.BytesIO()
            results.save(buffer, format="PNG")
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="image/png")
        elif hasattr(results, 'to_dict'):  # It's a PixelFlow Detections object
            return JSONResponse(content=results.to_dict())
        else:  # It's a dict (VLM results)
            return JSONResponse(content=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# --- Model Management Endpoints ---

@app.get("/models",
         summary="List Available Models",
         description="Get all available model families with their variants and loaded status.")
def list_available_models():
    """
    List all available model families with their variants.

    Variants are discovered from adapters (single source of truth).
    Also returns which variants are currently loaded in memory.

    Returns:
        dict: Available models organized by family, with variant lists, descriptions, and loaded status
    """
    if not hasattr(app.state, "model_factory"):
        raise HTTPException(status_code=503, detail="Server is starting up, model factory not initialized.")

    families = get_available_families()
    loaded_models = app.state.model_manager.list_loaded_models() if hasattr(app.state, "model_manager") else []
    result = {}

    for family in families:
        try:
            # Get variants from adapter (single source of truth)
            variants = app.state.model_factory.get_available_variants(family)

            # Get family info from registry
            info = get_model_info(family)

            # Find which variants are loaded
            loaded_variants = [
                variant for variant in variants
                if f"{family}/{variant}" in loaded_models
            ]

            result[family] = {
                'task_type': info['task_type'],
                'description': info['description'],
                'num_variants': len(variants),
                'variants': variants,
                'loaded': loaded_variants,
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

    Variants are discovered from the adapter's SUPPORTED_VARIANTS (single source of truth).

    Args:
        family: Model family name (e.g., 'detectron2', 'paddleocr')

    Returns:
        dict: Family name and list of available variants

    Example:
        GET /models/detectron2/variants
        Returns: {"family": "detectron2", "variants": ["mask_rcnn_R_50_FPN_3x", ...]}
    """
    if not hasattr(app.state, "model_factory"):
        raise HTTPException(status_code=503, detail="Server is starting up, model factory not initialized.")

    try:
        variants = app.state.model_factory.get_available_variants(family)
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