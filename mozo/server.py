import io
import cv2
import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse

# Import model manager and registry utilities
from .manager import ModelManager
from .registry import get_available_families, get_available_variants, get_model_info

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
    Initialize model manager (no models loaded yet - they load on-demand).

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
    image_path = Path(__file__).parent.parent / "vision" / "example.jpg"

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Example image not found at vision/example.jpg")

    return FileResponse(image_path, media_type="image/jpeg")


# --- Prediction Endpoints ---

@app.post("/predict/{family}/{variant}",
          summary="Run Model Prediction",
          description="Upload an image and get predictions from any available model variant.")
async def predict(
    family: str,
    variant: str,
    file: UploadFile = File(..., description="Image file to process."),
    prompt: str = "Describe this image in detail."
):
    """
    Universal prediction endpoint supporting all model families and variants.

    Args:
        family: Model family (e.g., 'detectron2', 'depth_anything', 'qwen2.5_vl')
        variant: Model variant (e.g., 'mask_rcnn_R_50_FPN_3x', 'small', '7b-instruct')
        file: Image file to process
        prompt: Text prompt for vision-language models (used by qwen2.5_vl)

    Returns:
        JSON response with predictions (format depends on model type)

    Examples:
        POST /predict/detectron2/mask_rcnn_R_50_FPN_3x
        POST /predict/detectron2/faster_rcnn_X_101_32x8d_FPN_3x
        POST /predict/depth_anything/small
        POST /predict/depth_anything/large
        POST /predict/qwen2.5_vl/7b-instruct?prompt=What objects are in this image?
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

    # Get or load model (lazy loading happens here)
    try:
        model = app.state.model_manager.get_model(family, variant)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    # Run prediction
    try:
        # Vision-language models need prompt parameter
        if family == 'qwen2.5_vl':
            results = model.predict(image, prompt=prompt)
        else:
            results = model.predict(image)

        # Handle different return types
        if hasattr(results, 'save'):  # It's a PIL Image (depth map)
            buffer = io.BytesIO()
            results.save(buffer, format="PNG")
            buffer.seek(0)
            return StreamingResponse(buffer, media_type="image/png")

        elif hasattr(results, 'to_dict'):  # It's a PixelFlow Detections object
            # PixelFlow's to_dict() now properly serializes numpy arrays to base64/lists
            return JSONResponse(content=results.to_dict())

        else:  # It's a dict (VLM results)
            return JSONResponse(content=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# --- Model Management Endpoints ---

@app.get("/models",
         summary="List Available Models",
         description="Get all available model families and their variants.")
def list_available_models():
    """
    List all available model families and their variants.

    Returns:
        dict: Available models organized by family, with variant lists and descriptions
    """
    families = get_available_families()
    result = {}

    for family in families:
        variants = get_available_variants(family)
        info = get_model_info(family)
        result[family] = {
            'task_type': info['task_type'],
            'description': info['description'],
            'num_variants': len(variants),
            'variants': variants
        }

    return result


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