from PIL import Image
import numpy as np
import cv2

try:
    from transformers import pipeline
except ImportError:
    print("="*50)
    print("ERROR: `transformers` is not installed.")
    print("Please install it with: `pip install transformers`")
    print("="*50)
    raise

class DepthAnythingPredictor:
    """
    Universal Depth Anything adapter - handles all model size variants.
    Supports small, base, and large variants of Depth Anything V2.

    Self-contained adapter with complete configuration.
    """

    # Complete variant configuration (single source of truth)
    SUPPORTED_VARIANTS = {
        'small': {},
        'base': {},
        'large': {},
    }

    # Mapping of variant names to HuggingFace model IDs (implementation detail)
    _MODEL_IDS = {
        'small': 'depth-anything/Depth-Anything-V2-Small-hf',
        'base': 'depth-anything/Depth-Anything-V2-Base-hf',
        'large': 'depth-anything/Depth-Anything-V2-Large-hf',
    }

    def __init__(self, variant="small", **kwargs):
        """
        Initialize Depth Anything predictor with specific model size variant.

        Args:
            variant: Model size variant - 'small', 'base', or 'large'
                    small:  Fastest, lowest memory, good for real-time applications
                    base:   Balanced speed and accuracy
                    large:  Best accuracy, slower, higher memory usage
            **kwargs: Override parameters (reserved for future use)

        Raises:
            ValueError: If variant is not supported
        """
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Supported variants: {list(self.SUPPORTED_VARIANTS.keys())}"
            )

        self.variant = variant
        model_name = self._MODEL_IDS[variant]

        print(f"Loading Depth Anything model (variant: {variant}, model: {model_name})...")
        self.pipe = pipeline(task="depth-estimation", model=model_name)
        print(f"Depth Anything model loaded successfully (variant: {variant}).")

    def predict(self, image: np.ndarray) -> Image.Image:
        """
        Runs depth estimation on an image.

        Args:
            image: A numpy array representing the input image in BGR format (from cv2).

        Returns:
            A PIL.Image object representing the depth map.
        """
        print("Running depth estimation...")
        # The pipeline expects a PIL Image in RGB format.
        # cv2 reads images as BGR, so we need to convert it.
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        result = self.pipe(pil_image)
        
        # The pipeline returns a dictionary, the depth map is in the "depth" key
        depth_map = result["depth"]
        
        print("Depth estimation complete.")
        return depth_map
