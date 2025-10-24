import numpy as np
import cv2
from PIL import Image

try:
    from diffusers import StableDiffusionInpaintPipeline
    import torch
except ImportError:
    print("="*50)
    print("ERROR: Diffusers and PyTorch are not installed.")
    print("Please install them with: pip install diffusers torch")
    print("="*50)
    raise

class StabilityInpaintingPredictor:
    """
    Adapter for the Stability AI Stable Diffusion 2 Inpainting model.

    IMPORTANT: This model requires both an image and a mask as input.
    The mask should be a grayscale image where white areas indicate regions to inpaint.

    Self-contained adapter with complete configuration.
    """

    # Complete variant configuration (single source of truth)
    SUPPORTED_VARIANTS = {
        'default': {'device': 'cpu'},
    }

    # Model IDs mapped to variants (implementation detail)
    _MODEL_IDS = {
        'default': 'stabilityai/stable-diffusion-2-inpainting',
    }

    def __init__(self, variant='default', **kwargs):
        """
        Initialize the inpainting pipeline from Hugging Face.

        Args:
            variant: Model variant ('default')
            **kwargs: Override parameters (device)

        Raises:
            ValueError: If variant is not supported
        """
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Supported variants: {list(self.SUPPORTED_VARIANTS.keys())}"
            )

        # Merge defaults with overrides
        config = {**self.SUPPORTED_VARIANTS[variant], **kwargs}
        device = config.get('device', 'cpu')

        self.variant = variant
        model_id = self._MODEL_IDS[variant]

        print(f"Loading Stability AI Inpainting model (variant: {variant})...")

        self.pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == 'gpu' else torch.float32,
        )
        self.device = 'cuda' if device == 'gpu' else 'cpu'
        self.pipeline.to(self.device)
        
        print(f"Stability AI Inpainting model loaded successfully on device '{self.device}'.")

    def predict(self, image: np.ndarray, mask: np.ndarray, prompt: str):
        """
        Run inpainting on an image.

        Args:
            image: Input image as numpy array (H, W, 3) in BGR format (OpenCV standard)
            mask: Mask image as numpy array (H, W, 3) in BGR format (white areas will be inpainted)
            prompt: Text prompt to guide the inpainting

        Returns:
            PIL.Image.Image: The resulting inpainted image
        """
        print(f"Running Stability AI Inpainting prediction...")

        # Diffusers pipeline expects RGB format
        # Convert from BGR (OpenCV standard) to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Convert mask from BGR to RGB
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        pil_mask = Image.fromarray(mask_rgb)

        # Run inpainting pipeline
        result_image = self.pipeline(prompt=prompt, image=pil_image, mask_image=pil_mask).images[0]

        print("Inpainting prediction finished.")
        return result_image
