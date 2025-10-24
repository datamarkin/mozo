from diffusers import StableDiffusionInpaintingPipeline
import torch
from PIL import Image

class StabilityInpaintingPredictor:
    """
    Adapter for the Stability AI Stable Diffusion 2 Inpainting model.
    """

    def __init__(self, variant='default', device='cpu', **kwargs):
        """
        Initialize the inpainting pipeline from Hugging Face.

        Args:
            variant: Model variant (not used for this model, but part of the adapter signature).
            device: Device to run on - 'cpu' or 'gpu'.
        """
        model_id = "stabilityai/stable-diffusion-2-inpainting"
        
        print(f"Loading Stability AI Inpainting model (variant: {variant})...")
        
        self.pipeline = StableDiffusionInpaintingPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == 'gpu' else torch.float32,
        )
        self.device = 'cuda' if device == 'gpu' else 'cpu'
        self.pipeline.to(self.device)
        
        print(f"Stability AI Inpainting model loaded successfully on device '{self.device}'.")

    def predict(self, image: Image.Image, mask: Image.Image, prompt: str):
        """
        Run inpainting on an image.

        Args:
            image (PIL.Image.Image): The source image.
            mask (PIL.Image.Image): The mask image (white areas will be inpainted).
            prompt (str): The text prompt to guide the inpainting.

        Returns:
            PIL.Image.Image: The resulting inpainted image.
        """
        print(f"Running Stability AI Inpainting prediction...")

        # The pipeline expects PIL images, which are passed directly from the server
        result_image = self.pipeline(prompt=prompt, image=image, mask_image=mask).images[0]

        print("Inpainting prediction finished.")
        return result_image
