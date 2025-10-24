import numpy as np
import cv2
from PIL import Image

try:
    from transformers import BlipProcessor, BlipForQuestionAnswering
    import torch
except ImportError:
    print("="*50)
    print("ERROR: Transformers and PyTorch are not installed.")
    print("Please install them with: pip install transformers torch")
    print("="*50)
    raise

from ..utils import create_openai_response

class BlipVqaPredictor:
    """
    Adapter for the Salesforce BLIP model for Visual Question Answering.
    """

    SUPPORTED_VARIANTS = {
        'base': {
            'description': 'BLIP VQA base model - General-purpose visual question answering',
            'model_id': 'Salesforce/blip-vqa-base',
        },
        'capfilt-large': {
            'description': 'BLIP VQA large model with caption filtering - Higher accuracy',
            'model_id': 'Salesforce/blip-vqa-capfilt-large',
        },
    }

    def __init__(self, variant='base', device='cpu', **kwargs):
        """
        Initialize the BLIP VQA model from Hugging Face.

        Args:
            variant: Model variant to use ('base', 'capfilt-large')
            device: Device to run on - 'cpu' or 'gpu'

        Raises:
            ValueError: If variant is not supported
        """
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Choose from: {list(self.SUPPORTED_VARIANTS.keys())}"
            )

        self.variant = variant
        model_id = self.SUPPORTED_VARIANTS[variant]['model_id']

        print(f"Loading BLIP VQA model (variant: {variant})...")
        
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(model_id)
        
        self.device = 'cuda' if device == 'gpu' else 'cpu'
        self.model.to(self.device)

        print(f"BLIP VQA model loaded successfully on device '{self.device}'.")

    def predict(self, image: np.ndarray, prompt: str):
        """
        Run visual question answering with the BLIP model.

        Args:
            image (np.ndarray): The input image in BGR format (from OpenCV).
            prompt (str): The question to ask about the image.

        Returns:
            dict: An OpenAI-compatible dictionary containing the answer.
        """
        print(f"Running BLIP VQA prediction (question: '{prompt}')...")

        # Convert image from BGR (OpenCV) to RGB (PIL)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # Preprocess inputs
        inputs = self.processor(pil_image, prompt, return_tensors="pt").to(self.device)
        
        # Generate output
        output_ids = self.model.generate(**inputs)
        
        # Decode the answer
        answer = self.processor.decode(output_ids[0], skip_special_tokens=True)

        print(f"BLIP VQA answer: '{answer}'")

        # Format the output into the OpenAI standard
        return create_openai_response(f"blip-vqa-{self.variant}", answer)
