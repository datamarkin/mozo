from typing import Union

import numpy as np
import cv2
from PIL import Image

from ..utils import load_image

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

    Self-contained adapter with complete configuration.
    """

    # Complete variant configuration (single source of truth)
    SUPPORTED_VARIANTS = {
        'base': {'device': 'cpu'},
        'capfilt-large': {'device': 'cpu'},
    }

    # Model IDs mapped to variants (implementation detail)
    _MODEL_IDS = {
        'base': 'Salesforce/blip-vqa-base',
        'capfilt-large': 'Salesforce/blip-vqa-capfilt-large',
    }

    def __init__(self, variant='base', **kwargs):
        """
        Initialize the BLIP VQA model from Hugging Face.

        Args:
            variant: Model variant to use ('base', 'capfilt-large')
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

        print(f"Loading BLIP VQA model (variant: {variant})...")
        
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForQuestionAnswering.from_pretrained(model_id)
        
        self.device = 'cuda' if device == 'gpu' else 'cpu'
        self.model.to(self.device)

        print(f"BLIP VQA model loaded successfully on device '{self.device}'.")

    def predict(self, image: Union[str, np.ndarray], prompt: str):
        """
        Run visual question answering with the BLIP model.

        Args:
            image: File path (str) or numpy array (BGR format)
            prompt (str): The question to ask about the image.

        Returns:
            dict: An OpenAI-compatible dictionary containing the answer.
        """
        image = load_image(image)
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
