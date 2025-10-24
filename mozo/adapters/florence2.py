import numpy as np
import cv2
from PIL import Image

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch
except ImportError:
    print("="*50)
    print("ERROR: Transformers and PyTorch are not installed.")
    print("Please install them with: pip install transformers torch")
    print("="*50)
    raise

try:
    import pixelflow as pf
except ImportError:
    print("="*50)
    print("ERROR: PixelFlow is not installed.")
    print("Please install it with: pip install pixelflow")
    print("="*50)
    raise

from ..utils import create_openai_response

class Florence2Predictor:
    """
    Adapter for the Microsoft Florence-2 model.
    Supports multiple tasks like object detection, captioning, and OCR.
    """

    SUPPORTED_VARIANTS = {
        'detection': {
            'description': 'Object detection with bounding boxes',
            'task_prompt': '<OD>',
        },
        'segmentation': {
            'description': 'Instance segmentation (currently unimplemented)',
            'task_prompt': '<SEG>',
        },
        'captioning': {
            'description': 'Generate image captions',
            'task_prompt': '<CAPTION>',
        },
        'detailed_captioning': {
            'description': 'Generate detailed image captions',
            'task_prompt': '<DETAILED_CAPTION>',
        },
        'more_detailed_captioning': {
            'description': 'Generate very detailed image captions',
            'task_prompt': '<MORE_DETAILED_CAPTION>',
        },
        'ocr': {
            'description': 'Optical character recognition',
            'task_prompt': '<OCR>',
        },
        'ocr_with_region': {
            'description': 'OCR with region detection',
            'task_prompt': '<OCR_WITH_REGION>',
        },
    }

    def __init__(self, variant='detection', device='cpu', **kwargs):
        """
        Initialize the Florence-2 model from Hugging Face.

        Args:
            variant: Task variant to perform (e.g., 'detection', 'captioning')
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
        model_id = 'microsoft/Florence-2-large'

        print(f"Loading Florence-2 model (variant: {variant})...")

        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self.device = 'cuda' if device == 'gpu' else 'cpu'
        self.model.to(self.device)

        print(f"Florence-2 model loaded successfully on device '{self.device}'.")

    def _parse_od_output(self, result_text, image_size):
        """
        Parses the object detection output from Florence-2 into PixelFlow detections.
        Expected format: <OD> ... <box_1><box_2>...<box_n>

        NOTE: This method is not yet implemented. Florence-2 detection output parsing
        requires complex string parsing of the model's specific format.
        """
        raise NotImplementedError(
            "Florence-2 object detection parsing is not yet implemented. "
            "The detection output format is complex and requires custom parsing logic. "
            "Use 'captioning', 'ocr', or other text-based variants instead."
        )


    def predict(self, image: np.ndarray, prompt: str = None):
        """
        Run inference with the Florence-2 model.

        Args:
            image: Input image as numpy array (H, W, 3) in BGR format (OpenCV standard)
            prompt: Optional user prompt. If not provided, uses default task prompt for variant

        Returns:
            dict: OpenAI-compatible response for text tasks (captioning, OCR)
            pf.detections.Detections: For detection/segmentation tasks (if implemented)
        """
        task_prompt = self.SUPPORTED_VARIANTS[self.variant]['task_prompt']

        # Use user prompt if provided, otherwise use the default task prompt
        final_prompt = prompt if prompt else task_prompt

        print(f"Running Florence-2 prediction (variant: {self.variant})...")

        # Florence-2 processor expects RGB format
        # Convert from BGR (OpenCV standard) to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(text=final_prompt, images=pil_image, return_tensors="pt").to(self.device)
        
        generated_ids = self.model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3
        )
        
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Post-process to get the clean result
        # The specific parsing logic depends on the task
        # E.g., for <OD>, the output is '<OD> ... <box> ...'
        # We need to clean the prompt out of the generated text
        parsed_text = self.processor.post_process_generation(generated_text, task=task_prompt, image_size=pil_image.size)

        # Handle different output types based on the variant
        if self.variant in ['detection', 'segmentation']:
            # For vision tasks, parse the output and return Detections
            # NOTE: Both detection and segmentation are currently unimplemented
            return self._parse_od_output(parsed_text, pil_image.size)
        else:
            # For text tasks, return the OpenAI-compatible dictionary
            # The actual content is in the parsed_text dictionary
            text_content = parsed_text.get(task_prompt, "")
            return create_openai_response(f"florence-2-{self.variant}", text_content)
