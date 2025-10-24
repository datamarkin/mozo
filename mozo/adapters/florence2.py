import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import io
import numpy as np
import cv2

# Assuming pixelflow is available and has these classes
try:
    import pixelflow as pf
except ImportError:
    pf = None

from ..utils import create_openai_response

class Florence2Predictor:
    """
    Adapter for the Microsoft Florence-2 model.
    Supports multiple tasks like object detection, captioning, and OCR.
    """

    def __init__(self, variant='detection', device='cpu', **kwargs):
        """
        Initialize the Florence-2 model from Hugging Face.

        Args:
            variant (str): The task to perform (e.g., 'detection', 'captioning').
            device (str): Device to run on - 'cpu' or 'gpu'.
        """
        model_id = 'microsoft/Florence-2-large'
        
        print(f"Loading Florence-2 model (variant: {variant})...")

        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        
        self.device = 'cuda' if device == 'gpu' else 'cpu'
        self.model.to(self.device)
        
        self.variant = variant
        self.task_prompts = {
            'detection': '<OD>',
            'segmentation': '<SEG>',
            'captioning': '<CAPTION>',
            'detailed_captioning': '<DETAILED_CAPTION>',
            'more_detailed_captioning': '<MORE_DETAILED_CAPTION>',
            'ocr': '<OCR>',
            'ocr_with_region': '<OCR_WITH_REGION>',
        }
        
        if variant not in self.task_prompts:
            raise ValueError(f"Unsupported variant for Florence-2: '{variant}'. Choose from: {list(self.task_prompts.keys())}")

        print(f"Florence-2 model loaded successfully on device '{self.device}'.")

    def _parse_od_output(self, result_text, image_size):
        """
        Parses the object detection output from Florence-2 into PixelFlow detections.
        Expected format: <OD> ... <box_1><box_2>...<box_n>
        """
        if not pf:
            raise ImportError("PixelFlow is not installed. Cannot process object detection output.")

        # Placeholder parsing logic - this will need to be very robust
        # This is a simplified example. Real parsing is complex.
        try:
            bboxes = []
            labels = []
            # A real implementation would need to parse the specific format Florence-2 outputs,
            # which includes bounding box coordinates and labels within the string.
            # For now, we return an empty detection object as a placeholder.
            # Example of what would be needed:
            # for bbox_str, label_str in parse_florence_string(result_text):
            #     x1, y1, x2, y2 = bbox_str
            #     bboxes.append([x1, y1, x2, y2])
            #     labels.append(label_str)

            if not bboxes:
                return pf.detections.Detections()

            return pf.detections.Detections(
                xyxy=np.array(bboxes),
                class_name=np.array(labels)
            )
        except Exception as e:
            print(f"Error parsing Florence-2 detection output: {e}")
            return pf.detections.Detections()


    def predict(self, image: np.ndarray, prompt: str = None):
        """
        Run inference with the Florence-2 model.

        Args:
            image (np.ndarray): The input image in BGR format (from OpenCV).
            prompt (str, optional): An optional user prompt. If not provided, the default
                                    task prompt for the variant is used.

        Returns:
            dict or pixelflow.Detections or PIL.Image.Image
        """
        task_prompt = self.task_prompts[self.variant]
        
        # Use user prompt if provided, otherwise use the default task prompt
        final_prompt = prompt if prompt else task_prompt

        print(f"Running Florence-2 prediction (variant: {self.variant})...")

        # Convert image from BGR (OpenCV) to RGB (PIL)
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

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
            # Note: Segmentation would require more complex parsing for masks
            return self._parse_od_output(parsed_text, pil_image.size)
        else:
            # For text tasks, return the OpenAI-compatible dictionary
            # The actual content is in the parsed_text dictionary
            text_content = parsed_text.get(task_prompt, "")
            return create_openai_response(f"florence-2-{self.variant}", text_content)
