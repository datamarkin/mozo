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

    A unified vision-language model supporting multiple computer vision tasks:

    **Detection Tasks:**
    - Object detection with bounding boxes
    - Dense region captioning (detection with descriptive labels)

    **Segmentation Tasks:**
    - Text-grounded instance segmentation (requires prompt)

    **Text-Based Tasks:**
    - Image captioning (basic, detailed, more detailed)
    - OCR with and without region detection

    Returns PixelFlow Detections for detection/segmentation tasks and
    OpenAI-compatible dict for text-based tasks.
    """

    # Variant configurations (valid __init__ parameters only)
    # Task prompts are stored separately in _TASK_PROMPTS
    SUPPORTED_VARIANTS = {
        # Detection variants
        'detection': {'variant': 'detection', 'device': 'cpu'},
        'detection_with_caption': {'variant': 'detection_with_caption', 'device': 'cpu'},
        # Segmentation variant (requires prompt)
        'segmentation': {'variant': 'segmentation', 'device': 'cpu'},
        # Text-based variants
        'captioning': {'variant': 'captioning', 'device': 'cpu'},
        'detailed_captioning': {'variant': 'detailed_captioning', 'device': 'cpu'},
        'more_detailed_captioning': {'variant': 'more_detailed_captioning', 'device': 'cpu'},
        'ocr': {'variant': 'ocr', 'device': 'cpu'},
        'ocr_with_region': {'variant': 'ocr_with_region', 'device': 'cpu'},
    }

    # Task prompts mapped to variants (used internally by predict())
    _TASK_PROMPTS = {
        # Detection prompts
        'detection': '<OD>',
        'detection_with_caption': '<DENSE_REGION_CAPTION>',
        # Segmentation prompt
        'segmentation': '<REFERRING_EXPRESSION_SEGMENTATION>',
        # Text-based prompts
        'captioning': '<CAPTION>',
        'detailed_captioning': '<DETAILED_CAPTION>',
        'more_detailed_captioning': '<MORE_DETAILED_CAPTION>',
        'ocr': '<OCR>',
        'ocr_with_region': '<OCR_WITH_REGION>',
    }

    def __init__(self, variant='captioning', device='cpu', **kwargs):
        """
        Initialize the Florence-2 model from Hugging Face.

        Args:
            variant: Task variant to perform. Available variants:
                    - Detection: 'detection', 'detection_with_caption'
                    - Segmentation: 'segmentation' (requires prompt in predict())
                    - Captioning: 'captioning', 'detailed_captioning', 'more_detailed_captioning'
                    - OCR: 'ocr', 'ocr_with_region'
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

        # Use eager attention to avoid _supports_sdpa compatibility issues
        # Fall back to default loading if parameter not supported
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                attn_implementation="eager"
            )
        except TypeError:
            # Older transformers version may not support attn_implementation
            print("Note: Loading without attn_implementation (transformers version may be older)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True
            )

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        self.device = 'cuda' if device == 'gpu' else 'cpu'
        self.model.to(self.device)

        print(f"Florence-2 model loaded successfully on device '{self.device}'.")

    def predict(self, image: np.ndarray, prompt: str = None):
        """
        Run inference with the Florence-2 model.

        Args:
            image: Input image as numpy array (H, W, 3) in BGR format (OpenCV standard)
            prompt: Prompt behavior varies by variant:
                   - Detection variants: Ignored (always uses task-specific prompts like '<OD>')
                   - Segmentation: Required (e.g., "person", "car") - describes what to segment
                   - Text variants: Optional override of default task prompt

        Returns:
            For detection/segmentation variants:
                pf.detections.Detections: Detections object with bboxes, masks, and labels
            For text-based variants (captioning, OCR):
                dict: OpenAI-compatible response containing generated text

        Raises:
            ValueError: If segmentation variant is used without providing a prompt
        """
        # Debug: Check image parameter immediately
        print(f"DEBUG: predict() called with image type: {type(image)}")
        print(f"DEBUG: image is None: {image is None}")
        if image is not None and hasattr(image, 'shape'):
            print(f"DEBUG: image shape: {image.shape}")

        task_prompt = self._TASK_PROMPTS[self.variant]

        # Handle prompt construction based on variant type
        if self.variant == 'segmentation':
            # Segmentation requires a text prompt (e.g., "person", "car")
            if not prompt:
                raise ValueError(
                    "Segmentation variant requires a 'prompt' parameter. "
                    "Example: prompt='person' or prompt='car'. "
                    "This tells Florence-2 what object to segment."
                )
            # For segmentation, concatenate task prompt with user prompt
            final_prompt = f"{task_prompt}{prompt}"

        elif self.variant in ['detection', 'detection_with_caption']:
            # Detection tasks MUST use task-specific prompts, ignore user prompts
            final_prompt = task_prompt

        else:
            # Text-based tasks (captioning, OCR): use user prompt if provided, otherwise task prompt
            final_prompt = prompt if prompt else task_prompt

        print(f"Running Florence-2 prediction (variant: {self.variant})...")
        print(f"  Task prompt: {task_prompt}")
        print(f"  Final prompt: {final_prompt}")

        # Validate image input
        if image is None:
            raise ValueError("Image parameter is None. Please provide a valid image.")
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Image must be a numpy array, got {type(image)}")

        # Florence-2 processor expects RGB format
        # Convert from BGR (OpenCV standard) to RGB for PIL
        print(f"DEBUG: About to call cv2.cvtColor with image shape: {image.shape}")
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            print(f"DEBUG: cv2.cvtColor successful, image_rgb shape: {image_rgb.shape}")
        except Exception as e:
            print(f"DEBUG: cv2.cvtColor failed with error: {e}")
            raise

        pil_image = Image.fromarray(image_rgb)
        print(f"DEBUG: PIL Image created, size: {pil_image.size}")

        print(f"DEBUG: About to call processor with text='{final_prompt}', image size={pil_image.size}")
        try:
            inputs = self.processor(text=final_prompt, images=pil_image, return_tensors="pt")
            print(f"DEBUG: Processor successful, inputs keys: {inputs.keys()}")
            print(f"DEBUG: input_ids shape: {inputs['input_ids'].shape}")
            print(f"DEBUG: pixel_values shape: {inputs['pixel_values'].shape}")
            inputs = inputs.to(self.device)
            print(f"DEBUG: Inputs moved to device: {self.device}")
        except Exception as e:
            print(f"DEBUG: Processor failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"DEBUG: About to call model.generate()")
        try:
            # Ensure model is in eval mode to avoid past_key_values issues
            self.model.eval()

            # Use minimal generation parameters - Florence-2 has issues with generation configs
            generated_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                early_stopping=False,
                do_sample=False,
                num_beams=3,
                use_cache=False  # Disable KV cache to avoid past_key_values bug
            )
            print(f"DEBUG: Generation successful, output shape: {generated_ids.shape}")
        except Exception as e:
            print(f"DEBUG: Generation failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

        print(f"DEBUG: About to decode generated_ids")
        try:
            generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
            print(f"DEBUG: Generated text: {generated_text[:100]}...")
        except Exception as e:
            print(f"DEBUG: Decoding failed with error: {e}")
            import traceback
            traceback.print_exc()
            raise

        # Post-process to get the clean result
        # The processor converts raw model output to structured format
        print(f"DEBUG: Calling post_process_generation with task={task_prompt}, image_size={pil_image.size}")
        parsed_result = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=pil_image.size
        )
        print(f"DEBUG: post_process_generation result keys: {parsed_result.keys() if isinstance(parsed_result, dict) else type(parsed_result)}")
        print(f"DEBUG: parsed_result content: {parsed_result}")

        # Handle different output types based on variant
        if self.variant in ['detection', 'detection_with_caption', 'segmentation']:
            # Use PixelFlow's built-in converter for detection/segmentation tasks
            print(f"DEBUG: Converting to PixelFlow Detections using from_florence2")
            detections = pf.detections.from_florence2(
                parsed_result,
                task_prompt=task_prompt,
                image_size=pil_image.size
            )
            print(f"Detected/Segmented {len(detections)} object(s).")
            return detections

        else:
            # Text-based tasks: return OpenAI-compatible response
            text_content = parsed_result.get(task_prompt, "")
            return create_openai_response(f"florence-2-{self.variant}", text_content)
