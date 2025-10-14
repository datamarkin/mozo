from PIL import Image
import numpy as np
import cv2

try:
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
except ImportError:
    print("="*50)
    print("ERROR: Qwen2.5-VL dependencies not installed.")
    print("Install with:")
    print("  pip install git+https://github.com/huggingface/transformers")
    print("  pip install qwen-vl-utils")
    print("="*50)
    raise

class Qwen2_5VLPredictor:
    """
    Universal Qwen2.5-VL adapter for vision-language understanding.

    Supports:
    - Image understanding and description
    - Visual question answering (VQA)
    - Object recognition and counting
    - Text extraction (OCR)
    - Chart and diagram analysis
    - Visual reasoning
    """

    # Registry of all supported Qwen2.5-VL model variants
    SUPPORTED_VARIANTS = {
        '7b-instruct': 'Qwen/Qwen2.5-VL-7B-Instruct',
        # Future variants can be added here:
        # '2b-instruct': 'Qwen/Qwen2.5-VL-2B-Instruct',
        # '72b-instruct': 'Qwen/Qwen2.5-VL-72B-Instruct',
    }

    def __init__(self, variant="7b-instruct", device="auto", torch_dtype="auto"):
        """
        Initialize Qwen2.5-VL predictor with specific model variant.

        Args:
            variant: Model size variant - '7b-instruct'
                    7b-instruct: 7 billion parameters, balanced performance
            device: Device placement - 'auto', 'cpu', 'cuda', 'mps'
                   'auto' will automatically use best available device
            torch_dtype: Precision - 'auto', 'float16', 'bfloat16', 'float32'
                        'auto' will choose based on device capabilities

        Raises:
            ValueError: If variant is not supported

        Note:
            This is a large model (7B parameters). First load will download ~7-14GB.
            Recommended: 16GB+ RAM and GPU/MPS for reasonable performance.
        """
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Choose from: {list(self.SUPPORTED_VARIANTS.keys())}"
            )

        self.variant = variant
        model_name = self.SUPPORTED_VARIANTS[variant]

        print(f"Loading Qwen2.5-VL model (variant: {variant}, model: {model_name})...")
        print("Note: This is a 7B model and may take time to load on first run...")

        # Load model and processor
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        print(f"Qwen2.5-VL model loaded successfully (variant: {variant}).")
        print(f"Model device: {self.model.device}")

    def predict(self, image: np.ndarray, prompt: str = "Describe this image in detail.") -> dict:
        """
        Run vision-language understanding on an image.

        Args:
            image: Input image as numpy array (BGR format from cv2)
            prompt: Text prompt/question about the image
                   Examples:
                   - "Describe this image in detail."
                   - "What objects are in this image?"
                   - "How many people are visible?"
                   - "What is the text in this image?"
                   - "Analyze this chart."

        Returns:
            dict: {
                'text': str,      # Generated text response
                'prompt': str,    # Original prompt used
                'variant': str    # Model variant used
            }

        Example:
            >>> predictor = Qwen2_5VLPredictor()
            >>> result = predictor.predict(image, "What is in this image?")
            >>> print(result['text'])
        """
        print(f"Running Qwen2.5-VL inference with prompt: '{prompt[:50]}...'")

        # Convert BGR (OpenCV format) to RGB (PIL format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Prepare conversation in Qwen format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Process vision information
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for model
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to model device
        inputs = inputs.to(self.model.device)

        # Generate response
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=512  # Adjust based on expected response length
        )

        # Trim input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode generated text
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        print(f"Qwen2.5-VL inference complete. Generated {len(output_text)} characters.")

        return {
            'text': output_text,
            'prompt': prompt,
            'variant': self.variant
        }
