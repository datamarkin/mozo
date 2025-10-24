from PIL import Image
import numpy as np
import cv2
import re

try:
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
except ImportError:
    print("="*50)
    print("ERROR: Qwen3-VL dependencies not installed.")
    print("Install with:")
    print("  pip install transformers>=4.50")
    print("Note: Qwen3-VL requires transformers 4.50 or later")
    print("="*50)
    raise

class Qwen3VLPredictor:
    """
    Universal Qwen3-VL adapter for vision-language understanding with reasoning.

    This adapter supports models that output chain-of-thought reasoning via
    <thinking>...</thinking> tags, providing transparent, explainable AI.

    Supports:
    - Image understanding and description
    - Visual question answering (VQA) with reasoning steps
    - Object recognition and counting with explanations
    - Text extraction (OCR) with context
    - Chart and diagram analysis with step-by-step breakdown
    - Visual reasoning with transparent thought process

    Self-contained adapter with complete configuration.
    """

    # Complete variant configuration (single source of truth)
    SUPPORTED_VARIANTS = {
        '2b-thinking': {'device': 'cpu', 'torch_dtype': 'auto'},
    }

    # Mapping of variant names to HuggingFace model IDs (implementation detail)
    _MODEL_IDS = {
        '2b-thinking': 'Qwen/Qwen3-VL-2B-Thinking',
    }

    def __init__(self, variant="2b-thinking", **kwargs):
        """
        Initialize Qwen3-VL predictor with specific model variant.

        Args:
            variant: Model size variant - '2b-thinking'
                    2b-thinking: 2 billion parameters, compact with reasoning
            **kwargs: Override parameters (device, torch_dtype)
                     device: Device placement - 'auto', 'cpu', 'cuda', 'mps'
                            'auto' will automatically use best available device
                            Note: 'cpu' recommended for stability with 2B model
                     torch_dtype: Precision - 'auto', 'float16', 'bfloat16', 'float32'
                                 'auto' will choose based on device capabilities

        Raises:
            ValueError: If variant is not supported

        Note:
            This is a compact 2B model (~4-8GB RAM). First load will download ~4-8GB.
            Recommended: 8GB+ RAM. CPU is sufficient for reasonable performance.
            The "thinking" variant outputs reasoning steps before final answers.
        """
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Supported variants: {list(self.SUPPORTED_VARIANTS.keys())}"
            )

        # Merge defaults with overrides
        config = {**self.SUPPORTED_VARIANTS[variant], **kwargs}
        device = config.get('device', 'cpu')
        torch_dtype = config.get('torch_dtype', 'auto')

        self.variant = variant
        model_name = self._MODEL_IDS[variant]

        print(f"Loading Qwen3-VL model (variant: {variant}, model: {model_name})...")
        print("Note: This is a 2B thinking model. First load may download ~4-8GB...")

        # Load model and processor
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device
        )
        self.processor = AutoProcessor.from_pretrained(model_name)

        print(f"Qwen3-VL model loaded successfully (variant: {variant}).")
        print(f"Model device: {self.model.device}")
        print("Note: This model outputs reasoning steps via <thinking> tags.")

    def _parse_thinking_tags(self, text: str) -> tuple:
        """
        Parse thinking tags from model output.

        Extracts <thinking>...</thinking> blocks and returns both the thinking
        content and the text with thinking blocks removed.

        Args:
            text: Raw model output text

        Returns:
            tuple: (thinking_content or None, text_without_thinking)

        Examples:
            Input: "<thinking>Step 1\nStep 2</thinking>Answer here"
            Output: ("Step 1\nStep 2", "Answer here")

            Input: "Direct answer without thinking"
            Output: (None, "Direct answer without thinking")
        """
        # Pattern to match <thinking>...</thinking> blocks (including multiline)
        thinking_pattern = r'<thinking>(.*?)</thinking>'

        # Find all thinking blocks
        thinking_matches = re.findall(thinking_pattern, text, re.DOTALL)

        # Extract thinking content if found
        if thinking_matches:
            # Join multiple thinking blocks with newlines
            thinking_content = '\n\n'.join(match.strip() for match in thinking_matches)

            # Remove thinking blocks from text
            text_without_thinking = re.sub(thinking_pattern, '', text, flags=re.DOTALL)
            text_without_thinking = text_without_thinking.strip()

            return thinking_content, text_without_thinking
        else:
            return None, text

    def predict(self, image: np.ndarray, prompt: str = "Describe this image in detail.") -> dict:
        """
        Run vision-language understanding with reasoning on an image.

        Args:
            image: Input image as numpy array (BGR format from cv2)
            prompt: Text prompt/question about the image
                   Examples:
                   - "Describe this image in detail."
                   - "What objects are in this image?"
                   - "How many people are visible?"
                   - "What is the text in this image?"
                   - "Analyze this chart and explain your reasoning."

        Returns:
            dict: {
                'text': str,         # Final answer without thinking tags
                'thinking': str or None,  # Extracted reasoning steps (if present)
                'raw_output': str,   # Full model output with thinking tags
                'prompt': str,       # Original prompt used
                'variant': str       # Model variant used
            }

        Example:
            >>> predictor = Qwen3VLPredictor()
            >>> result = predictor.predict(image, "What is in this image?")
            >>> print(result['text'])  # Final answer
            >>> if result['thinking']:
            ...     print(result['thinking'])  # Reasoning steps
        """
        print(f"Running Qwen3-VL inference with prompt: '{prompt[:50]}...'")

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

        # Prepare inputs using Qwen3-VL simplified API
        # Note: Qwen3-VL uses a simpler API than Qwen2.5-VL
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )

        # Move inputs to model device
        inputs = inputs.to(self.model.device)

        # Generate response
        # Note: Thinking models may need more tokens for reasoning steps
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=1024  # Increased to accommodate thinking + answer
        )

        # Trim input tokens from generated output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode generated text (with thinking tags)
        raw_output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        # Parse thinking tags
        thinking_content, final_text = self._parse_thinking_tags(raw_output)

        if thinking_content:
            print(f"Qwen3-VL inference complete. Generated {len(raw_output)} characters (includes reasoning).")
            print(f"  - Thinking: {len(thinking_content)} characters")
            print(f"  - Answer: {len(final_text)} characters")
        else:
            print(f"Qwen3-VL inference complete. Generated {len(raw_output)} characters (no thinking tags found).")

        return {
            'text': final_text,
            'thinking': thinking_content,
            'raw_output': raw_output,
            'prompt': prompt,
            'variant': self.variant
        }
