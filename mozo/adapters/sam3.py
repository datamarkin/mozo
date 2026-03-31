"""
SAM3 (Segment Anything Model 3) adapter for text-prompted instance segmentation.

Meta's SAM3 introduces Promptable Concept Segmentation (PCS) - the ability to segment
all instances of objects matching a text description (e.g., "yellow school bus",
"person wearing a red hat"). Supports 270,000+ unique concepts.

Requirements:
    - Python >= 3.10
    - transformers (from main branch): pip install git+https://github.com/huggingface/transformers
    - torch >= 2.0.0
    - HuggingFace authentication (huggingface-cli login)
    - Accept model license at https://huggingface.co/facebook/sam3
"""

from typing import Union

import cv2
import numpy as np

from ..utils import load_image

try:
    import torch
except ImportError:
    print("=" * 50)
    print("ERROR: SAM3 requires `torch`.")
    print("Please install with: pip install torch")
    print("=" * 50)
    raise

try:
    from transformers import Sam3Model, Sam3Processor
except ImportError:
    print("=" * 50)
    print("ERROR: SAM3 requires HuggingFace transformers with SAM3 support.")
    print("SAM3 is only available in the development version of transformers.")
    print("Please install with: pip install git+https://github.com/huggingface/transformers")
    print("Note: Requires Python >= 3.10")
    print("=" * 50)
    raise

try:
    import pixelflow as pf
except ImportError:
    print("=" * 50)
    print("ERROR: PixelFlow is not installed.")
    print("Please install it with: pip install pixelflow")
    print("=" * 50)
    raise


class Sam3Predictor:
    """
    SAM3 adapter for text-prompted instance segmentation.

    Uses HuggingFace transformers implementation for cross-platform support
    (Linux, macOS with MPS, Windows).

    Enables open-vocabulary segmentation using natural language text prompts.
    Unlike SAM 1/2 which required visual prompts (points, boxes), SAM3 can
    segment all instances matching a text description in a single inference.

    Features:
        - Text prompts: "red car", "person wearing hat", "yellow school bus"
        - Multi-class prompts: "person, car, dog" (comma-separated)
        - Multi-instance detection: finds ALL matching objects, not just one
        - 270,000+ supported concepts
        - ~30ms inference on H200 GPU

    Output:
        Returns PixelFlow Detections with masks, bounding boxes, and confidence scores.
    """

    SUPPORTED_VARIANTS = {
        'default': {'device': 'cuda', 'dtype': 'float16'},
        'cpu': {'device': 'cpu', 'dtype': 'float32'},
        'mps': {'device': 'mps', 'dtype': 'float32'},
    }

    def __init__(self, variant: str = "default", **kwargs):
        """
        Initialize SAM3 predictor using HuggingFace transformers.

        Args:
            variant: Model variant
                - 'default': GPU with float16 (recommended, ~3.4GB VRAM)
                - 'cpu': CPU with float32 (slower, no GPU required)
                - 'mps': Apple Silicon with float32
            **kwargs: Override default parameters (device, dtype)

        Raises:
            ValueError: If variant is not supported

        Note:
            First run downloads ~3.4GB model from HuggingFace.
            Requires HuggingFace authentication: `huggingface-cli login`
        """
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Supported variants: {list(self.SUPPORTED_VARIANTS.keys())}"
            )

        # Merge defaults with overrides
        config = {**self.SUPPORTED_VARIANTS[variant], **kwargs}
        device = config.get('device', 'cuda')
        dtype_str = config.get('dtype', 'float16')

        # Handle device availability
        if device == 'cuda':
            if not torch.cuda.is_available():
                # Check for MPS (Apple Silicon)
                if torch.backends.mps.is_available():
                    print("CUDA not available, using MPS (Apple Silicon)")
                    device = 'mps'
                    dtype_str = 'float32'
                else:
                    print("CUDA not available, falling back to CPU")
                    device = 'cpu'
                    dtype_str = 'float32'
        elif device == 'mps':
            if not torch.backends.mps.is_available():
                print("MPS not available, falling back to CPU")
                device = 'cpu'
                dtype_str = 'float32'

        self.device = device
        self.dtype = torch.float16 if dtype_str == 'float16' else torch.float32
        self.variant = variant

        print(f"Loading SAM3 model (variant: {variant}, device: {device})...")

        # Load model from HuggingFace
        self.model = Sam3Model.from_pretrained("facebook/sam3")
        self.model = self.model.to(device)
        if device not in ['cpu', 'mps']:
            self.model = self.model.to(dtype=self.dtype)
        self.model.eval()

        self.processor = Sam3Processor.from_pretrained("facebook/sam3")

        print(f"SAM3 model loaded successfully (variant: {variant}, device: {device}).")

    def predict(
        self,
        image: Union[str, np.ndarray],
        prompt: str = "object",
        threshold: float = 0.5,
        mask_threshold: float = 0.5
    ) -> "pf.detections.Detections":
        """
        Segment all objects matching the text prompt(s).

        Args:
            image: File path (str) or numpy array in BGR format (OpenCV standard)
            prompt: Text description of objects to segment. Supports comma-separated
                   multiple classes.
                   Examples:
                   - Single: "person", "red car", "yellow school bus"
                   - Multiple: "person, car, dog" (each processed separately)
            threshold: Detection confidence threshold [0.0-1.0]. Default: 0.5
            mask_threshold: Mask binarization threshold [0.0-1.0]. Default: 0.5

        Returns:
            pf.detections.Detections: PixelFlow Detections containing:
                - bbox: [x1, y1, x2, y2] bounding box
                - confidence: detection score
                - class_name: the text prompt used
                - class_id: index of the prompt (for multi-class)
                - masks: binary segmentation mask

        Example:
            >>> from mozo import ModelManager
            >>> manager = ModelManager()
            >>> model = manager.get_model('sam3', 'default')
            >>>
            >>> # Segment single class
            >>> detections = model.predict(image, prompt="person")
            >>> print(f"Found {len(detections)} people")
            >>>
            >>> # Segment multiple classes
            >>> detections = model.predict(image, prompt="person, car, dog")
            >>> for det in detections:
            ...     print(f"{det.class_name}: {det.confidence:.2f}")
        """
        from PIL import Image as PILImage

        # Load and convert image
        image_array = load_image(image)
        h, w = image_array.shape[:2]

        # Convert BGR (OpenCV) to RGB (PIL)
        image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        pil_image = PILImage.fromarray(image_rgb)

        # Split comma-separated prompts
        prompts = [p.strip() for p in prompt.split(",") if p.strip()]

        print(f"Running SAM3 segmentation with {len(prompts)} prompt(s): {prompts}")

        # Collect all detections from all prompts
        all_detections = pf.detections.Detections()

        for class_id, single_prompt in enumerate(prompts):
            print(f"  Processing: '{single_prompt}' (class_id={class_id})...")

            # Process inputs using HuggingFace processor
            inputs = self.processor(
                images=pil_image,
                text=single_prompt,
                return_tensors="pt"
            )

            # Move inputs to device
            inputs = {k: v.to(self.device) if hasattr(v, 'to') else v for k, v in inputs.items()}

            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)

            # Post-process results
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=[(h, w)]
            )[0]

            # Convert to PixelFlow Detections using built-in converter
            detections = pf.detections.from_sam3(
                results,
                prompt=single_prompt,
                class_id=class_id
            )

            # Add to combined results
            for det in detections:
                all_detections.add_detection(det)

            print(f"    Found {len(detections)} objects for '{single_prompt}'")

        print(f"Total: Found {len(all_detections)} objects across all prompts.")
        return all_detections
