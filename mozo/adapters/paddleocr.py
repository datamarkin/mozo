from typing import Union

import numpy as np

from ..utils import load_image

try:
    from paddleocr import PaddleOCR
except ImportError:
    print("="*50)
    print("ERROR: PaddleOCR is not installed.")
    print("Please install it with: pip install paddleocr")
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


class PaddleOCRPredictor:
    """
    Universal PaddleOCR adapter for PP-OCRv5 text recognition.
    Supports mobile and server variants with multi-language capabilities.

    Self-contained adapter with complete configuration.
    """

    # Complete variant configuration (single source of truth)
    SUPPORTED_VARIANTS = {
        'mobile': {'language': 'en', 'device': 'cpu', 'use_angle_cls': True},
        'server': {'language': 'en', 'device': 'cpu', 'use_angle_cls': True},
        'mobile-chinese': {'language': 'ch', 'device': 'cpu', 'use_angle_cls': True},
        'server-chinese': {'language': 'ch', 'device': 'cpu', 'use_angle_cls': True},
        'mobile-multilingual': {'language': 'en', 'device': 'cpu', 'use_angle_cls': True},
    }

    def __init__(self, variant='mobile', **kwargs):
        """
        Initialize PaddleOCR predictor with specific variant.

        Args:
            variant: Model variant name ('mobile', 'server', 'mobile-chinese', 'server-chinese', 'mobile-multilingual')
            **kwargs: Override parameters (language, device, use_angle_cls, det_model, rec_model)
                     language: Language code override (e.g., 'en', 'ch', 'fr', 'german', 'korean', 'japan')
                              If None, uses variant's default language
                     device: Device to run on - 'cpu' or 'gpu'

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

        self.variant = variant
        self.language = config.get('language', 'en')

        print(f"Loading PaddleOCR PP-OCRv5 (variant: {variant}, language: {self.language})...")

        # Build initialization parameters with only essential params
        # Note: PaddleOCR API varies across versions, so we use minimal params
        ocr_params = {
            'use_angle_cls': config.get('use_angle_cls', True),
            'lang': self.language,
        }

        # Add custom model paths if specified (for advanced users)
        if config.get('det_model'):
            ocr_params['det_model_dir'] = config['det_model']
        if config.get('rec_model'):
            ocr_params['rec_model_dir'] = config['rec_model']

        # Try to initialize with progressive parameter removal for compatibility
        init_attempts = [
            # Attempt 1: All parameters
            ocr_params.copy(),
            # Attempt 2: Remove use_angle_cls (might not be supported)
            {k: v for k, v in ocr_params.items() if k != 'use_angle_cls'},
            # Attempt 3: Minimal - just language
            {'lang': self.language},
        ]

        last_error = None
        for attempt_params in init_attempts:
            try:
                self.ocr = PaddleOCR(**attempt_params)
                break  # Success!
            except TypeError as e:
                last_error = e
                # Continue to next attempt
                continue
        else:
            # All attempts failed
            raise RuntimeError(
                f"Failed to initialize PaddleOCR with any parameter combination. "
                f"Last error: {last_error}. Please check your PaddleOCR installation."
            )

        print(f"PaddleOCR PP-OCRv5 loaded successfully (variant: {variant}, language: {self.language}).")

    def predict(self, image: Union[str, np.ndarray]):
        """
        Run OCR on image and return PixelFlow Detections.

        Args:
            image: File path (str) or numpy array (BGR format)

        Returns:
            pf.detections.Detections: PixelFlow Detections object containing text detections
                                     with OCRData structure including bbox, text, confidence,
                                     angle, and direction
        """
        image = load_image(image)
        print(f"Running PaddleOCR prediction (variant: {self.variant})...")

        # PaddleOCR expects BGR format (OpenCV standard) - no conversion needed
        # Use PaddleOCR 3.x predict() API
        ocr_output = self.ocr.predict(image)

        # The result is a list of OCRResult objects. Since we process one image,
        # we get one Result object.
        if not ocr_output or len(ocr_output) == 0:
            print("No text detected in image.")
            return pf.detections.Detections()

        # Use PixelFlow's from_paddleocr3() converter for PaddleOCR 3.x API
        try:
            detections = pf.detections.from_paddleocr3(ocr_output, language=self.language)
        except Exception as e:
            raise RuntimeError(f"PixelFlow conversion failed: {e}")

        print(f"Found {len(detections)} text regions.")
        return detections
