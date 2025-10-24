import numpy as np

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
    """

    # Variant configurations
    # Note: These configs contain both __init__ parameters (variant, device)
    # and internal settings (det_model, rec_model, use_angle_cls, lang)
    # The internal settings are read by __init__ from SUPPORTED_VARIANTS
    SUPPORTED_VARIANTS = {
        'mobile': {
            'variant': 'mobile',
            'device': 'cpu',
            'det_model': None,  # Use default PP-OCRv5_mobile_det
            'rec_model': None,  # Use default PP-OCRv5_mobile_rec
            'use_angle_cls': True,
            'lang': 'en',
        },
        'server': {
            'variant': 'server',
            'device': 'cpu',
            'det_model': None,  # Use default PP-OCRv5_server_det
            'rec_model': None,  # Use default PP-OCRv5_server_rec
            'use_angle_cls': True,
            'lang': 'en',
        },
        'mobile-chinese': {
            'variant': 'mobile-chinese',
            'device': 'cpu',
            'det_model': None,
            'rec_model': None,
            'use_angle_cls': True,
            'lang': 'ch',  # Simplified Chinese
        },
        'server-chinese': {
            'variant': 'server-chinese',
            'device': 'cpu',
            'det_model': None,
            'rec_model': None,
            'use_angle_cls': True,
            'lang': 'ch',
        },
        'mobile-multilingual': {
            'variant': 'mobile-multilingual',
            'device': 'cpu',
            'det_model': None,
            'rec_model': None,
            'use_angle_cls': True,
            'lang': 'en',  # Can be changed via language parameter
        },
    }

    def __init__(self, variant='mobile', language=None, device='cpu', **kwargs):
        """
        Initialize PaddleOCR predictor with specific variant.

        Args:
            variant: Model variant name ('mobile', 'server', 'mobile-chinese', 'server-chinese')
            language: Language code override (e.g., 'en', 'ch', 'fr', 'german', 'korean', 'japan')
                     If None, uses variant's default language
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
        variant_config = self.SUPPORTED_VARIANTS[variant]

        # Override language if provided
        self.language = language if language is not None else variant_config['lang']

        print(f"Loading PaddleOCR PP-OCRv5 (variant: {variant}, language: {self.language})...")

        # Build initialization parameters with only essential params
        # Note: PaddleOCR API varies across versions, so we use minimal params
        ocr_params = {
            'use_angle_cls': variant_config['use_angle_cls'],
            'lang': self.language,
        }

        # Add custom model paths if specified (for advanced users)
        if variant_config.get('det_model'):
            ocr_params['det_model_dir'] = variant_config['det_model']
        if variant_config.get('rec_model'):
            ocr_params['rec_model_dir'] = variant_config['rec_model']

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

    def predict(self, image: np.ndarray):
        """
        Run OCR on image and return PixelFlow Detections.

        Args:
            image: numpy array (H, W, 3) in BGR format (OpenCV standard)

        Returns:
            pf.detections.Detections: PixelFlow Detections object containing text detections
                                     with OCRData structure including bbox, text, confidence,
                                     angle, and direction
        """
        print(f"Running PaddleOCR prediction (variant: {self.variant})...")

        # PaddleOCR expects BGR format (OpenCV standard) - no conversion needed
        # Try running OCR with different parameter combinations for compatibility
        try:
            # Attempt 1: Try with cls parameter (angle classification)
            ocr_output = self.ocr.ocr(image, cls=True)
        except TypeError:
            try:
                # Attempt 2: Try without cls parameter
                ocr_output = self.ocr.ocr(image)
            except Exception as e:
                raise RuntimeError(f"Prediction failed: {e}")

        # The result is a list containing results for each image. Since we process
        # one image, we extract the first element.
        # It can be [None] or [[]] if no text is found.
        if not ocr_output or not ocr_output[0]:
            print("No text detected in image.")
            return pf.detections.Detections()

        results = ocr_output[0]

        # Use PixelFlow's built-in converter for consistent OCRData structure
        try:
            detections = pf.detections.from_paddleocr(results, language=self.language)
        except Exception as e:
            raise RuntimeError(f"PixelFlow conversion failed: {e}")

        print(f"Found {len(detections)} text regions.")
        return detections
