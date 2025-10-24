import numpy as np
import cv2

try:
    import easyocr
except ImportError:
    print("="*50)
    print("ERROR: EasyOCR is not installed.")
    print("Please install it with: pip install easyocr")
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


class EasyOCRPredictor:
    """
    Universal EasyOCR adapter for text recognition.
    Supports 80+ languages with easy setup and good general-purpose accuracy.

    Self-contained adapter with complete configuration.
    """

    # Complete variant configuration (single source of truth)
    SUPPORTED_VARIANTS = {
        'english-light': {
            'languages': ['en'],
            'recog_network': None,
            'device': 'cpu',
        },
        'english-full': {
            'languages': ['en'],
            'recog_network': 'english_g2',  # Higher accuracy model
            'device': 'cpu',
        },
        'multilingual': {
            'languages': ['en', 'ch_sim', 'fr', 'de', 'es'],
            'recog_network': None,
            'device': 'cpu',
        },
        'chinese': {
            'languages': ['ch_sim', 'en'],
            'recog_network': None,
            'device': 'cpu',
        },
        'custom': {
            'languages': None,  # User must provide
            'recog_network': None,
            'device': 'cpu',
        },
    }

    def __init__(self, variant='english-light', **kwargs):
        """
        Initialize EasyOCR predictor with specific variant.

        Args:
            variant: Model variant name ('english-light', 'english-full', 'multilingual', 'chinese', 'custom')
            **kwargs: Override parameters (languages, device, recog_network)
                     languages: Language code list override (e.g., ['en'], ['ch_sim', 'en'])
                               Required for 'custom' variant, optional for others
                     device: Device to run on - 'cpu' or 'gpu'
                     Additional parameters for easyocr.Reader (download_enabled, model_storage_directory, etc.)

        Raises:
            ValueError: If variant is not supported or custom variant without languages
        """
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Supported variants: {list(self.SUPPORTED_VARIANTS.keys())}"
            )

        # Merge defaults with overrides
        config = {**self.SUPPORTED_VARIANTS[variant], **kwargs}

        self.variant = variant

        # Determine language list
        if variant == 'custom':
            if 'languages' not in config or config['languages'] is None:
                raise ValueError(
                    "Custom variant requires 'languages' parameter. "
                    "Example: languages=['en', 'fr', 'de']"
                )
            self.languages = config['languages']
        else:
            # Use variant's default languages, but allow override from kwargs
            self.languages = config.get('languages')

        # Determine device (EasyOCR uses gpu=True/False)
        device = config.get('device', 'cpu')
        self.use_gpu = (device == 'gpu')

        print(f"Loading EasyOCR (variant: {variant}, languages: {self.languages}, gpu: {self.use_gpu})...")

        # Build Reader initialization parameters
        reader_params = {
            'lang_list': self.languages,
            'gpu': self.use_gpu,
        }

        # Add recog_network if specified
        if config.get('recog_network'):
            reader_params['recog_network'] = config['recog_network']

        # Add any additional parameters from kwargs (excluding our known params)
        # Common params: download_enabled, model_storage_directory, quantize, cudnn_benchmark
        extra_params = {k: v for k, v in kwargs.items() if k not in ['languages', 'device', 'recog_network']}
        reader_params.update(extra_params)

        # Try to initialize with progressive parameter removal for compatibility
        init_attempts = [
            # Attempt 1: All parameters
            reader_params.copy(),
            # Attempt 2: Remove recog_network (might not be supported in older versions)
            {k: v for k, v in reader_params.items() if k != 'recog_network'},
            # Attempt 3: Minimal - just language and gpu
            {'lang_list': self.languages, 'gpu': self.use_gpu},
        ]

        last_error = None
        for attempt_idx, attempt_params in enumerate(init_attempts):
            try:
                self.reader = easyocr.Reader(**attempt_params)
                break  # Success!
            except (TypeError, ValueError) as e:
                last_error = e
                # Continue to next attempt
                continue
        else:
            # All attempts failed
            raise RuntimeError(
                f"Failed to initialize EasyOCR Reader with any parameter combination. "
                f"Last error: {last_error}. Please check your EasyOCR installation."
            )

        print(f"EasyOCR loaded successfully (variant: {variant}, languages: {self.languages}).")

    def predict(self, image: np.ndarray):
        """
        Run OCR on image and return PixelFlow Detections.

        Args:
            image: numpy array (H, W, 3) in BGR format (OpenCV standard)

        Returns:
            pf.detections.Detections: PixelFlow Detections object containing text detections
                                     with OCRData structure including bbox, text, confidence,
                                     and polygon segments
        """
        print(f"Running EasyOCR prediction (variant: {self.variant})...")

        # EasyOCR expects RGB format, so convert from BGR
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Run OCR with progressive parameter fallback for compatibility
        try:
            # Attempt 1: Try with common parameters
            results = self.reader.readtext(
                image_rgb,
                detail=1,  # Return detailed results with confidence
                paragraph=False,  # Don't combine into paragraphs
            )
        except TypeError:
            try:
                # Attempt 2: Try with minimal parameters
                results = self.reader.readtext(image_rgb)
            except Exception as e:
                raise RuntimeError(f"Prediction failed: {e}")

        # Handle empty results
        if not results:
            print("No text detected in image.")
            return pf.detections.Detections()

        # Determine primary language for metadata
        # EasyOCR supports multiple languages (list), but PixelFlow expects a single string
        if not self.languages:
            language = 'en'
        elif len(self.languages) == 1:
            language = self.languages[0]
        else:
            language = 'multi'  # Multi-language mode

        # Use PixelFlow's built-in converter for consistent OCRData structure
        detections = pf.detections.from_easyocr(results, language=language)

        print(f"Found {len(detections)} text regions.")
        return detections
