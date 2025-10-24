import numpy as np

try:
    from paddleocr import PPStructureV3
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


class PPStructurePredictor:
    """
    Universal PP-StructureV3 adapter for document structure analysis.
    Supports layout detection, table recognition, and formula recognition.
    """

    SUPPORTED_VARIANTS = {
        'layout-only': {
            'description': 'Layout detection only - Fast region identification',
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_table_recognition': False,
            'use_formula_recognition': False,
        },
        'full': {
            'description': 'Full analysis - Layout, text, tables, and formulas',
            'use_doc_orientation_classify': True,
            'use_doc_unwarping': True,
            'use_table_recognition': True,
            'use_formula_recognition': True,
        },
        'table-analysis': {
            'description': 'Focus on tables - Layout and table structure recognition',
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_table_recognition': True,
            'use_formula_recognition': False,
        },
        'formula-analysis': {
            'description': 'Focus on formulas - Layout and mathematical formulas',
            'use_doc_orientation_classify': False,
            'use_doc_unwarping': False,
            'use_table_recognition': False,
            'use_formula_recognition': True,
        },
    }

    def __init__(self, variant='full', language='en', device='cpu', **kwargs):
        """
        Initialize PP-StructureV3 predictor with specific variant.

        Args:
            variant: Model variant name ('layout-only', 'full', 'table-analysis', 'formula-analysis')
            language: Language code (e.g., 'en', 'ch', 'fr')
            device: Device to run on - 'cpu' or 'gpu'
            **kwargs: Additional parameters to override variant defaults

        Raises:
            ValueError: If variant is not supported
        """
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Choose from: {list(self.SUPPORTED_VARIANTS.keys())}"
            )

        self.variant = variant
        self.language = language
        variant_config = self.SUPPORTED_VARIANTS[variant]

        print(f"Loading PP-StructureV3 (variant: {variant}, language: {self.language})...")

        # Build initialization parameters
        pipeline_params = {
            'use_doc_orientation_classify': variant_config.get('use_doc_orientation_classify', False),
            'use_doc_unwarping': variant_config.get('use_doc_unwarping', False),
            'use_table_recognition': variant_config.get('use_table_recognition', False),
            'use_formula_recognition': variant_config.get('use_formula_recognition', False),
        }

        # Allow kwargs to override variant defaults
        pipeline_params.update(kwargs)

        # Try to initialize with progressive parameter removal for compatibility
        init_attempts = [
            # Attempt 1: All parameters
            pipeline_params.copy(),
            # Attempt 2: Remove doc orientation and unwarping (might not be supported)
            {
                'use_table_recognition': pipeline_params.get('use_table_recognition', False),
                'use_formula_recognition': pipeline_params.get('use_formula_recognition', False),
            },
            # Attempt 3: Minimal - just table recognition
            {
                'use_table_recognition': pipeline_params.get('use_table_recognition', False),
            },
            # Attempt 4: Empty dict (use all defaults)
            {},
        ]

        last_error = None
        for attempt_params in init_attempts:
            try:
                self.pipeline = PPStructureV3(**attempt_params)
                break  # Success!
            except TypeError as e:
                last_error = e
                # Continue to next attempt
                continue
        else:
            # All attempts failed
            raise RuntimeError(
                f"Failed to initialize PPStructureV3 with any parameter combination. "
                f"Last error: {last_error}. Please check your PaddleOCR installation."
            )

        print(f"PP-StructureV3 loaded successfully (variant: {variant}, language: {self.language}).")

    def predict(self, image: np.ndarray):
        """
        Run document structure analysis on image and return PixelFlow Detections.

        Args:
            image: numpy array (H, W, 3) in BGR format (OpenCV standard)

        Returns:
            pf.detections.Detections: PixelFlow Detections object with OCRData structure containing:
                - Layout regions (text, table, formula, image, etc.)
                - Bounding boxes
                - OCR text (if enabled)
                - Table HTML (if table recognition enabled)
                - Formula LaTeX (if formula recognition enabled)
                - Element type classification
                - Page index for multi-page documents
        """
        print(f"Running PP-StructureV3 prediction (variant: {self.variant})...")

        # PPStructureV3 expects BGR format (OpenCV standard) - no conversion needed
        # Unlike EasyOCR which requires RGB, PP-Structure works directly with OpenCV images
        # Run structure analysis - try different invocation methods for compatibility
        try:
            # Attempt 1: Try with predict() method
            results = self.pipeline.predict(image)
        except AttributeError:
            try:
                # Attempt 2: Try calling pipeline directly
                results = self.pipeline(image)
            except Exception as e:
                raise RuntimeError(f"Structure analysis failed: {e}")

        # Handle empty results
        if not results:
            print("No document structure detected in image.")
            return pf.detections.Detections()

        # Convert all results to PixelFlow Detections using built-in converter
        all_detections = pf.detections.Detections()

        # PPStructureV3 returns a list of result objects (one per page/image)
        for page_idx, result_obj in enumerate(results):
            # Use PixelFlow's from_ppstructure converter for consistent OCRData structure
            page_detections = pf.detections.from_ppstructure(
                result_obj,
                language=self.language,
                page_index=page_idx
            )

            # Merge page detections into all_detections
            for detection in page_detections.detections:
                all_detections.add_detection(detection)

        print(f"Found {len(all_detections)} document regions.")
        return all_detections
