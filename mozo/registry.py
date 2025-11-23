"""
Model Registry for Mozo

Lightweight registry for model discovery and routing.
Variant names are listed here for fast discovery without importing adapters.
Full variant configuration lives in adapters (single source of truth).

NOTE: Registry can be out of sync with adapters - this is acceptable.
If a variant exists in adapter but not registry, it will still work.
Registry is primarily for fast /models API endpoint.

Usage:
    # To add a new model family, add an entry to MODEL_REGISTRY
    # To add a new variant, add it to the 'variants' list AND the adapter's SUPPORTED_VARIANTS

Example:
    'detectron2': {
        'adapter_class': 'Detectron2Predictor',
        'module': 'mozo.adapters.detectron2',
        'task_type': 'object_detection',
        'description': 'Detectron2 models...',
        'variants': [
            'mask_rcnn_R_50_FPN_3x',
            'faster_rcnn_R_50_FPN_3x',
            # ... just variant names
        ]
    }
"""

# Main model registry - maps family names to adapter configurations
MODEL_REGISTRY = {
    'detectron2': {
        'adapter_class': 'Detectron2Predictor',
        'module': 'mozo.adapters.detectron2',
        'task_type': 'object_detection',
        'description': 'Detectron2 models for object detection, instance segmentation, and keypoint detection',
        'variants': [
            # Mask R-CNN (Instance Segmentation)
            'mask_rcnn_R_50_FPN_3x', 'mask_rcnn_R_50_C4_1x', 'mask_rcnn_R_50_C4_3x',
            'mask_rcnn_R_50_DC5_1x', 'mask_rcnn_R_50_DC5_3x', 'mask_rcnn_R_50_FPN_1x',
            'mask_rcnn_R_101_C4_3x', 'mask_rcnn_R_101_DC5_3x', 'mask_rcnn_R_101_FPN_3x',
            'mask_rcnn_X_101_32x8d_FPN_3x',
            # Faster R-CNN (Object Detection)
            'faster_rcnn_R_50_C4_1x', 'faster_rcnn_R_50_C4_3x', 'faster_rcnn_R_50_DC5_1x',
            'faster_rcnn_R_50_DC5_3x', 'faster_rcnn_R_50_FPN_1x', 'faster_rcnn_R_50_FPN_3x',
            'faster_rcnn_R_101_C4_3x', 'faster_rcnn_R_101_DC5_3x', 'faster_rcnn_R_101_FPN_3x',
            'faster_rcnn_X_101_32x8d_FPN_3x',
            # RetinaNet (Object Detection)
            'retinanet_R_50_FPN_1x', 'retinanet_R_50_FPN_3x', 'retinanet_R_101_FPN_3x',
            # Keypoint R-CNN (Keypoint Detection)
            'keypoint_rcnn_R_50_FPN_1x', 'keypoint_rcnn_R_50_FPN_3x', 'keypoint_rcnn_R_101_FPN_3x',
            'keypoint_rcnn_X_101_32x8d_FPN_3x',
        ],
    },

    'depth_anything': {
        'adapter_class': 'DepthAnythingPredictor',
        'module': 'mozo.adapters.depth_anything',
        'task_type': 'depth_estimation',
        'description': 'Depth Anything V2 models for monocular depth estimation',
        'variants': ['small', 'base', 'large'],
    },

    'qwen2.5_vl': {
        'adapter_class': 'Qwen2_5VLPredictor',
        'module': 'mozo.adapters.qwen2_5_vl',
        'task_type': 'visual_question_answering',
        'description': 'Qwen2.5-VL models for vision-language understanding, VQA, and image analysis',
        'variants': ['7b-instruct'],
    },

    'qwen3_vl': {
        'adapter_class': 'Qwen3VLPredictor',
        'module': 'mozo.adapters.qwen3_vl',
        'task_type': 'visual_question_answering_with_reasoning',
        'description': 'Qwen3-VL models with chain-of-thought reasoning for explainable vision-language understanding',
        'variants': ['2b-thinking'],
    },

    'paddleocr': {
        'adapter_class': 'PaddleOCRPredictor',
        'module': 'mozo.adapters.paddleocr',
        'task_type': 'ocr',
        'description': 'PaddleOCR PP-OCRv5 - Universal scene text recognition supporting 80+ languages with mobile and server variants',
        'variants': ['mobile', 'server', 'mobile-chinese', 'server-chinese', 'mobile-multilingual'],
    },

    'ppstructure': {
        'adapter_class': 'PPStructurePredictor',
        'module': 'mozo.adapters.ppstructure',
        'task_type': 'document_analysis',
        'description': 'PP-StructureV3 - Document structure analysis with layout detection, table recognition, and formula extraction',
        'variants': ['layout-only', 'full', 'table-analysis', 'formula-analysis'],
    },

    'easyocr': {
        'adapter_class': 'EasyOCRPredictor',
        'module': 'mozo.adapters.easyocr',
        'task_type': 'ocr',
        'description': 'EasyOCR - User-friendly OCR with 80+ languages, easy setup, and good general-purpose accuracy',
        'variants': ['english-light', 'english-full', 'multilingual', 'chinese'],
    },

    'stability_inpainting': {
        'adapter_class': 'StabilityInpaintingPredictor',
        'module': 'mozo.adapters.stability_inpainting',
        'task_type': 'image_generation',
        'description': 'Stability AI Stable Diffusion 2 Inpainting - Generate and modify image content using text prompts (REQUIRES: mask image parameter)',
        'variants': ['default'],
    },

    'florence2': {
        'adapter_class': 'Florence2Predictor',
        'module': 'mozo.adapters.florence2',
        'task_type': 'multi_task_vision',
        'description': 'Microsoft Florence-2 for vision tasks including captioning and OCR (detection/segmentation not yet implemented)',
        'variants': [
            'detection', 'detection_with_caption', 'segmentation',
            'captioning', 'detailed_captioning', 'more_detailed_captioning',
            'ocr', 'ocr_with_region',
        ],
    },

    'blip_vqa': {
        'adapter_class': 'BlipVqaPredictor',
        'module': 'mozo.adapters.blip_vqa',
        'task_type': 'visual_question_answering',
        'description': 'Salesforce BLIP for visual question answering - Answer questions about images using vision-language understanding',
        'variants': ['base', 'capfilt-large'],
    },

    'datamarkin': {
        'adapter_class': 'DatamarkinPredictor',
        'module': 'mozo.adapters.datamarkin',
        'task_type': 'online_inference',
        'description': 'Datamarkin Vision Service - Cloud-based model inference for keypoint detection, object detection, and segmentation. Variant name is the training_id.',
        'variants': [],  # Dynamic variants - any training_id is valid
    },
}


def get_available_families():
    """
    Get list of all available model families for discovery and API endpoints.

    Problem: Users need to discover which model families are available without
    importing adapters or reading source code. API endpoints need to list available
    families without loading any models.

    Solution: Returns all registered model family names from the registry. This is
    a lightweight, fast operation that doesn't import or instantiate any adapters.

    Returns:
        list: List of model family names (e.g., ['detectron2', 'depth_anything', 'qwen2.5_vl', ...])

    Example:
        ```python
        from mozo.registry import get_available_families

        families = get_available_families()
        print(f"Available model families: {families}")
        # Output: ['detectron2', 'depth_anything', 'qwen2.5_vl', 'paddleocr', ...]

        # Check if a specific family is available
        if 'detectron2' in families:
            print("Detectron2 models are available")
        ```

    Note:
        - This is a fast lookup (no imports, no model loading)
        - Used by REST API /models endpoint
        - Returns all families registered in MODEL_REGISTRY
    """
    return list(MODEL_REGISTRY.keys())


def get_available_variants(family):
    """
    Get list of variant names for a model family from registry for fast discovery.

    Problem: Each model family has multiple variants (e.g., Detectron2 has 27 variants).
    Users need to discover available variants without importing heavy adapter modules or
    loading models. API endpoints need to list variants quickly for documentation and
    validation.

    Solution: Returns variant names from the lightweight registry. This avoids importing
    adapters, which can trigger heavy dependencies (PyTorch, Transformers, etc.).

    IMPORTANT: Registry is for fast discovery only. Adapters are the authoritative source
    for variant configurations. Some adapters may support additional variants not listed
    in the registry - the adapter will still work, this list is just for convenience.

    Args:
        family: Model family name (e.g., 'detectron2', 'depth_anything')

    Returns:
        list: Variant names for the family (e.g., ['mask_rcnn_R_50_FPN_3x', ...])
             Returns empty list for families with dynamic variants (e.g., datamarkin)

    Raises:
        ValueError: If family name is not found in registry

    Example:
        ```python
        from mozo.registry import get_available_variants

        # List all Detectron2 variants
        variants = get_available_variants('detectron2')
        print(f"Detectron2 has {len(variants)} variants")
        print(variants[:3])  # ['mask_rcnn_R_50_FPN_3x', 'mask_rcnn_R_50_C4_1x', ...]

        # Check if specific variant exists
        if 'mask_rcnn_R_50_FPN_3x' in variants:
            print("Mask R-CNN variant is available")
        ```

    Note:
        - Fast lookup (no adapter imports, no model loading)
        - Registry may be out of sync with adapters - this is acceptable
        - Adapters validate variants during instantiation
        - Empty list means dynamic variants (adapter accepts any variant)
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: '{family}'. Available families: {get_available_families()}")

    return MODEL_REGISTRY[family].get('variants', [])


def get_model_info(family, variant=None):
    """
    Get detailed information about a model family from registry.

    Problem: Users need to understand what a model family does, which task type it
    handles, and what variants are available before loading models. API endpoints
    need this metadata for documentation and validation without loading adapters.

    Solution: Returns comprehensive family metadata from registry including task type,
    description, adapter class, module path, and available variants. Optionally validates
    that a specific variant exists in the registry.

    Args:
        family: Model family name (e.g., 'detectron2', 'depth_anything')
        variant: Optional variant name for validation. If provided, checks if variant
                exists in registry (raises ValueError if not found)

    Returns:
        dict: Model family information with keys:
            - family: Family name
            - adapter_class: Adapter class name
            - module: Python module path to adapter
            - task_type: Task category (e.g., 'object_detection', 'ocr')
            - description: Human-readable family description
            - variants: List of available variant names

    Raises:
        ValueError: If family name not found in registry
        ValueError: If variant provided and not found in registry (except datamarkin)

    Example:
        ```python
        from mozo.registry import get_model_info

        # Get family information
        info = get_model_info('detectron2')
        print(f"Task: {info['task_type']}")  # 'object_detection'
        print(f"Description: {info['description']}")
        print(f"Variants: {len(info['variants'])}")  # 27

        # Validate a specific variant exists
        try:
            info = get_model_info('detectron2', 'mask_rcnn_R_50_FPN_3x')
            print("Variant is valid")
        except ValueError as e:
            print(f"Variant not found: {e}")
        ```

    Note:
        - Fast metadata lookup (no adapter imports)
        - Used by REST API /models/{family}/{variant}/info endpoint
        - Datamarkin family accepts any variant (dynamic variants)
        - Variant validation is advisory only - adapters are authoritative
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: '{family}'. Available families: {get_available_families()}")

    family_config = MODEL_REGISTRY[family]

    # Validate variant if provided
    if variant is not None:
        variants = family_config.get('variants', [])
        if variants and variant not in variants:
            # Special case: datamarkin accepts any variant (empty list)
            if family != 'datamarkin':
                raise ValueError(f"Unknown variant '{variant}' for family '{family}'. Available: {variants}")

    return {
        'family': family,
        'adapter_class': family_config['adapter_class'],
        'module': family_config['module'],
        'task_type': family_config['task_type'],
        'description': family_config.get('description', ''),
        'variants': family_config.get('variants', []),
    }
