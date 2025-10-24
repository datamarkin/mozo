"""
Model Registry for Mozo

Centralized registry of all available model families and their variants.
This registry maps model families to their adapter classes and supported variants.

Usage:
    # To add a new model family, add an entry to MODEL_REGISTRY
    # To add a new variant to an existing family, add it to the 'variants' dict

Example:
    'detectron2': {
        'adapter_class': 'Detectron2Predictor',
        'task_type': 'object_detection',
        'variants': {
            'mask_rcnn_R_50_FPN_3x': {
                'variant': 'mask_rcnn_R_50_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            }
        }
    }
"""

# Main model registry - maps family names to adapter configurations
MODEL_REGISTRY = {
    'detectron2': {
        'adapter_class': 'Detectron2Predictor',
        'module': 'mozo.adapters.detectron2',
        'task_type': 'object_detection',
        'description': 'Detectron2 models for object detection, instance segmentation, and keypoint detection',
    },

    'depth_anything': {
        'adapter_class': 'DepthAnythingPredictor',
        'module': 'mozo.adapters.depth_anything',
        'task_type': 'depth_estimation',
        'description': 'Depth Anything V2 models for monocular depth estimation',
    },

    'qwen2.5_vl': {
        'adapter_class': 'Qwen2_5VLPredictor',
        'module': 'mozo.adapters.qwen2_5_vl',
        'task_type': 'visual_question_answering',
        'description': 'Qwen2.5-VL models for vision-language understanding, VQA, and image analysis',
    },

    'qwen3_vl': {
        'adapter_class': 'Qwen3VLPredictor',
        'module': 'mozo.adapters.qwen3_vl',
        'task_type': 'visual_question_answering_with_reasoning',
        'description': 'Qwen3-VL models with chain-of-thought reasoning for explainable vision-language understanding',
    },

    'paddleocr': {
        'adapter_class': 'PaddleOCRPredictor',
        'module': 'mozo.adapters.paddleocr',
        'task_type': 'ocr',
        'description': 'PaddleOCR PP-OCRv5 - Universal scene text recognition supporting 80+ languages with mobile and server variants',
    },

    'ppstructure': {
        'adapter_class': 'PPStructurePredictor',
        'module': 'mozo.adapters.ppstructure',
        'task_type': 'document_analysis',
        'description': 'PP-StructureV3 - Document structure analysis with layout detection, table recognition, and formula extraction',
    },

    'easyocr': {
        'adapter_class': 'EasyOCRPredictor',
        'module': 'mozo.adapters.easyocr',
        'task_type': 'ocr',
        'description': 'EasyOCR - User-friendly OCR with 80+ languages, easy setup, and good general-purpose accuracy',
    },

    'stability_inpainting': {
        'adapter_class': 'StabilityInpaintingPredictor',
        'module': 'mozo.adapters.stability_inpainting',
        'task_type': 'image_generation',
        'description': 'Stability AI Stable Diffusion 2 Inpainting - Generate and modify image content using text prompts and masks',
    },

    'florence2': {
        'adapter_class': 'Florence2Predictor',
        'module': 'mozo.adapters.florence2',
        'task_type': 'multi_task_vision',
        'description': 'Microsoft Florence-2 for vision tasks including captioning and OCR (detection/segmentation not yet implemented)',
    },

    'blip_vqa': {
        'adapter_class': 'BlipVqaPredictor',
        'module': 'mozo.adapters.blip_vqa',
        'task_type': 'visual_question_answering',
        'description': 'Salesforce BLIP for visual question answering - Answer questions about images using vision-language understanding',
    },

    'datamarkin': {
        'adapter_class': 'DatamarkinPredictor',
        'module': 'mozo.adapters.datamarkin',
        'task_type': 'online_inference',
        'description': 'Datamarkin Vision Service - Cloud-based model inference for keypoint detection, object detection, and segmentation. Variant name is the training_id.',
    },
}


def get_available_families():
    """
    Get list of all available model families.

    Returns:
        list: List of model family names
    """
    return list(MODEL_REGISTRY.keys())


def get_available_variants(family):
    """
    Get list of all available variants for a model family.

    NOTE: This function is deprecated. Variants are now discovered from adapters.
    Use ModelFactory.get_available_variants() instead.

    Args:
        family: Model family name

    Returns:
        list: Empty list (variants are discovered from adapters)

    Raises:
        ValueError: If family is not in registry
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: '{family}'. Available families: {get_available_families()}")

    # Variants are now in adapters, not registry
    # Use ModelFactory.get_available_variants() instead
    return []


def get_model_info(family, variant=None):
    """
    Get detailed information about a model family.

    NOTE: Variant-specific info is no longer available from registry.
    Variants are discovered from adapters. Use ModelFactory for variant details.

    Args:
        family: Model family name
        variant: Optional variant name (deprecated, ignored)

    Returns:
        dict: Model family information

    Raises:
        ValueError: If family is not found
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: '{family}'. Available families: {get_available_families()}")

    family_config = MODEL_REGISTRY[family]

    # Return family-level info only
    return {
        'family': family,
        'adapter_class': family_config['adapter_class'],
        'module': family_config['module'],
        'task_type': family_config['task_type'],
        'description': family_config.get('description', ''),
    }
