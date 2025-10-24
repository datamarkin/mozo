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
        'variants': {
            # Mask R-CNN variants (Instance Segmentation)
            'mask_rcnn_R_50_FPN_3x': {
                'variant': 'mask_rcnn_R_50_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'mask_rcnn_R_50_C4_1x': {
                'variant': 'mask_rcnn_R_50_C4_1x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'mask_rcnn_R_50_C4_3x': {
                'variant': 'mask_rcnn_R_50_C4_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'mask_rcnn_R_101_FPN_3x': {
                'variant': 'mask_rcnn_R_101_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'mask_rcnn_X_101_32x8d_FPN_3x': {
                'variant': 'mask_rcnn_X_101_32x8d_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },

            # Faster R-CNN variants (Object Detection)
            'faster_rcnn_R_50_FPN_3x': {
                'variant': 'faster_rcnn_R_50_FPN_3x',
                'confidence_threshold': 0.7,
                'device': 'cpu'
            },
            'faster_rcnn_R_50_C4_1x': {
                'variant': 'faster_rcnn_R_50_C4_1x',
                'confidence_threshold': 0.7,
                'device': 'cpu'
            },
            'faster_rcnn_R_101_FPN_3x': {
                'variant': 'faster_rcnn_R_101_FPN_3x',
                'confidence_threshold': 0.7,
                'device': 'cpu'
            },
            'faster_rcnn_X_101_32x8d_FPN_3x': {
                'variant': 'faster_rcnn_X_101_32x8d_FPN_3x',
                'confidence_threshold': 0.7,
                'device': 'cpu'
            },

            # RetinaNet variants (Object Detection)
            'retinanet_R_50_FPN_1x': {
                'variant': 'retinanet_R_50_FPN_1x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'retinanet_R_50_FPN_3x': {
                'variant': 'retinanet_R_50_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'retinanet_R_101_FPN_3x': {
                'variant': 'retinanet_R_101_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },

            # Keypoint R-CNN variants (Keypoint Detection)
            'keypoint_rcnn_R_50_FPN_3x': {
                'variant': 'keypoint_rcnn_R_50_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'keypoint_rcnn_R_101_FPN_3x': {
                'variant': 'keypoint_rcnn_R_101_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
            'keypoint_rcnn_X_101_32x8d_FPN_3x': {
                'variant': 'keypoint_rcnn_X_101_32x8d_FPN_3x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },

            # RPN variants (Region Proposal Network)
            'rpn_R_50_FPN_1x': {
                'variant': 'rpn_R_50_FPN_1x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },

            # Fast R-CNN variants
            'fast_rcnn_R_50_FPN_1x': {
                'variant': 'fast_rcnn_R_50_FPN_1x',
                'confidence_threshold': 0.5,
                'device': 'cpu'
            },
        }
    },

    'depth_anything': {
        'adapter_class': 'DepthAnythingPredictor',
        'module': 'mozo.adapters.depth_anything',
        'task_type': 'depth_estimation',
        'description': 'Depth Anything V2 models for monocular depth estimation',
        'variants': {
            'small': {
                'variant': 'small'
            },
            'base': {
                'variant': 'base'
            },
            'large': {
                'variant': 'large'
            },
        }
    },

    'qwen2.5_vl': {
        'adapter_class': 'Qwen2_5VLPredictor',
        'module': 'mozo.adapters.qwen2_5_vl',
        'task_type': 'visual_question_answering',
        'description': 'Qwen2.5-VL models for vision-language understanding, VQA, and image analysis',
        'variants': {
            '7b-instruct': {
                'variant': '7b-instruct',
                'device': 'cpu',  # MPS has compatibility issues with Qwen2.5-VL
                'torch_dtype': 'auto'
            },
        }
    },

    'qwen3_vl': {
        'adapter_class': 'Qwen3VLPredictor',
        'module': 'mozo.adapters.qwen3_vl',
        'task_type': 'visual_question_answering_with_reasoning',
        'description': 'Qwen3-VL models with chain-of-thought reasoning for explainable vision-language understanding',
        'variants': {
            '2b-thinking': {
                'variant': '2b-thinking',
                'device': 'cpu',  # CPU recommended for 2B model - stable and efficient
                'torch_dtype': 'auto'
            },
        }
    },

    'paddleocr': {
        'adapter_class': 'PaddleOCRPredictor',
        'module': 'mozo.adapters.paddleocr',
        'task_type': 'ocr',
        'description': 'PaddleOCR PP-OCRv5 - Universal scene text recognition supporting 80+ languages with mobile and server variants',
        'variants': {
            'mobile': {
                'variant': 'mobile',
                'language': 'en',
                'device': 'cpu'
            },
            'server': {
                'variant': 'server',
                'language': 'en',
                'device': 'cpu'
            },
            'mobile-chinese': {
                'variant': 'mobile-chinese',
                'language': 'ch',
                'device': 'cpu'
            },
            'server-chinese': {
                'variant': 'server-chinese',
                'language': 'ch',
                'device': 'cpu'
            },
            'mobile-multilingual': {
                'variant': 'mobile-multilingual',
                'language': 'en',
                'device': 'cpu'
            },
        }
    },

    'ppstructure': {
        'adapter_class': 'PPStructurePredictor',
        'module': 'mozo.adapters.ppstructure',
        'task_type': 'document_analysis',
        'description': 'PP-StructureV3 - Document structure analysis with layout detection, table recognition, and formula extraction',
        'variants': {
            'layout-only': {
                'variant': 'layout-only',
                'language': 'en',
                'device': 'cpu'
            },
            'full': {
                'variant': 'full',
                'language': 'en',
                'device': 'cpu'
            },
            'table-analysis': {
                'variant': 'table-analysis',
                'language': 'en',
                'device': 'cpu'
            },
            'formula-analysis': {
                'variant': 'formula-analysis',
                'language': 'en',
                'device': 'cpu'
            },
        }
    },

    'easyocr': {
        'adapter_class': 'EasyOCRPredictor',
        'module': 'mozo.adapters.easyocr',
        'task_type': 'ocr',
        'description': 'EasyOCR - User-friendly OCR with 80+ languages, easy setup, and good general-purpose accuracy',
        'variants': {
            'english-light': {
                'variant': 'english-light',
                'device': 'cpu'
            },
            'english-full': {
                'variant': 'english-full',
                'device': 'cpu'
            },
            'multilingual': {
                'variant': 'multilingual',
                'device': 'cpu'
            },
            'chinese': {
                'variant': 'chinese',
                'device': 'cpu'
            },
            'custom': {
                'variant': 'custom',
                'languages': ['en'],  # Default, user should override
                'device': 'cpu'
            },
        }
    },

    'stability_inpainting': {
        'adapter_class': 'StabilityInpaintingPredictor',
        'module': 'mozo.adapters.stability_inpainting',
        'task_type': 'image_generation',
        'description': 'Stability AI Stable Diffusion 2 for inpainting, generating new image content based on a mask and prompt.',
        'variants': {
            'default': {
                'variant': 'default',
                'device': 'cpu'
            },
        }
    },

    'florence2': {
        'adapter_class': 'Florence2Predictor',
        'module': 'mozo.adapters.florence2',
        'task_type': 'multi_task_vision',
        'description': 'Microsoft Florence-2 for a variety of vision tasks including object detection, segmentation, captioning, and OCR.',
        'variants': {
            'detection': {
                'variant': 'detection',
                'device': 'cpu'
            },
            'segmentation': {
                'variant': 'segmentation',
                'device': 'cpu'
            },
            'captioning': {
                'variant': 'captioning',
                'device': 'cpu'
            },
            'detailed_captioning': {
                'variant': 'detailed_captioning',
                'device': 'cpu'
            },
            'more_detailed_captioning': {
                'variant': 'more_detailed_captioning',
                'device': 'cpu'
            },
            'ocr': {
                'variant': 'ocr',
                'device': 'cpu'
            },
            'ocr_with_region': {
                'variant': 'ocr_with_region',
                'device': 'cpu'
            },
        }
    },

    'blip_vqa': {
        'adapter_class': 'BlipVqaPredictor',
        'module': 'mozo.adapters.blip_vqa',
        'task_type': 'visual_question_answering',
        'description': 'Salesforce BLIP for Visual Question Answering.',
        'variants': {
            'base': {
                'variant': 'base',
                'device': 'cpu'
            },
            'large': {
                'variant': 'large',
                'device': 'cpu'
            },
        }
    },

    'datamarkin': {
        'adapter_class': 'DatamarkinPredictor',
        'module': 'mozo.adapters.datamarkin',
        'task_type': 'online_inference',
        'description': 'Datamarkin Vision Service - Cloud-based model inference for keypoint detection, object detection, and segmentation. Variant name is the training_id.',
        'variants': {
            # Dynamic variants: Any variant name becomes a training_id automatically
            # No predefined variants needed by default
            #
            # Users can optionally add predefined variants here for convenience:
            # 'wings-v4': {
            #     'variant': 'wings-v4',
            #     'bearer_token': None,  # From env or param
            #     'base_url': 'https://vision.datamarkin.com',
            #     'timeout': 30,
            # },
        }
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

    Args:
        family: Model family name

    Returns:
        list: List of variant names for the family

    Raises:
        ValueError: If family is not in registry
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: '{family}'. Available families: {get_available_families()}")
    return list(MODEL_REGISTRY[family]['variants'].keys())


def get_model_info(family, variant=None):
    """
    Get detailed information about a model family or specific variant.

    Args:
        family: Model family name
        variant: Optional variant name. If None, returns family-level info

    Returns:
        dict: Model information

    Raises:
        ValueError: If family or variant is not found
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: '{family}'. Available families: {get_available_families()}")

    family_config = MODEL_REGISTRY[family]

    if variant is None:
        # Return family-level info
        return {
            'family': family,
            'adapter_class': family_config['adapter_class'],
            'task_type': family_config['task_type'],
            'description': family_config.get('description', ''),
            'num_variants': len(family_config['variants']),
            'variants': list(family_config['variants'].keys())
        }
    else:
        # Return variant-specific info
        if variant not in family_config['variants']:
            raise ValueError(
                f"Unknown variant '{variant}' for family '{family}'. "
                f"Available variants: {list(family_config['variants'].keys())}"
            )
        return {
            'family': family,
            'variant': variant,
            'task_type': family_config['task_type'],
            'parameters': family_config['variants'][variant]
        }
