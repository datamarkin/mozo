"""
Model Registry for Mozo

Lightweight registry for model discovery and routing.
Variant names are listed here for fast discovery without importing adapters.
Full variant configuration lives in adapters (single source of truth).

It exists so /models can answer without importing an adapter -- and therefore without importing
torch. That is the whole reason the variant lists are written out twice.

To add a family: add an entry here, and add the same variants to its adapter's ``VARIANTS``.
The two are kept in step by a test rather than by importing one from the other -- this module
must stay importable without torch, and every adapter pulls torch in.

Example:
    'rfdetr': {
        'adapter_class': 'RFDETRPredictor',
        'module': 'mozo.adapters.rfdetr',
        'task_type': 'object_detection',
        'description': '...',
        'variants': ['nano', 'small'],
    }
"""

# Main model registry - maps family names to adapter configurations
MODEL_REGISTRY = {
    'depth_anything_v2': {
        'adapter_class': 'DepthAnythingV2Predictor',
        'module': 'mozo.adapters.depth_anything_v2',
        'task_type': 'depth_estimation',
        'description': (
            'Depth Anything V2 by TikTok & HKU — monocular depth estimation. '
            '3 relative-depth variants (small/base/large) and the same three sizes '
            'fine-tuned for metric depth indoors (indoor-*) and outdoors (outdoor-*). '
            'Relative base/large are CC-BY-NC-4.0; the rest are Apache 2.0.'
        ),
        'variants': [
            'small', 'base', 'large',
            'indoor-small', 'indoor-base', 'indoor-large',
            'outdoor-small', 'outdoor-base', 'outdoor-large',
        ],
    },

    'rfdetr': {
        'adapter_class': 'RFDETRPredictor',
        'module': 'mozo.adapters.rfdetr',
        'task_type': 'object_detection',
        'description': (
            'RF-DETR by Roboflow — real-time transformer detection & instance segmentation. '
            '4 detection variants (nano/small/medium/large) and 4 segmentation variants '
            '(seg-nano/seg-small/seg-medium/seg-large). All variants Apache 2.0 licensed.'
        ),
        'variants': [
            'nano', 'small', 'medium', 'large',
            'seg-nano', 'seg-small', 'seg-medium', 'seg-large',
        ],
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
        list: List of model family names (e.g., ['depth_anything_v2', 'rfdetr'])

    Example:
        ```python
        from mozo.registry import get_available_families

        families = get_available_families()
        print(f"Available model families: {families}")
        # Output: ['depth_anything_v2', 'rfdetr']

        # Check if a specific family is available
        if 'rfdetr' in families:
            print("RF-DETR models are available")
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

    Problem: Each model family has multiple variants (e.g. Depth Anything V2 has 9).
    Users need to discover available variants without importing heavy adapter modules or
    loading models. API endpoints need to list variants quickly for documentation and
    validation.

    Solution: Returns variant names from the lightweight registry. This avoids importing
    adapters, which can trigger heavy dependencies (PyTorch, Transformers, etc.).

    IMPORTANT: Registry is for fast discovery only. Adapters are the authoritative source
    for variant configurations. Some adapters may support additional variants not listed
    in the registry - the adapter will still work, this list is just for convenience.

    Args:
        family: Model family name (e.g., 'rfdetr', 'depth_anything_v2')

    Returns:
        list: Variant names for the family (e.g., ['mask_rcnn_R_50_FPN_3x', ...])
             An empty list means the family accepts any variant (dynamic variants)

    Raises:
        ValueError: If family name is not found in registry

    Example:
        ```python
        from mozo.registry import get_available_variants

        # List all Depth Anything V2 variants
        variants = get_available_variants('depth_anything_v2')
        print(f"Depth Anything V2 has {len(variants)} variants")
        print(variants[:3])  # ['faster_rcnn_R_50_FPN_1x', 'faster_rcnn_R_50_FPN_3x', ...]

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
        family: Model family name (e.g., 'rfdetr', 'depth_anything_v2')
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
        ValueError: If variant provided and not found in the family's variant list
                   (families with an empty variant list accept any variant)

    Example:
        ```python
        from mozo.registry import get_model_info

        # Get family information
        info = get_model_info('rfdetr')
        print(f"Task: {info['task_type']}")  # 'object_detection'
        print(f"Description: {info['description']}")
        print(f"Variants: {len(info['variants'])}")  # 27

        # Validate a specific variant exists
        try:
            info = get_model_info('rfdetr', 'nano')
            print("Variant is valid")
        except ValueError as e:
            print(f"Variant not found: {e}")
        ```

    Note:
        - Fast metadata lookup (no adapter imports)
        - Used by REST API /models/{family}/{variant}/info endpoint
        - Families with an empty variant list accept any variant
        - Variant validation is advisory only - adapters are authoritative
    """
    if family not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model family: '{family}'. Available families: {get_available_families()}")

    family_config = MODEL_REGISTRY[family]

    # Validate variant if provided. An empty variant list means the family
    # accepts any variant, so there is nothing to validate against.
    if variant is not None:
        variants = family_config.get('variants', [])
        if variants and variant not in variants:
            raise ValueError(f"Unknown variant '{variant}' for family '{family}'. Available: {variants}")

    return {
        'family': family,
        'adapter_class': family_config['adapter_class'],
        'module': family_config['module'],
        'task_type': family_config['task_type'],
        'description': family_config.get('description', ''),
        'variants': family_config.get('variants', []),
    }
