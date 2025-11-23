"""
Model Factory for Mozo

Factory pattern implementation for dynamic model instantiation.
Handles loading adapter classes and creating model instances from registry configurations.
"""

import importlib
from .registry import MODEL_REGISTRY
from .device import get_default_device


class ModelFactory:
    """
    Factory class for creating model instances dynamically from registry configuration.

    Problem: Different ML frameworks (Detectron2, HuggingFace, PaddleOCR, etc.) have
    completely different APIs and initialization patterns. Hard-coding model instantiation
    for each framework creates rigid, difficult-to-extend code.

    Solution: ModelFactory uses the registry pattern to dynamically import and instantiate
    the correct adapter class for any model family. This enables adding new model families
    without modifying core server code - just register the adapter and it's available.

    The factory is intentionally "dumb" - it only handles adapter loading and instantiation.
    All variant validation, configuration, and model initialization logic lives in the
    adapters themselves, keeping responsibilities clear.

    Example:
        ```python
        from mozo.factory import ModelFactory

        factory = ModelFactory()

        # Factory looks up 'detectron2' in registry, imports adapter, instantiates
        model = factory.create_model('detectron2', 'mask_rcnn_R_50_FPN_3x')

        # Factory handles completely different framework transparently
        model = factory.create_model('depth_anything', 'small')

        # List all available families
        families = factory.get_available_families()
        print(families)  # ['detectron2', 'depth_anything', 'qwen2.5_vl', ...]
        ```

    Note:
        - Adapter classes are cached after first import for performance
        - Registry is for discovery; adapters are source of truth for variants
        - Factory delegates all validation and configuration to adapters
    """

    def __init__(self):
        """Initialize the model factory."""
        self._adapter_cache = {}  # Cache for loaded adapter classes

    def _get_adapter_class(self, module_path, class_name):
        """
        Dynamically import and return an adapter class.

        Args:
            module_path: Python module path (e.g., 'mozo.adapters.detectron2')
            class_name: Class name to import (e.g., 'Detectron2Predictor')

        Returns:
            The adapter class

        Raises:
            ImportError: If module or class cannot be loaded
        """
        cache_key = f"{module_path}.{class_name}"

        # Return cached class if available
        if cache_key in self._adapter_cache:
            return self._adapter_cache[cache_key]

        try:
            # Dynamically import the module
            module = importlib.import_module(module_path)

            # Get the class from the module
            adapter_class = getattr(module, class_name)

            # Cache for future use
            self._adapter_cache[cache_key] = adapter_class

            return adapter_class

        except ImportError as e:
            raise ImportError(
                f"Failed to import module '{module_path}': {e}"
            ) from e
        except AttributeError as e:
            raise ImportError(
                f"Module '{module_path}' does not have class '{class_name}': {e}"
            ) from e

    def create_model(self, family, variant, device=None, **override_params):
        """
        Create a model instance from family and variant identifiers.

        Problem: Each ML framework requires different instantiation code - different imports,
        different parameter names, different initialization patterns. Supporting multiple
        frameworks means either duplicating logic or creating complex conditional code.

        Solution: Factory looks up the adapter class for the requested family in the registry,
        dynamically imports it, and instantiates it with the variant and any additional
        parameters. The adapter handles all framework-specific logic, keeping the factory
        simple and extensible.

        The factory intentionally does minimal work:
        1. Validate family exists in registry
        2. Import the appropriate adapter class (cached after first import)
        3. Apply device auto-detection if not specified
        4. Instantiate adapter with variant + parameters
        5. Return the model instance

        All variant validation, configuration parsing, and model loading happens in the
        adapter, not the factory. This separation of concerns makes both components simpler.

        Args:
            family: Model family name (e.g., 'detectron2', 'depth_anything', 'datamarkin')
            variant: Model variant name (e.g., 'mask_rcnn_R_50_FPN_3x', 'wings-v4')
                    Variant names are adapter-specific; adapters validate variants
            device: Compute device - 'cuda', 'mps', 'cpu', or None (auto-detect)
                   If None, automatically selects best available device
            **override_params: Additional parameters passed directly to adapter constructor
                             Examples: bearer_token for datamarkin

        Returns:
            Instantiated model predictor object with a predict() method for inference

        Raises:
            ValueError: If family name is not found in MODEL_REGISTRY
            ImportError: If adapter module or class cannot be imported
            RuntimeError: If adapter instantiation fails (wrapped exception from adapter)

        Example:
            ```python
            factory = ModelFactory()

            # Standard detection model
            model = factory.create_model('detectron2', 'mask_rcnn_R_50_FPN_3x')
            detections = model.predict(image)

            # Cloud-based model with authentication
            model = factory.create_model('datamarkin', 'wings-v4', bearer_token='your_token')
            detections = model.predict(image)

            # Override device for GPU
            model = factory.create_model('depth_anything', 'small', device='cuda')
            depth_map = model.predict(image)
            ```

        Note:
            - Adapter classes are imported once and cached for performance
            - Factory warns if variant not in registry but still attempts (adapter validates)
            - Registry is for discovery; adapters are authoritative on supported variants
        """
        # Validate family exists
        if family not in MODEL_REGISTRY:
            available_families = list(MODEL_REGISTRY.keys())
            raise ValueError(
                f"Unknown model family: '{family}'. "
                f"Available families: {available_families}"
            )

        family_config = MODEL_REGISTRY[family]

        # Get adapter class information
        module_path = family_config['module']
        class_name = family_config['adapter_class']

        # Optional: Warn if variant not in registry (but still try)
        # Registry is just for discovery - adapter is source of truth
        registry_variants = family_config.get('variants', [])
        if registry_variants and variant not in registry_variants:
            print(f"[ModelFactory] Warning: variant '{variant}' not in registry for '{family}', "
                  f"attempting anyway (adapter will validate)")

        # Load adapter class (lazy import)
        adapter_class = self._get_adapter_class(module_path, class_name)

        # Apply device: use provided device, or auto-detect if None
        effective_device = device if device is not None else get_default_device()
        print(f"[ModelFactory] Using device: {effective_device}" +
              (" (auto-detected)" if device is None else " (user-specified)"))

        # Let adapter handle everything - pass device in params
        try:
            model_instance = adapter_class(variant=variant, device=effective_device, **override_params)
            return model_instance
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate {class_name} for variant '{variant}': {e}"
            ) from e

    def get_available_families(self):
        """
        Get list of all available model families.

        Returns:
            list: Model family names
        """
        return list(MODEL_REGISTRY.keys())

    def get_available_variants(self, family):
        """
        Get list of all available variants for a model family from registry.

        NOTE: Registry is for fast discovery only. Adapters are source of truth.
        Some adapters may support additional variants not listed in registry.

        Args:
            family: Model family name

        Returns:
            list: Variant names for the family

        Raises:
            ValueError: If family not found
        """
        if family not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model family: '{family}'")

        family_config = MODEL_REGISTRY[family]
        variants = family_config.get('variants', [])

        return variants

    def clear_cache(self):
        """Clear the adapter class cache."""
        self._adapter_cache.clear()
