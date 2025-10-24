"""
Model Factory for Mozo

Factory pattern implementation for dynamic model instantiation.
Handles loading adapter classes and creating model instances from registry configurations.
"""

import importlib
from .registry import MODEL_REGISTRY


class ModelFactory:
    """
    Factory class for creating model instances dynamically.

    The factory reads from MODEL_REGISTRY to instantiate the correct adapter class
    with the appropriate parameters for the requested model variant.
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

    def create_model(self, family, variant, **override_params):
        """
        Create a model instance from family and variant identifiers.

        Factory is dumb - just loads adapter and passes variant.
        Adapters handle all validation and configuration.

        Args:
            family: Model family name (e.g., 'detectron2', 'depth_anything', 'datamarkin')
            variant: Model variant name (e.g., 'mask_rcnn_R_50_FPN_3x', 'wings-v4')
            **override_params: Additional parameters passed to adapter
                             (e.g., bearer_token for datamarkin, device override)

        Returns:
            Instantiated model predictor object

        Raises:
            ValueError: If family is not found
            ImportError: If adapter class cannot be loaded

        Example:
            >>> factory = ModelFactory()
            >>> model = factory.create_model('detectron2', 'mask_rcnn_R_50_FPN_3x')
            >>> model = factory.create_model('datamarkin', 'wings-v4', bearer_token='xxx')
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

        # Let adapter handle everything - no special cases
        try:
            model_instance = adapter_class(variant=variant, **override_params)
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
