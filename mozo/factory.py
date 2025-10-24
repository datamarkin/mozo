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

        Args:
            family: Model family name (e.g., 'detectron2', 'depth_anything', 'datamarkin')
            variant: Model variant name (e.g., 'mask_rcnn_R_50_FPN_3x', 'wings-v4')
                    For datamarkin family, variant becomes the training_id
            **override_params: Additional parameters to override variant defaults
                             (e.g., bearer_token for datamarkin)

        Returns:
            Instantiated model predictor object

        Raises:
            ValueError: If family or variant is not found in registry
            ImportError: If adapter class cannot be loaded

        Example:
            >>> factory = ModelFactory()
            >>> # Standard model
            >>> model = factory.create_model('detectron2', 'mask_rcnn_R_50_FPN_3x')
            >>> # Datamarkin with dynamic variant
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

        # Special handling for datamarkin: dynamic variants
        if family == 'datamarkin':
            # If variant not in registry, create it dynamically
            if variant not in family_config['variants']:
                # Variant name IS the training_id for datamarkin
                variant_params = {
                    'variant': variant,  # training_id
                    'bearer_token': None,
                    'base_url': 'https://vision.datamarkin.com',
                    'timeout': 30,
                }
                print(f"[ModelFactory] Creating dynamic datamarkin variant: {variant}")
            else:
                # Use predefined variant config from registry
                variant_params = family_config['variants'][variant].copy()
        else:
            # Standard behavior: variant must exist in registry
            if variant not in family_config['variants']:
                available_variants = list(family_config['variants'].keys())
                raise ValueError(
                    f"Unknown variant '{variant}' for family '{family}'. "
                    f"Available variants: {available_variants}"
                )
            variant_params = family_config['variants'][variant].copy()

        # Get adapter class information
        module_path = family_config['module']
        class_name = family_config['adapter_class']

        # Merge variant params with override params (override takes precedence)
        final_params = {**variant_params, **override_params}

        # Load the adapter class
        adapter_class = self._get_adapter_class(module_path, class_name)

        # Instantiate the adapter with final parameters
        try:
            model_instance = adapter_class(**final_params)
            return model_instance
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate {class_name} with parameters {final_params}: {e}"
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
        Get list of all available variants for a model family.

        Args:
            family: Model family name

        Returns:
            list: Variant names for the family

        Raises:
            ValueError: If family not found
        """
        if family not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model family: '{family}'")
        return list(MODEL_REGISTRY[family]['variants'].keys())

    def clear_cache(self):
        """Clear the adapter class cache."""
        self._adapter_cache.clear()
