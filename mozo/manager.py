"""
Model Manager for Mozo

Manages the lifecycle of model instances including:
- Lazy loading (models loaded on-demand, not at startup)
- Thread-safe access
- Usage tracking
- Automatic cleanup of inactive models
- Memory management
"""

import time
import gc
from threading import Lock
from typing import Dict, List, Optional
from .factory import ModelFactory


class ModelManager:
    """
    Manages the lifecycle of model instances.

    Key features:
    - Lazy loading: Models are only loaded when first requested
    - Thread-safe: Multiple concurrent requests handled correctly
    - Usage tracking: Timestamp recorded for each model access
    - Memory management: Can unload models to free memory

    Similar to the vision engine's dynamic model deployment pattern.
    """

    def __init__(self):
        """Initialize the model manager."""
        self._models: Dict[str, object] = {}  # model_id → model instance
        self._last_used: Dict[str, float] = {}  # model_id → timestamp
        self._locks: Dict[str, Lock] = {}  # model_id → Lock (for thread-safe loading)
        self._global_lock = Lock()  # Lock for managing _locks dict
        self._factory = ModelFactory()

    def _get_model_id(self, family: str, variant: str) -> str:
        """
        Generate a unique model ID from family and variant.

        Args:
            family: Model family name
            variant: Model variant name

        Returns:
            str: Unique model identifier in format 'family/variant'
        """
        return f"{family}/{variant}"

    def _parse_model_id(self, model_id: str) -> tuple:
        """
        Parse a model ID into family and variant components.

        Args:
            model_id: Model identifier in format 'family/variant'

        Returns:
            tuple: (family, variant)

        Raises:
            ValueError: If model_id format is invalid
        """
        if '/' not in model_id:
            raise ValueError(
                f"Invalid model_id format: '{model_id}'. "
                f"Expected format: 'family/variant' (e.g., 'detectron2/mask_rcnn_R_50_FPN_3x')"
            )

        parts = model_id.split('/', 1)
        return parts[0], parts[1]

    def get_model(self, family: str, variant: str):
        """
        Get a model instance, loading it if necessary (lazy loading).

        This method is thread-safe: if multiple threads request the same model
        simultaneously, only one will load it while others wait.

        Args:
            family: Model family name (e.g., 'detectron2', 'depth_anything')
            variant: Model variant name (e.g., 'mask_rcnn_R_50_FPN_3x', 'small')

        Returns:
            Model predictor instance

        Raises:
            ValueError: If family or variant is invalid
            RuntimeError: If model fails to load

        Example:
            >>> manager = ModelManager()
            >>> model = manager.get_model('detectron2', 'mask_rcnn_R_50_FPN_3x')
            >>> predictions = model.predict(image)
        """
        model_id = self._get_model_id(family, variant)

        # Ensure a lock exists for this model (thread-safe lock creation)
        with self._global_lock:
            if model_id not in self._locks:
                self._locks[model_id] = Lock()

        # Only one thread can load a given model at a time
        with self._locks[model_id]:
            # Check if model is already loaded
            if model_id not in self._models:
                print(f"[ModelManager] Loading model: {model_id} (family={family}, variant={variant})...")
                try:
                    self._models[model_id] = self._factory.create_model(family, variant)
                    print(f"[ModelManager] Model {model_id} loaded successfully.")
                except Exception as e:
                    print(f"[ModelManager] ERROR: Failed to load model {model_id}: {e}")
                    raise RuntimeError(f"Failed to load model {model_id}") from e
            else:
                print(f"[ModelManager] Model {model_id} already loaded, reusing existing instance.")

            # Update last used timestamp
            self._last_used[model_id] = time.time()

            return self._models[model_id]

    def get_model_by_id(self, model_id: str):
        """
        Get a model instance by its full ID (family/variant format).

        Args:
            model_id: Full model identifier (e.g., 'detectron2/mask_rcnn_R_50_FPN_3x')

        Returns:
            Model predictor instance

        Raises:
            ValueError: If model_id format is invalid
        """
        family, variant = self._parse_model_id(model_id)
        return self.get_model(family, variant)

    def unload_model(self, family: str, variant: str) -> bool:
        """
        Explicitly unload a model to free memory.

        Args:
            family: Model family name
            variant: Model variant name

        Returns:
            bool: True if model was unloaded, False if it wasn't loaded

        Example:
            >>> manager.unload_model('detectron2', 'mask_rcnn_R_50_FPN_3x')
        """
        model_id = self._get_model_id(family, variant)
        return self.unload_model_by_id(model_id)

    def unload_model_by_id(self, model_id: str) -> bool:
        """
        Explicitly unload a model by its ID to free memory.

        Args:
            model_id: Full model identifier

        Returns:
            bool: True if model was unloaded, False if it wasn't loaded
        """
        if model_id in self._models:
            print(f"[ModelManager] Unloading model: {model_id}...")
            del self._models[model_id]
            if model_id in self._last_used:
                del self._last_used[model_id]

            # Explicitly trigger garbage collection to free memory
            gc.collect()

            print(f"[ModelManager] Model {model_id} unloaded successfully.")
            return True
        else:
            print(f"[ModelManager] Model {model_id} not loaded, nothing to unload.")
            return False

    def list_loaded_models(self) -> List[str]:
        """
        Get list of currently loaded model IDs.

        Returns:
            list: Model IDs of all loaded models
        """
        return list(self._models.keys())

    def get_inactive_models(self, inactive_seconds: int = 600) -> List[str]:
        """
        Find models that haven't been used in the specified time period.

        Args:
            inactive_seconds: Time threshold in seconds (default: 600 = 10 minutes)

        Returns:
            list: Model IDs of inactive models
        """
        current_time = time.time()
        inactive = []

        for model_id, last_used in self._last_used.items():
            if current_time - last_used > inactive_seconds:
                inactive.append(model_id)

        return inactive

    def cleanup_inactive_models(self, inactive_seconds: int = 600) -> int:
        """
        Automatically unload models that haven't been used recently.

        This is similar to the vision engine's automatic model cleanup.

        Args:
            inactive_seconds: Time threshold in seconds (default: 600 = 10 minutes)

        Returns:
            int: Number of models unloaded

        Example:
            >>> # Unload models inactive for more than 10 minutes
            >>> count = manager.cleanup_inactive_models(600)
            >>> print(f"Unloaded {count} inactive models")
        """
        inactive_models = self.get_inactive_models(inactive_seconds)
        count = 0

        for model_id in inactive_models:
            if self.unload_model_by_id(model_id):
                count += 1

        if count > 0:
            print(f"[ModelManager] Cleanup: Unloaded {count} inactive model(s).")

        return count

    def get_model_info(self, model_id: Optional[str] = None) -> dict:
        """
        Get information about loaded models.

        Args:
            model_id: Optional model ID. If None, returns info about all loaded models

        Returns:
            dict: Model information including load status and last used time
        """
        if model_id is None:
            # Return info for all loaded models
            result = {}
            current_time = time.time()
            for mid in self._models.keys():
                last_used = self._last_used.get(mid, 0)
                result[mid] = {
                    'loaded': True,
                    'last_used': last_used,
                    'inactive_seconds': int(current_time - last_used) if last_used > 0 else 0
                }
            return result
        else:
            # Return info for specific model
            if model_id in self._models:
                last_used = self._last_used.get(model_id, 0)
                current_time = time.time()
                return {
                    'model_id': model_id,
                    'loaded': True,
                    'last_used': last_used,
                    'inactive_seconds': int(current_time - last_used) if last_used > 0 else 0
                }
            else:
                return {
                    'model_id': model_id,
                    'loaded': False
                }

    def unload_all_models(self) -> int:
        """
        Unload all currently loaded models.

        Useful for cleanup or testing.

        Returns:
            int: Number of models unloaded
        """
        model_ids = list(self._models.keys())
        count = 0

        for model_id in model_ids:
            if self.unload_model_by_id(model_id):
                count += 1

        return count
