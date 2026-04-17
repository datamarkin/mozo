import cv2
import numpy as np
import torch
from PIL import Image
from typing import Dict, List, Optional, Union

from ..device import get_default_device

try:
    import rfdetr
except ImportError:
    print("=" * 50)
    print("ERROR: rfdetr is not installed.")
    print("Please install it with: pip install rfdetr")
    print("=" * 50)
    raise

try:
    import pixelflow as pf
except ImportError:
    print("=" * 50)
    print("ERROR: PixelFlow is not installed.")
    print("Please install it with: pip install pixelflow")
    print("=" * 50)
    raise


class RFDETRPredictor:
    """Adapter for RF-DETR detection and segmentation models by Roboflow."""

    # Maps mozo variant name → rfdetr class name (pretrained COCO models)
    SUPPORTED_VARIANTS = {
        # Detection variants (Apache 2.0)
        'nano':        'RFDETRNano',
        'small':       'RFDETRSmall',
        'medium':      'RFDETRMedium',
        'large':       'RFDETRLarge',
        # Segmentation variants (Apache 2.0)
        'seg-nano':    'RFDETRSegNano',
        'seg-small':   'RFDETRSegSmall',
        'seg-medium':  'RFDETRSegMedium',
        'seg-large':   'RFDETRSegLarge',
    }

    # Maps (project_type, model_size) → rfdetr class name (fine-tuned checkpoints)
    # Note: RFDETRBase exists as a detection class but has no COCO pretrained variant in mozo.
    # Note: ('segmentation', 'base') is absent — no RFDETRSegBase class exists in rfdetr.
    _FINETUNED_CLASS_MAP = {
        ('detection',    'small'):  'RFDETRSmall',
        ('detection',    'base'):   'RFDETRBase',
        ('detection',    'large'):  'RFDETRLarge',
        ('segmentation', 'small'):  'RFDETRSegSmall',
        ('segmentation', 'medium'): 'RFDETRSegMedium',
        ('segmentation', 'large'):  'RFDETRSegLarge',
    }

    def __init__(self, variant: str = 'medium', device: str = None, **kwargs):
        self.variant = variant
        self.device = device or get_default_device()
        checkpoint_path = kwargs.get('checkpoint_path')

        if checkpoint_path:
            # Fine-tuned checkpoint: variant is an opaque training ID used as cache key only.
            model_size = kwargs.get('model_size')
            project_type = kwargs.get('project_type')
            resolution = kwargs.get('resolution', 560)

            if not model_size or not project_type:
                raise ValueError(
                    "Fine-tuned checkpoint loading requires 'model_size' and 'project_type' kwargs."
                )

            key = (project_type, model_size)
            if key not in self._FINETUNED_CLASS_MAP:
                raise ValueError(
                    f"Unsupported combination (project_type='{project_type}', model_size='{model_size}'). "
                    f"Valid combinations: {list(self._FINETUNED_CLASS_MAP.keys())}"
                )

            class_name = self._FINETUNED_CLASS_MAP[key]
            print(f"Loading RF-DETR fine-tuned '{variant}' ({class_name}, resolution={resolution}, device={self.device})...")
            model_class = getattr(rfdetr, class_name)
            # Do NOT pass num_classes — the checkpoint preserves the trained head.
            # Passing num_classes would reinitialize the head and corrupt the loaded weights.
            self.model = model_class(resolution=resolution, pretrain_weights=checkpoint_path, device=self.device)
            self.labels = kwargs.get('labels')
            print(f"RF-DETR fine-tuned '{variant}' loaded successfully.")

        else:
            # Pretrained COCO model
            if variant not in self.SUPPORTED_VARIANTS:
                raise ValueError(
                    f"Unsupported variant: '{variant}'. "
                    f"Choose from: {list(self.SUPPORTED_VARIANTS.keys())}"
                )
            class_name = self.SUPPORTED_VARIANTS[variant]
            print(f"Loading RF-DETR {variant} ({class_name}, device={self.device})...")
            model_class = getattr(rfdetr, class_name)
            self.model = model_class(device=self.device)
            self.labels = None  # pixelflow defaults
            print(f"RF-DETR {variant} loaded successfully.")

    def predict(self, image: Union[str, np.ndarray], threshold: float = 0.5,
                labels=None):
        """
        Run inference on image.

        Args:
            image: numpy array (H, W, 3) in BGR format (OpenCV standard) or file path string
            threshold: confidence threshold for filtering predictions (default 0.5)
            labels: overrides the adapter's default labels for this call (optional).

        Returns:
            Detections object from pixelflow (includes masks for seg-* variants)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        sv_detections = self.model.predict(image, threshold=threshold)

        if self.device == 'mps':
            torch.mps.empty_cache()

        return pf.detections.from_supervision(
            sv_detections, labels=labels if labels is not None else self.labels
        )
