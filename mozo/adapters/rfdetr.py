import cv2
import numpy as np
from PIL import Image
from typing import Union


class RFDETRPredictor:
    """Adapter for RF-DETR detection and segmentation models by Roboflow."""

    # Maps mozo variant name → rfdetr class name
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

    def __init__(self, variant: str = 'medium', **kwargs):
        if variant not in self.SUPPORTED_VARIANTS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Choose from: {list(self.SUPPORTED_VARIANTS.keys())}"
            )
        self.variant = variant
        class_name = self.SUPPORTED_VARIANTS[variant]

        print(f"Loading RF-DETR {variant} ({class_name})...")
        import rfdetr
        model_class = getattr(rfdetr, class_name)
        self.model = model_class()
        print(f"RF-DETR {variant} loaded successfully.")

    def predict(self, image: Union[str, np.ndarray], threshold: float = 0.5):
        """
        Run inference on image.

        Args:
            image: numpy array (H, W, 3) in BGR format (OpenCV standard) or file path string
            threshold: confidence threshold for filtering predictions (default 0.5)

        Returns:
            Detections object from pixelflow (includes masks for seg-* variants)
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        sv_detections = self.model.predict(image, threshold=threshold)

        import pixelflow as pf
        return pf.detections.from_supervision(sv_detections)
