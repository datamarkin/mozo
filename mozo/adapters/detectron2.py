from typing import Union

import cv2
import numpy as np

from ..utils import load_image

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
except ImportError:
    print("="*50)
    print("ERROR: Detectron2 is not installed.")
    print("Please install it following the instructions at:")
    print("https://detectron2.readthedocs.io/en/latest/tutorials/install.html")
    print("="*50)
    raise

try:
    import pixelflow as pf
except ImportError:
    print("="*50)
    print("ERROR: PixelFlow is not installed.")
    print("Please install it with: pip install pixelflow")
    print("="*50)
    raise

class Detectron2Predictor:
    """
    Universal Detectron2 adapter - handles ALL detectron2 model variants.
    Supports multiple model families: Mask R-CNN, Faster R-CNN, RetinaNet, Keypoint R-CNN, etc.

    Config and weights resolve independently:
    - config_path overrides model zoo config derivation
    - checkpoint_path overrides model zoo weights derivation
    - labels are passed through to pixelflow (None = pixelflow defaults)
    """

    # Mapping of variant names to Detectron2 model zoo config files
    _CONFIG_MAP = {
        # Mask R-CNN (Instance Segmentation)
        'mask_rcnn_R_50_FPN_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
        'mask_rcnn_R_50_C4_1x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_1x.yaml',
        'mask_rcnn_R_50_C4_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_C4_3x.yaml',
        'mask_rcnn_R_50_DC5_1x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_1x.yaml',
        'mask_rcnn_R_50_DC5_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_DC5_3x.yaml',
        'mask_rcnn_R_50_FPN_1x': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml',
        'mask_rcnn_R_101_C4_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_101_C4_3x.yaml',
        'mask_rcnn_R_101_DC5_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_101_DC5_3x.yaml',
        'mask_rcnn_R_101_FPN_3x': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
        'mask_rcnn_X_101_32x8d_FPN_3x': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
        # Faster R-CNN (Object Detection)
        'faster_rcnn_R_50_C4_1x': 'COCO-Detection/faster_rcnn_R_50_C4_1x.yaml',
        'faster_rcnn_R_50_C4_3x': 'COCO-Detection/faster_rcnn_R_50_C4_3x.yaml',
        'faster_rcnn_R_50_DC5_1x': 'COCO-Detection/faster_rcnn_R_50_DC5_1x.yaml',
        'faster_rcnn_R_50_DC5_3x': 'COCO-Detection/faster_rcnn_R_50_DC5_3x.yaml',
        'faster_rcnn_R_50_FPN_1x': 'COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml',
        'faster_rcnn_R_50_FPN_3x': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
        'faster_rcnn_R_101_C4_3x': 'COCO-Detection/faster_rcnn_R_101_C4_3x.yaml',
        'faster_rcnn_R_101_DC5_3x': 'COCO-Detection/faster_rcnn_R_101_DC5_3x.yaml',
        'faster_rcnn_R_101_FPN_3x': 'COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml',
        'faster_rcnn_X_101_32x8d_FPN_3x': 'COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml',
        # RetinaNet (Object Detection)
        'retinanet_R_50_FPN_1x': 'COCO-Detection/retinanet_R_50_FPN_1x.yaml',
        'retinanet_R_50_FPN_3x': 'COCO-Detection/retinanet_R_50_FPN_3x.yaml',
        'retinanet_R_101_FPN_3x': 'COCO-Detection/retinanet_R_101_FPN_3x.yaml',
        # Keypoint R-CNN (Keypoint Detection)
        'keypoint_rcnn_R_50_FPN_1x': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_1x.yaml',
        'keypoint_rcnn_R_50_FPN_3x': 'COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml',
        'keypoint_rcnn_R_101_FPN_3x': 'COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml',
        'keypoint_rcnn_X_101_32x8d_FPN_3x': 'COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml',
    }

    def __init__(self, variant="mask_rcnn_R_50_FPN_3x", **kwargs):
        """
        Initialize Detectron2 predictor.

        Config and weights resolve independently — explicit paths override model zoo derivation.

        Args:
            variant: Model variant name, used to derive config/weights from model zoo
                     when config_path/checkpoint_path are not provided.
            **kwargs:
                config_path: Path to local config YAML (overrides model zoo config)
                checkpoint_path: Path to local .pth weights (overrides model zoo weights)
                labels: Structured labels list for pixelflow (optional, None = pixelflow defaults)
                confidence_threshold: Score threshold for predictions (default 0.5)
                device: Compute device (default 'cpu')
        """
        confidence_threshold = kwargs.get('confidence_threshold', 0.5)
        device = kwargs.get('device', 'cpu')
        config_path = kwargs.get('config_path')
        checkpoint_path = kwargs.get('checkpoint_path')

        self.variant = variant

        # Resolve config
        cfg = get_cfg()
        if config_path:
            cfg.merge_from_file(config_path)
        else:
            if variant not in self._CONFIG_MAP:
                raise ValueError(
                    f"Unsupported variant: '{variant}'. "
                    f"Supported variants: {list(self._CONFIG_MAP.keys())}"
                )
            cfg.merge_from_file(model_zoo.get_config_file(self._CONFIG_MAP[variant]))

        # Resolve weights
        if checkpoint_path:
            cfg.MODEL.WEIGHTS = checkpoint_path
        else:
            if variant not in self._CONFIG_MAP:
                raise ValueError(
                    f"Unsupported variant: '{variant}'. "
                    f"Supported variants: {list(self._CONFIG_MAP.keys())}"
                )
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(self._CONFIG_MAP[variant])

        cfg.MODEL.DEVICE = device
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold

        print(f"Loading Detectron2 model (variant: {variant})...")
        self.predictor = DefaultPredictor(cfg)
        self.labels = kwargs.get('labels')  # pass-through to pixelflow, None = pixelflow defaults
        print(f"Detectron2 model loaded (variant: {variant}).")

    def predict(self, image: Union[str, np.ndarray]):
        """
        Runs inference on an image and returns PixelFlow Detections.

        Args:
            image: File path (str) or numpy array (BGR format)

        Returns:
            pf.detections.Detections: PixelFlow Detections object containing all detected objects
        """
        image = load_image(image)
        print("Running prediction...")
        outputs = self.predictor(image)

        detections = pf.detections.from_detectron2(outputs, labels=self.labels)

        print(f"Found {len(detections)} objects.")
        return detections
