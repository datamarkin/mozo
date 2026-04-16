from typing import Union

import cv2
import numpy as np

from ..utils import load_image

try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data import MetadataCatalog
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

    Self-contained adapter with complete configuration.
    """

    # Complete variant configuration (single source of truth)
    SUPPORTED_VARIANTS = {
        'mask_rcnn_R_50_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'mask_rcnn_R_50_C4_1x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'mask_rcnn_R_50_C4_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'mask_rcnn_R_50_DC5_1x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'mask_rcnn_R_50_DC5_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'mask_rcnn_R_50_FPN_1x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'mask_rcnn_R_101_C4_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'mask_rcnn_R_101_DC5_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'mask_rcnn_R_101_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'mask_rcnn_X_101_32x8d_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_R_50_C4_1x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_R_50_C4_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_R_50_DC5_1x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_R_50_DC5_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_R_50_FPN_1x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_R_50_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_R_101_C4_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_R_101_DC5_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_R_101_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'faster_rcnn_X_101_32x8d_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'retinanet_R_50_FPN_1x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'retinanet_R_50_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'retinanet_R_101_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'keypoint_rcnn_R_50_FPN_1x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'keypoint_rcnn_R_50_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'keypoint_rcnn_R_101_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
        'keypoint_rcnn_X_101_32x8d_FPN_3x': {'confidence_threshold': 0.5, 'device': 'cpu'},
    }

    # Mapping of variant names to Detectron2 model zoo config files (implementation detail)
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
        Initialize Detectron2 predictor with specific model variant.

        Args:
            variant: Model variant name (e.g., 'mask_rcnn_R_50_FPN_3x', 'faster_rcnn_X_101_32x8d_FPN_3x')
            **kwargs: Override default parameters (confidence_threshold, device)
                For fine-tuned models, also accepts:
                - checkpoint_path: Path to local .pth weights file
                - class_names: List of class names (required with checkpoint_path)
                - num_classes: Number of classes (defaults to len(class_names))

        Raises:
            ValueError: If variant is not supported or class_names missing for fine-tuned model
        """
        checkpoint_path = kwargs.get('checkpoint_path')

        if checkpoint_path:
            # --- Fine-tuned model ---
            class_names = kwargs.get('class_names')
            if not class_names:
                raise ValueError("Fine-tuned checkpoint requires 'class_names'.")

            num_classes = kwargs.get('num_classes', len(class_names))
            confidence_threshold = kwargs.get('confidence_threshold', 0.5)
            device = kwargs.get('device', 'cpu')

            if variant not in self._CONFIG_MAP:
                raise ValueError(
                    f"Unsupported variant: '{variant}'. "
                    f"Supported variants: {list(self._CONFIG_MAP.keys())}"
                )

            self.variant = variant
            config_file = self._CONFIG_MAP[variant]

            print(f"Loading fine-tuned Detectron2 model (variant: {variant})...")
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
            cfg.MODEL.WEIGHTS = checkpoint_path
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
            cfg.MODEL.DEVICE = device

            if 'retinanet' in variant:
                cfg.MODEL.RETINANET.NUM_CLASSES = num_classes

            self.predictor = DefaultPredictor(cfg)
            self.class_names = class_names
            print(f"Fine-tuned Detectron2 model loaded (variant: {variant}).")

        else:
            # --- Pre-trained COCO model ---
            if variant not in self.SUPPORTED_VARIANTS:
                raise ValueError(
                    f"Unsupported variant: '{variant}'. "
                    f"Supported variants: {list(self.SUPPORTED_VARIANTS.keys())}"
                )

            config = {**self.SUPPORTED_VARIANTS[variant], **kwargs}
            confidence_threshold = config.get('confidence_threshold', 0.5)
            device = config.get('device', 'cpu')

            self.variant = variant
            config_file = self._CONFIG_MAP[variant]

            print(f"Loading Detectron2 model (variant: {variant})...")
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(config_file))
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
            cfg.MODEL.DEVICE = device

            self.predictor = DefaultPredictor(cfg)

            dataset_name = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else "coco_2017_val"
            metadata = MetadataCatalog.get(dataset_name)
            self.class_names = metadata.thing_classes

            print(f"Detectron2 model loaded successfully (variant: {variant}).")

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

        # Use PixelFlow's existing converter for Detectron2
        detections = pf.detections.from_detectron2(outputs, class_names=self.class_names)

        print(f"Found {len(detections)} objects.")
        return detections
