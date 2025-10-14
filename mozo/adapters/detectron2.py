import cv2
import numpy as np

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
    """

    # Registry of all supported detectron2 model variants
    SUPPORTED_CONFIGS = {
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

        # RPN (Region Proposal Network)
        'rpn_R_50_C4_1x': 'COCO-Detection/rpn_R_50_C4_1x.yaml',
        'rpn_R_50_FPN_1x': 'COCO-Detection/rpn_R_50_FPN_1x.yaml',

        # Fast R-CNN
        'fast_rcnn_R_50_FPN_1x': 'COCO-Detection/fast_rcnn_R_50_FPN_1x.yaml',
    }

    def __init__(self, variant="mask_rcnn_R_50_FPN_3x", confidence_threshold=0.5, device="cpu"):
        """
        Initialize Detectron2 predictor with specific model variant.

        Args:
            variant: Model variant name (e.g., 'mask_rcnn_R_50_FPN_3x', 'faster_rcnn_X_101_32x8d_FPN_3x')
            confidence_threshold: Detection confidence threshold (0.0-1.0)
            device: Device to run on - 'cpu' or 'cuda'

        Raises:
            ValueError: If variant is not supported
        """
        if variant not in self.SUPPORTED_CONFIGS:
            raise ValueError(
                f"Unsupported variant: '{variant}'. "
                f"Choose from: {list(self.SUPPORTED_CONFIGS.keys())}"
            )

        self.variant = variant
        config_file = self.SUPPORTED_CONFIGS[variant]

        print(f"Loading Detectron2 model (variant: {variant})...")
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(config_file))
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(config_file)
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.DEVICE = device

        self.predictor = DefaultPredictor(cfg)

        # Get class names from metadata
        dataset_name = cfg.DATASETS.TRAIN[0] if cfg.DATASETS.TRAIN else "coco_2017_val"
        metadata = MetadataCatalog.get(dataset_name)
        self.class_names = metadata.thing_classes

        print(f"Detectron2 model loaded successfully (variant: {variant}).")

    def predict(self, image: np.ndarray):
        """
        Runs inference on an image and returns PixelFlow Detections.

        Returns:
            pf.detections.Detections: PixelFlow Detections object containing all detected objects
        """
        print("Running prediction...")
        outputs = self.predictor(image)

        # Use PixelFlow's existing converter for Detectron2
        detections = pf.detections.from_detectron2(outputs)

        print(f"Found {len(detections)} objects.")
        return detections
