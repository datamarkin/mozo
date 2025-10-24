import os
import io
import numpy as np
import cv2
from PIL import Image

try:
    import requests
except ImportError:
    print("="*50)
    print("ERROR: requests is not installed.")
    print("Please install it with: pip install requests")
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


class DatamarkinPredictor:
    """
    Datamarkin Vision Service adapter for online model inference.

    This adapter provides cloud-based inference by calling Datamarkin's hosted models
    via HTTP API. No local model loading required - all inference happens server-side.

    Key Features:
    - Dynamic training IDs (variant name becomes the training_id)
    - Bearer token authentication (currently sent as api_key in form data)
    - Returns PixelFlow Detections for unified interface
    - Supports keypoint detection, object detection, and segmentation

    Architecture:
    - variant → training_id (direct mapping, no boilerplate)
    - bearer_token → api_key in form data (will migrate to Authorization header)
    - PixelFlow converter → standardized output format
    """

    # Empty SUPPORTED_VARIANTS - all variants are dynamic (any training_id works)
    # The factory will handle creating variant configs on-the-fly
    SUPPORTED_VARIANTS = {}

    def __init__(self, variant, bearer_token=None,
                 base_url='https://vision.datamarkin.com',
                 timeout=30, **kwargs):
        """
        Initialize Datamarkin Vision Service adapter.

        Args:
            variant: Training ID for the model (e.g., 'wings-v4', 'my-custom-model')
                    The variant name directly becomes the training_id - no conversion needed!
            bearer_token: Authentication token (optional)
                         - If provided, sent in request
                         - If None, checks DATAMARKIN_TOKEN environment variable
                         - If still None, assumes public model (no auth)
            base_url: API base URL (default: https://vision.datamarkin.com)
            timeout: Request timeout in seconds (default: 30)
            **kwargs: Additional parameters (reserved for future use)

        Environment Variables:
            DATAMARKIN_TOKEN: Bearer token (fallback if not provided as parameter)

        Example:
            >>> # With explicit token
            >>> predictor = DatamarkinPredictor('wings-v4', bearer_token='xxx')

            >>> # With environment variable
            >>> # export DATAMARKIN_TOKEN="xxx"
            >>> predictor = DatamarkinPredictor('wings-v4')

            >>> # Public model (no auth)
            >>> predictor = DatamarkinPredictor('public-model-id')
        """
        # Variant IS the training_id (clean API design!)
        self.training_id = variant
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self.timeout = timeout

        # Get bearer token from parameter or environment variable
        self.bearer_token = bearer_token or os.getenv('DATAMARKIN_TOKEN')

        # Log initialization
        print(f"Initialized Datamarkin adapter:")
        print(f"  Training ID: {self.training_id}")
        print(f"  Base URL: {self.base_url}")
        print(f"  Timeout: {self.timeout}s")

        if self.bearer_token:
            print(f"  Authentication: Enabled")
        else:
            print(f"  Authentication: Public model (no token)")

    def predict(self, image: np.ndarray):
        """
        Run online inference via Datamarkin Vision Service.

        Args:
            image: Input image as numpy array (H, W, 3) in BGR format (OpenCV standard)

        Returns:
            pf.detections.Detections: PixelFlow Detections object containing:
                - Bounding boxes
                - Class names and IDs
                - Confidence scores
                - Keypoints (if available)
                - Segmentation masks (if available)

        Raises:
            RuntimeError: If API request fails or returns error status

        Example:
            >>> import cv2
            >>> predictor = DatamarkinPredictor('wings-v4', bearer_token='xxx')
            >>> image = cv2.imread('butterfly.jpg')
            >>> detections = predictor.predict(image)
            >>> print(f"Found {len(detections)} objects")
            >>> for det in detections:
            ...     print(f"  {det.class_name}: {det.bbox}")
        """
        print(f"Running Datamarkin online inference...")
        print(f"  Training ID: {self.training_id}")
        print(f"  Image shape: {image.shape}")

        # Datamarkin API expects RGB format
        # Convert from BGR (OpenCV standard) to RGB for PIL
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        # Convert PIL Image to PNG bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)

        # Build API request
        url = f"{self.base_url}/predict/{self.training_id}"
        files = {'file': ('image.png', buffer, 'image/png')}

        # Current authentication method: form data with api_key
        # NOTE: This will migrate to Authorization header in the future
        # When migrating, change these 2 lines:
        #   headers = {'Authorization': f'Bearer {self.bearer_token}'}
        #   payload = {}
        if self.bearer_token:
            payload = {'api_key': self.bearer_token}
        else:
            payload = {}  # Public model - no authentication required

        headers = {}

        # Make HTTP request with timeout
        print(f"  Sending request to: {url}")
        try:
            response = requests.post(
                url,
                data=payload,
                files=files,
                headers=headers,
                timeout=self.timeout
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Datamarkin API timeout after {self.timeout}s. "
                f"The model may be loading or the server is overloaded. "
                f"Try increasing the timeout parameter or retry later."
            )
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                f"Failed to connect to Datamarkin API at {url}. "
                f"Check your internet connection and verify the base_url. Error: {e}"
            )
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Datamarkin API request failed: {e}")

        # Handle HTTP errors
        if response.status_code == 401:
            raise RuntimeError(
                f"Datamarkin API authentication failed (401 Unauthorized). "
                f"Check your bearer_token or DATAMARKIN_TOKEN environment variable."
            )
        elif response.status_code == 404:
            raise RuntimeError(
                f"Datamarkin API error (404 Not Found). "
                f"Training ID '{self.training_id}' not found. "
                f"Verify the training_id is correct."
            )
        elif response.status_code != 200:
            try:
                error_detail = response.json()
            except:
                error_detail = response.text
            raise RuntimeError(
                f"Datamarkin API error ({response.status_code}): {error_detail}"
            )

        # Parse JSON response
        try:
            result = response.json()
        except ValueError as e:
            raise RuntimeError(
                f"Failed to parse Datamarkin API response as JSON. "
                f"Response: {response.text[:200]}"
            )

        # Convert to PixelFlow Detections using built-in converter
        # The from_datamarkin() function handles the specific Datamarkin response format
        try:
            detections = pf.detections.from_datamarkin(result)
        except Exception as e:
            raise RuntimeError(f"Failed to convert Datamarkin response: {e}")

        print(f"Datamarkin inference complete.")
        print(f"  Found {len(detections)} object(s)")

        # Log keypoints if available
        keypoint_count = sum(1 for d in detections.detections if hasattr(d, 'keypoints') and d.keypoints)
        if keypoint_count > 0:
            print(f"  Objects with keypoints: {keypoint_count}")

        return detections
