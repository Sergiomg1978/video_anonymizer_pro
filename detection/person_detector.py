"""
Full-body person detection using YOLOv8.
Detects persons (COCO class 0) in video frames for downstream
tracking and classification.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Import detection library
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except (ImportError, Exception) as e:
    ULTRALYTICS_AVAILABLE = False
    logger.warning(f"Ultralytics not available, YOLO person detection disabled: {e}")

# Config imports
try:
    from config import YOLO_PERSON_MODEL, PERSON_DETECTION_CONFIDENCE_THRESHOLD
except ImportError:
    YOLO_PERSON_MODEL = "yolov8x.pt"
    PERSON_DETECTION_CONFIDENCE_THRESHOLD = 0.5

# COCO class index for 'person'
PERSON_CLASS_ID = 0


@dataclass
class PersonDetection:
    """Person detection result from YOLOv8."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) absolute pixel coordinates
    confidence: float
    class_name: str = "person"

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2,
        )


class PersonDetector:
    """Full-body person detector using YOLOv8.

    Uses the YOLOv8x model (largest variant) for best accuracy.
    Only returns detections for COCO class 0 (person).
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = PERSON_DETECTION_CONFIDENCE_THRESHOLD,
        model_path: str = YOLO_PERSON_MODEL,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = None

        self._initialize_model()

    def _initialize_model(self):
        """Load the YOLOv8 model for person detection."""
        if not ULTRALYTICS_AVAILABLE:
            logger.warning(
                "Ultralytics not installed. PersonDetector will return empty results. "
                "Install with: pip install ultralytics"
            )
            return

        try:
            self.model = YOLO(self.model_path)
            # Move model to the requested device
            if self.device != "cpu":
                try:
                    self.model.to(self.device)
                except Exception as e:
                    logger.warning(
                        f"Failed to move model to {self.device}, falling back to CPU: {e}"
                    )
                    self.device = "cpu"
            logger.info(
                f"YOLOv8 person model loaded ({self.model_path}) on {self.device}"
            )
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 person model ({self.model_path}): {e}")
            self.model = None

    @property
    def is_available(self) -> bool:
        """Whether the detector has a loaded model and can produce results."""
        return self.model is not None

    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """Detect persons in a single frame.

        Args:
            frame: BGR image as numpy array (H, W, 3).

        Returns:
            List of PersonDetection objects for every person found.
        """
        if self.model is None:
            return []

        if frame is None or frame.size == 0:
            logger.warning("detect() received an empty frame")
            return []

        try:
            # Run inference; verbose=False to suppress per-frame logs
            results = self.model(frame, verbose=False)

            if not results:
                return []

            return self._parse_results(results[0], frame.shape)

        except Exception as e:
            logger.error(f"Person detection failed: {e}")
            return []

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[PersonDetection]]:
        """Detect persons in multiple frames.

        Processes frames one at a time through YOLO (ultralytics handles
        internal batching when given a list, but explicit looping gives us
        finer error handling per frame).

        Args:
            frames: List of BGR images as numpy arrays.

        Returns:
            List of detection lists, one per input frame.
        """
        if not frames:
            return []

        if self.model is None:
            return [[] for _ in frames]

        batch_results: List[List[PersonDetection]] = []

        try:
            # Attempt native batch inference for efficiency
            yolo_results = self.model(frames, verbose=False)

            for idx, result in enumerate(yolo_results):
                try:
                    frame_shape = frames[idx].shape
                    detections = self._parse_results(result, frame_shape)
                    batch_results.append(detections)
                except Exception as e:
                    logger.warning(f"Failed to parse results for frame {idx}: {e}")
                    batch_results.append([])

        except Exception as e:
            # Fall back to per-frame processing
            logger.warning(f"Batch inference failed, falling back to per-frame: {e}")
            for frame in frames:
                batch_results.append(self.detect(frame))

        return batch_results

    def _parse_results(
        self, yolo_result, frame_shape: Tuple[int, ...]
    ) -> List[PersonDetection]:
        """Parse YOLO results and filter to person class only.

        Args:
            yolo_result: Single YOLO result object from ultralytics.
            frame_shape: (H, W, C) shape of the source frame for clipping.

        Returns:
            Filtered list of PersonDetection objects.
        """
        detections: List[PersonDetection] = []
        height, width = frame_shape[:2]

        for box in yolo_result.boxes:
            # Filter by class: only keep person (class 0)
            cls_id = int(box.cls[0].cpu().numpy())
            if cls_id != PERSON_CLASS_ID:
                continue

            confidence = float(box.conf[0].cpu().numpy())
            if confidence < self.confidence_threshold:
                continue

            # Extract and clip bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(width, x2))
            y2 = int(min(height, y2))

            # Skip degenerate boxes
            if x2 <= x1 or y2 <= y1:
                continue

            detections.append(
                PersonDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                    class_name="person",
                )
            )

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
