"""
Head detection module.
Attempts to use a dedicated YOLO head model; falls back to expanding
face bounding boxes when the model is unavailable.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Import detection library
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except (ImportError, Exception) as e:
    ULTRALYTICS_AVAILABLE = False
    logger.warning(f"Ultralytics not available, YOLO head detection disabled: {e}")

# Config imports
try:
    from config import YOLO_HEAD_MODEL, HEAD_DETECTION_CONFIDENCE_THRESHOLD
except ImportError:
    YOLO_HEAD_MODEL = "yolov5-head.pt"
    HEAD_DETECTION_CONFIDENCE_THRESHOLD = 0.4

# Expansion ratios used when deriving head bbox from face bbox
_FACE_TO_HEAD_VERTICAL_UP_RATIO = 0.40   # expand 40% of face height upward
_FACE_TO_HEAD_HORIZONTAL_RATIO = 0.20    # expand 20% of face width on each side


@dataclass
class HeadDetection:
    """Head detection result."""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) absolute pixel coordinates
    confidence: float
    source: str  # "yolo_head", "face_expansion", or other identifier

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]

    @property
    def center(self) -> Tuple[int, int]:
        return (
            (self.bbox[0] + self.bbox[2]) // 2,
            (self.bbox[1] + self.bbox[3]) // 2,
        )


class HeadDetector:
    """Head detector with YOLO model and face-expansion fallback.

    Primary path: uses a YOLO model trained on heads (e.g. yolov5-head.pt).
    Fallback path: when the model is not available or produces no detections,
    expands provided face bounding boxes to approximate the head region.
    """

    def __init__(
        self,
        device: str = "cuda",
        confidence_threshold: float = HEAD_DETECTION_CONFIDENCE_THRESHOLD,
        model_path: str = YOLO_HEAD_MODEL,
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model_path = model_path
        self.model = None
        self._use_fallback = False

        self._initialize_model()

    def _initialize_model(self):
        """Try to load the dedicated head detection YOLO model."""
        if not ULTRALYTICS_AVAILABLE:
            logger.info(
                "Ultralytics not installed. HeadDetector will use face-expansion fallback."
            )
            self._use_fallback = True
            return

        try:
            self.model = YOLO(self.model_path)
            if self.device != "cpu":
                try:
                    self.model.to(self.device)
                except Exception as e:
                    logger.warning(
                        f"Failed to move head model to {self.device}, falling back to CPU: {e}"
                    )
                    self.device = "cpu"
            logger.info(f"YOLO head model loaded ({self.model_path}) on {self.device}")
        except Exception as e:
            logger.warning(
                f"Failed to load YOLO head model ({self.model_path}): {e}. "
                "Will use face-expansion fallback."
            )
            self.model = None
            self._use_fallback = True

    @property
    def is_available(self) -> bool:
        """Whether the detector is usable (model loaded or fallback is possible)."""
        return self.model is not None or self._use_fallback

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        frame: np.ndarray,
        face_detections: Optional[List] = None,
    ) -> List[HeadDetection]:
        """Detect heads in a single frame.

        If the YOLO head model is loaded it is tried first.  If the model
        is unavailable (or returns nothing while face detections are
        provided), the fallback expands each face bbox to approximate the
        head region.

        Args:
            frame: BGR image as numpy array (H, W, 3).
            face_detections: Optional list of face detection objects.
                Each element must have a ``bbox`` attribute of the form
                ``(x1, y1, x2, y2)`` and a ``confidence`` attribute.

        Returns:
            List of HeadDetection results.
        """
        if frame is None or frame.size == 0:
            logger.warning("detect() received an empty frame")
            return []

        detections: List[HeadDetection] = []

        # Primary: YOLO head model
        if self.model is not None:
            detections = self._detect_yolo(frame)

        # Fallback: expand face bboxes when model unavailable or found nothing
        if not detections and face_detections:
            detections = self._detect_from_faces(face_detections, frame.shape)

        return detections

    def detect_batch(
        self,
        frames: List[np.ndarray],
        face_detections_per_frame: Optional[List[Optional[List]]] = None,
    ) -> List[List[HeadDetection]]:
        """Detect heads in multiple frames.

        Args:
            frames: List of BGR images.
            face_detections_per_frame: Optional parallel list of face
                detections per frame.  ``None`` entries or missing
                elements mean no face info is available for that frame.

        Returns:
            List of head detection lists, one per input frame.
        """
        if not frames:
            return []

        if face_detections_per_frame is None:
            face_detections_per_frame = [None] * len(frames)

        # Pad to same length if shorter
        while len(face_detections_per_frame) < len(frames):
            face_detections_per_frame.append(None)

        results: List[List[HeadDetection]] = []
        for frame, face_dets in zip(frames, face_detections_per_frame):
            results.append(self.detect(frame, face_detections=face_dets))

        return results

    # ------------------------------------------------------------------
    # YOLO head detection
    # ------------------------------------------------------------------

    def _detect_yolo(self, frame: np.ndarray) -> List[HeadDetection]:
        """Run YOLO-based head detection on a single frame."""
        detections: List[HeadDetection] = []
        height, width = frame.shape[:2]

        try:
            results = self.model(frame, verbose=False)
            if not results:
                return []

            for box in results[0].boxes:
                confidence = float(box.conf[0].cpu().numpy())
                if confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(max(0, x1))
                y1 = int(max(0, y1))
                x2 = int(min(width, x2))
                y2 = int(min(height, y2))

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    HeadDetection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        source="yolo_head",
                    )
                )

        except Exception as e:
            logger.error(f"YOLO head detection failed: {e}")

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    # ------------------------------------------------------------------
    # Face-expansion fallback
    # ------------------------------------------------------------------

    def _detect_from_faces(
        self,
        face_detections: List,
        frame_shape: Tuple[int, ...],
    ) -> List[HeadDetection]:
        """Derive head bounding boxes by expanding face bboxes.

        Args:
            face_detections: Face detection objects with ``.bbox`` and
                ``.confidence`` attributes.
            frame_shape: (H, W, C) of the source frame (for clipping).

        Returns:
            List of HeadDetection derived from face bboxes.
        """
        height, width = frame_shape[:2]
        detections: List[HeadDetection] = []

        for face in face_detections:
            try:
                face_bbox = face.bbox
                face_confidence = getattr(face, "confidence", 0.5)
            except AttributeError:
                logger.debug("Face detection missing bbox attribute, skipping")
                continue

            head_bbox = self.estimate_head_from_face(
                face_bbox, frame_width=width, frame_height=height
            )

            if head_bbox is None:
                continue

            # Slightly discount confidence since this is an estimate
            head_confidence = face_confidence * 0.9

            detections.append(
                HeadDetection(
                    bbox=head_bbox,
                    confidence=head_confidence,
                    source="face_expansion",
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    @staticmethod
    def estimate_head_from_face(
        face_bbox: Tuple[int, int, int, int],
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Expand a face bounding box to approximate the full head region.

        The head typically includes the forehead, hair, and ears that are
        not covered by a tight face bbox.  This method expands:
          - 40% of face height upward (forehead / hair)
          - 20% of face width on each side (ears / hair)
          - A small downward expansion (10%) for the chin margin.

        Args:
            face_bbox: (x1, y1, x2, y2) of the face.
            frame_width: Optional image width for clipping.
            frame_height: Optional image height for clipping.

        Returns:
            (x1, y1, x2, y2) of the estimated head bbox, or None if the
            input bbox is degenerate.
        """
        fx1, fy1, fx2, fy2 = face_bbox
        face_w = fx2 - fx1
        face_h = fy2 - fy1

        if face_w <= 0 or face_h <= 0:
            return None

        # Horizontal expansion: 20% of face width on each side
        h_expand = int(face_w * _FACE_TO_HEAD_HORIZONTAL_RATIO)

        # Vertical expansion: 40% upward, 10% downward
        v_expand_up = int(face_h * _FACE_TO_HEAD_VERTICAL_UP_RATIO)
        v_expand_down = int(face_h * 0.10)

        hx1 = fx1 - h_expand
        hy1 = fy1 - v_expand_up
        hx2 = fx2 + h_expand
        hy2 = fy2 + v_expand_down

        # Clip to frame boundaries if provided
        if frame_width is not None:
            hx1 = max(0, hx1)
            hx2 = min(frame_width, hx2)
        if frame_height is not None:
            hy1 = max(0, hy1)
            hy2 = min(frame_height, hy2)

        # Ensure non-degenerate
        if hx2 <= hx1 or hy2 <= hy1:
            return None

        return (int(hx1), int(hy1), int(hx2), int(hy2))
