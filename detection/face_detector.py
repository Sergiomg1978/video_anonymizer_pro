"""
Multi-model face detection system
Combines YOLOv8-Face, MediaPipe, and RetinaFace for robust face detection
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import logging

logger = logging.getLogger(__name__)

# Import detection libraries
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except (ImportError, Exception) as e:
    ULTRALYTICS_AVAILABLE = False
    logger.warning(f"Ultralytics not available, YOLO face detection disabled: {e}")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available, MediaPipe face detection disabled")

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("InsightFace not available, RetinaFace detection disabled")


@dataclass
class FaceDetection:
    """Face detection result"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    landmarks: Optional[np.ndarray] = None  # 5 facial landmarks
    embedding: Optional[np.ndarray] = None  # 512-d face embedding
    source_models: List[str] = None  # Which models detected this face

    def __post_init__(self):
        if self.source_models is None:
            self.source_models = []


class MultiFaceDetector:
    """Multi-model face detector combining YOLO, MediaPipe, and RetinaFace"""

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.3):
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Initialize models
        self.yolo_model = None
        self.mediapipe_detector = None
        self.insightface_app = None

        self._initialize_models()

        # NMS parameters
        self.nms_iou_threshold = 0.4
        self.min_models_for_boost = 2  # Minimum models detecting for confidence boost

    def _initialize_models(self):
        """Initialize all face detection models"""
        # YOLOv8-Face
        if ULTRALYTICS_AVAILABLE:
            try:
                self.yolo_model = YOLO('yolov8n-face.pt')  # Will download automatically
                logger.info("YOLOv8-Face model loaded")
            except Exception as e:
                logger.warning(f"Failed to load YOLOv8-Face: {e}")
                self.yolo_model = None

        # MediaPipe Face Detection
        if MEDIAPIPE_AVAILABLE:
            try:
                mp_face_detection = mp.solutions.face_detection
                self.mediapipe_detector = mp_face_detection.FaceDetection(
                    model_selection=1,  # Long-range model
                    min_detection_confidence=self.confidence_threshold
                )
                logger.info("MediaPipe Face Detection loaded")
            except Exception as e:
                logger.warning(f"Failed to load MediaPipe Face Detection: {e}")
                self.mediapipe_detector = None

        # InsightFace (RetinaFace)
        if INSIGHTFACE_AVAILABLE:
            try:
                self.insightface_app = insightface.app.FaceAnalysis(name='buffalo_l')
                self.insightface_app.prepare(ctx_id=-1 if self.device == "cpu" else 0)
                logger.info("InsightFace RetinaFace loaded")
            except Exception as e:
                logger.warning(f"Failed to load InsightFace: {e}")
                self.insightface_app = None

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a single frame"""
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces in multiple frames"""
        if not frames:
            return []

        # Run all detectors
        yolo_results = self._detect_yolo(frames) if self.yolo_model else [[] for _ in frames]
        mediapipe_results = self._detect_mediapipe(frames) if self.mediapipe_detector else [[] for _ in frames]
        insightface_results = self._detect_insightface(frames) if self.insightface_app else [[] for _ in frames]

        # Fuse results for each frame
        fused_results = []
        for frame_idx in range(len(frames)):
            frame_detections = []
            frame_detections.extend(yolo_results[frame_idx])
            frame_detections.extend(mediapipe_results[frame_idx])
            frame_detections.extend(insightface_results[frame_idx])

            # Apply weighted NMS fusion
            fused_detections = self._fuse_detections(frame_detections)
            fused_results.append(fused_detections)

        return fused_results

    def _detect_yolo(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces using YOLOv8-Face"""
        results = []

        for frame in frames:
            detections = []

            try:
                # Run YOLO inference
                yolo_result = self.yolo_model(frame, verbose=False)[0]

                for box in yolo_result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    confidence = float(box.conf[0].cpu().numpy())

                    if confidence >= self.confidence_threshold:
                        detection = FaceDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            source_models=["yolo"]
                        )
                        detections.append(detection)

            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")

            results.append(detections)

        return results

    def _detect_mediapipe(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces using MediaPipe"""
        results = []

        for frame in frames:
            detections = []

            try:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Run MediaPipe detection
                mp_results = self.mediapipe_detector.process(rgb_frame)

                if mp_results.detections:
                    height, width = frame.shape[:2]

                    for detection in mp_results.detections:
                        bbox = detection.location_data.relative_bounding_box

                        # Convert relative to absolute coordinates
                        x1 = int(bbox.xmin * width)
                        y1 = int(bbox.ymin * height)
                        x2 = int((bbox.xmin + bbox.width) * width)
                        y2 = int((bbox.ymin + bbox.height) * height)

                        confidence = detection.score[0]

                        if confidence >= self.confidence_threshold:
                            detection_obj = FaceDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                source_models=["mediapipe"]
                            )
                            detections.append(detection_obj)

            except Exception as e:
                logger.warning(f"MediaPipe detection failed: {e}")

            results.append(detections)

        return results

    def _detect_insightface(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces using InsightFace (RetinaFace)"""
        results = []

        for frame in frames:
            detections = []

            try:
                # Run InsightFace detection
                insightface_results = self.insightface_app.get(frame)

                for face in insightface_results:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[:4]

                    # InsightFace confidence is in det_score
                    confidence = float(face.det_score)

                    if confidence >= self.confidence_threshold:
                        # Extract landmarks (5 points)
                        landmarks = face.landmark_5 if hasattr(face, 'landmark_5') else None
                        if landmarks is not None:
                            landmarks = landmarks.astype(np.float32)

                        # Extract embedding
                        embedding = face.embedding if hasattr(face, 'embedding') else None
                        if embedding is not None:
                            embedding = embedding.astype(np.float32)

                        detection = FaceDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            landmarks=landmarks,
                            embedding=embedding,
                            source_models=["insightface"]
                        )
                        detections.append(detection)

            except Exception as e:
                logger.warning(f"InsightFace detection failed: {e}")

            results.append(detections)

        return results

    def _fuse_detections(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """Fuse detections from multiple models using weighted NMS"""
        if not detections:
            return []

        # Sort by confidence (highest first)
        detections.sort(key=lambda x: x.confidence, reverse=True)

        fused_detections = []

        while detections:
            # Take the highest confidence detection
            best_detection = detections.pop(0)

            # Find overlapping detections
            overlapping = []
            remaining = []

            for det in detections:
                iou = self._calculate_iou(best_detection.bbox, det.bbox)
                if iou >= self.nms_iou_threshold:
                    overlapping.append(det)
                else:
                    remaining.append(det)

            detections = remaining

            # Fuse overlapping detections
            if overlapping:
                fused_detection = self._merge_detections([best_detection] + overlapping)
                fused_detections.append(fused_detection)
            else:
                fused_detections.append(best_detection)

        return fused_detections

    def _merge_detections(self, detections: List[FaceDetection]) -> FaceDetection:
        """Merge multiple detections of the same face"""
        if not detections:
            return None

        # Collect all source models
        all_models = set()
        for det in detections:
            all_models.update(det.source_models)

        # Confidence boosting: if multiple models agree, increase confidence
        num_models = len(all_models)
        base_confidence = max(det.confidence for det in detections)

        if num_models >= self.min_models_for_boost:
            # Boost confidence for multi-model agreement
            confidence_boost = min(0.2, num_models * 0.1)  # Max 20% boost
            merged_confidence = min(1.0, base_confidence + confidence_boost)
        else:
            merged_confidence = base_confidence

        # Weighted bbox averaging (weight by confidence)
        total_weight = sum(d.confidence for d in detections)
        if total_weight == 0:
            total_weight = len(detections)

        weighted_x1 = sum(d.bbox[0] * d.confidence for d in detections) / total_weight
        weighted_y1 = sum(d.bbox[1] * d.confidence for d in detections) / total_weight
        weighted_x2 = sum(d.bbox[2] * d.confidence for d in detections) / total_weight
        weighted_y2 = sum(d.bbox[3] * d.confidence for d in detections) / total_weight

        merged_bbox = (int(weighted_x1), int(weighted_y1), int(weighted_x2), int(weighted_y2))

        # Use landmarks and embedding from highest confidence detection
        best_detection = max(detections, key=lambda x: x.confidence)

        return FaceDetection(
            bbox=merged_bbox,
            confidence=merged_confidence,
            landmarks=best_detection.landmarks,
            embedding=best_detection.embedding,
            source_models=list(all_models)
        )

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Calculate intersection
        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        # Calculate union
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area