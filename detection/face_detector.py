"""
Multi-model face detection system
Combines OpenCV DNN, MediaPipe Tasks API, and InsightFace/RetinaFace
for robust face detection.
"""

import os
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import logging
import urllib.request

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency imports – use INFO (not WARNING) for optional libs
# ---------------------------------------------------------------------------

# MediaPipe (Tasks API, >= 0.10.x)
MEDIAPIPE_AVAILABLE = False
try:
    import mediapipe as _mp
    from mediapipe.tasks.python import BaseOptions as _MPBaseOptions
    from mediapipe.tasks.python.vision import (
        FaceDetector as _MPFaceDetector,
        FaceDetectorOptions as _MPFaceDetectorOptions,
    )
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    logger.info("MediaPipe not installed – MediaPipe face detection disabled")

# InsightFace (RetinaFace)
INSIGHTFACE_AVAILABLE = False
try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    logger.info("InsightFace not installed – RetinaFace detection disabled")

# Directory for downloaded model files
_MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def _ensure_model_dir():
    """Create models directory if it doesn't exist."""
    os.makedirs(_MODEL_DIR, exist_ok=True)


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
    """Multi-model face detector combining OpenCV DNN, MediaPipe, and RetinaFace"""

    # OpenCV DNN face detector model URLs (ships with opencv-contrib or can be downloaded)
    _OPENCV_DNN_PROTO_URL = (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/"
        "face_detector/deploy.prototxt"
    )
    _OPENCV_DNN_MODEL_URL = (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/"
        "res10_300x300_ssd_iter_140000.caffemodel"
    )
    # MediaPipe face detection model
    _MEDIAPIPE_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_detector/blaze_face_short_range/float16/latest/"
        "blaze_face_short_range.tflite"
    )

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.3):
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Initialize models
        self.opencv_dnn_net = None
        self.mediapipe_detector = None
        self.insightface_app = None

        self._initialize_models()

        # NMS parameters
        self.nms_iou_threshold = 0.4
        self.min_models_for_boost = 2

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def _initialize_models(self):
        """Initialize all face detection models"""
        _ensure_model_dir()

        # 1. OpenCV DNN face detector (replaces non-existent yolov8n-face)
        self._init_opencv_dnn()

        # 2. MediaPipe Face Detection (Tasks API)
        if MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()

        # 3. InsightFace (RetinaFace)
        if INSIGHTFACE_AVAILABLE:
            self._init_insightface()

    def _download_file(self, url: str, dest: str) -> bool:
        """Download a file if it doesn't already exist. Returns True on success."""
        if os.path.exists(dest):
            return True
        try:
            logger.info(f"Downloading {os.path.basename(dest)}...")
            urllib.request.urlretrieve(url, dest)
            logger.info(f"Downloaded {os.path.basename(dest)}")
            return True
        except Exception as e:
            logger.warning(f"Failed to download {url}: {e}")
            return False

    def _init_opencv_dnn(self):
        """Initialize OpenCV DNN SSD face detector."""
        proto_path = os.path.join(_MODEL_DIR, "deploy.prototxt")
        model_path = os.path.join(_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

        if self._download_file(self._OPENCV_DNN_PROTO_URL, proto_path) and \
           self._download_file(self._OPENCV_DNN_MODEL_URL, model_path):
            try:
                self.opencv_dnn_net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                if self.device != "cpu":
                    try:
                        self.opencv_dnn_net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                        self.opencv_dnn_net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                    except Exception:
                        # CUDA not available for OpenCV DNN, use CPU
                        pass
                logger.info("OpenCV DNN face detector loaded")
            except Exception as e:
                logger.warning(f"Failed to load OpenCV DNN face detector: {e}")
                self.opencv_dnn_net = None
        else:
            logger.info("OpenCV DNN face detector models not available")

    def _init_mediapipe(self):
        """Initialize MediaPipe Face Detection using Tasks API."""
        model_path = os.path.join(_MODEL_DIR, "blaze_face_short_range.tflite")

        if not self._download_file(self._MEDIAPIPE_MODEL_URL, model_path):
            logger.info("MediaPipe face model not available")
            return

        try:
            options = _MPFaceDetectorOptions(
                base_options=_MPBaseOptions(model_asset_path=model_path),
                min_detection_confidence=self.confidence_threshold,
            )
            self.mediapipe_detector = _MPFaceDetector.create_from_options(options)
            logger.info("MediaPipe Face Detection loaded (Tasks API)")
        except Exception as e:
            logger.warning(f"Failed to load MediaPipe Face Detection: {e}")
            self.mediapipe_detector = None

    def _init_insightface(self):
        """Initialize InsightFace (RetinaFace)."""
        try:
            self.insightface_app = insightface.app.FaceAnalysis(name='buffalo_l')
            self.insightface_app.prepare(ctx_id=-1 if self.device == "cpu" else 0)
            logger.info("InsightFace RetinaFace loaded")
        except Exception as e:
            logger.warning(f"Failed to load InsightFace: {e}")
            self.insightface_app = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """Detect faces in a single frame"""
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces in multiple frames"""
        if not frames:
            return []

        # Run all detectors
        opencv_results = self._detect_opencv_dnn(frames) if self.opencv_dnn_net is not None else [[] for _ in frames]
        mediapipe_results = self._detect_mediapipe(frames) if self.mediapipe_detector else [[] for _ in frames]
        insightface_results = self._detect_insightface(frames) if self.insightface_app else [[] for _ in frames]

        # Fuse results for each frame
        fused_results = []
        for frame_idx in range(len(frames)):
            frame_detections = []
            frame_detections.extend(opencv_results[frame_idx])
            frame_detections.extend(mediapipe_results[frame_idx])
            frame_detections.extend(insightface_results[frame_idx])

            # Apply weighted NMS fusion
            fused_detections = self._fuse_detections(frame_detections)
            fused_results.append(fused_detections)

        return fused_results

    # ------------------------------------------------------------------
    # OpenCV DNN face detection
    # ------------------------------------------------------------------

    def _detect_opencv_dnn(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces using OpenCV DNN SSD face detector."""
        results = []

        for frame in frames:
            detections = []
            try:
                height, width = frame.shape[:2]
                blob = cv2.dnn.blobFromImage(
                    frame, 1.0, (300, 300), (104.0, 177.0, 123.0),
                    swapRB=False, crop=False
                )
                self.opencv_dnn_net.setInput(blob)
                output = self.opencv_dnn_net.forward()

                for i in range(output.shape[2]):
                    confidence = float(output[0, 0, i, 2])
                    if confidence >= self.confidence_threshold:
                        x1 = int(output[0, 0, i, 3] * width)
                        y1 = int(output[0, 0, i, 4] * height)
                        x2 = int(output[0, 0, i, 5] * width)
                        y2 = int(output[0, 0, i, 6] * height)

                        # Clamp to frame
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)

                        if x2 > x1 and y2 > y1:
                            detections.append(FaceDetection(
                                bbox=(x1, y1, x2, y2),
                                confidence=confidence,
                                source_models=["opencv_dnn"]
                            ))
            except Exception as e:
                logger.warning(f"OpenCV DNN detection failed: {e}")

            results.append(detections)
        return results

    # ------------------------------------------------------------------
    # MediaPipe face detection (Tasks API)
    # ------------------------------------------------------------------

    def _detect_mediapipe(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces using MediaPipe Tasks API"""
        results = []

        for frame in frames:
            detections = []

            try:
                # Convert BGR to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width = frame.shape[:2]

                # Create MediaPipe Image from numpy array
                mp_image = _mp.Image(
                    image_format=_mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # Run detection
                mp_result = self.mediapipe_detector.detect(mp_image)

                if mp_result.detections:
                    for detection in mp_result.detections:
                        bbox = detection.bounding_box

                        x1 = max(0, bbox.origin_x)
                        y1 = max(0, bbox.origin_y)
                        x2 = min(width, bbox.origin_x + bbox.width)
                        y2 = min(height, bbox.origin_y + bbox.height)

                        # Get highest category score
                        confidence = 0.0
                        if detection.categories:
                            confidence = detection.categories[0].score

                        if confidence >= self.confidence_threshold and x2 > x1 and y2 > y1:
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

    # ------------------------------------------------------------------
    # InsightFace detection
    # ------------------------------------------------------------------

    def _detect_insightface(self, frames: List[np.ndarray]) -> List[List[FaceDetection]]:
        """Detect faces using InsightFace (RetinaFace)"""
        results = []

        for frame in frames:
            detections = []

            try:
                insightface_results = self.insightface_app.get(frame)

                for face in insightface_results:
                    bbox = face.bbox.astype(int)
                    x1, y1, x2, y2 = bbox[:4]

                    confidence = float(face.det_score)

                    if confidence >= self.confidence_threshold:
                        landmarks = face.landmark_5 if hasattr(face, 'landmark_5') else None
                        if landmarks is not None:
                            landmarks = landmarks.astype(np.float32)

                        embedding = face.embedding if hasattr(face, 'embedding') else None
                        if embedding is not None:
                            embedding = embedding.astype(np.float32)

                        detections.append(FaceDetection(
                            bbox=(x1, y1, x2, y2),
                            confidence=confidence,
                            landmarks=landmarks,
                            embedding=embedding,
                            source_models=["insightface"]
                        ))

            except Exception as e:
                logger.warning(f"InsightFace detection failed: {e}")

            results.append(detections)

        return results

    # ------------------------------------------------------------------
    # NMS fusion
    # ------------------------------------------------------------------

    def _fuse_detections(self, detections: List[FaceDetection]) -> List[FaceDetection]:
        """Fuse detections from multiple models using weighted NMS"""
        if not detections:
            return []

        detections.sort(key=lambda x: x.confidence, reverse=True)
        fused_detections = []

        while detections:
            best_detection = detections.pop(0)
            overlapping = []
            remaining = []

            for det in detections:
                iou = self._calculate_iou(best_detection.bbox, det.bbox)
                if iou >= self.nms_iou_threshold:
                    overlapping.append(det)
                else:
                    remaining.append(det)

            detections = remaining

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

        all_models = set()
        for det in detections:
            all_models.update(det.source_models)

        num_models = len(all_models)
        base_confidence = max(det.confidence for det in detections)

        if num_models >= self.min_models_for_boost:
            confidence_boost = min(0.2, num_models * 0.1)
            merged_confidence = min(1.0, base_confidence + confidence_boost)
        else:
            merged_confidence = base_confidence

        total_weight = sum(d.confidence for d in detections)
        if total_weight == 0:
            total_weight = len(detections)

        weighted_x1 = sum(d.bbox[0] * d.confidence for d in detections) / total_weight
        weighted_y1 = sum(d.bbox[1] * d.confidence for d in detections) / total_weight
        weighted_x2 = sum(d.bbox[2] * d.confidence for d in detections) / total_weight
        weighted_y2 = sum(d.bbox[3] * d.confidence for d in detections) / total_weight

        merged_bbox = (int(weighted_x1), int(weighted_y1), int(weighted_x2), int(weighted_y2))

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

        x1_inter = max(x1_1, x1_2)
        y1_inter = max(y1_1, y1_2)
        x2_inter = min(x2_1, x2_2)
        y2_inter = min(y2_1, y2_2)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)

        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area
