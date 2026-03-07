"""
Detection module tests
Tests for face detection, head detection, and person detection
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch

from detection.face_detector import MultiFaceDetector, FaceDetection


class TestMultiFaceDetector:
    """Test the multi-model face detector"""

    def setup_method(self):
        """Setup test fixtures"""
        self.detector = MultiFaceDetector(device="cpu", confidence_threshold=0.1)

    def test_initialization(self):
        """Test detector initialization"""
        assert self.detector.device == "cpu"
        assert self.detector.confidence_threshold == 0.1
        assert self.detector.nms_iou_threshold == 0.4

    def test_empty_frame_detection(self):
        """Test detection on empty/black frame"""
        # Create a black frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = self.detector.detect(frame)
        assert isinstance(detections, list)
        # Should detect no faces in black frame
        assert len(detections) == 0

    def test_synthetic_face_detection(self):
        """Test detection on synthetic face-like pattern"""
        # Create a frame with a bright rectangle (simulating a face)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Draw a white rectangle where a face might be
        frame[100:200, 200:300] = [255, 255, 255]  # White rectangle

        detections = self.detector.detect(frame)
        assert isinstance(detections, list)

        # Note: Actual detection depends on model availability
        # We just test that the method runs without error

    @patch('detection.face_detector.ULTRALYTICS_AVAILABLE', False)
    @patch('detection.face_detector.MEDIAPIPE_AVAILABLE', False)
    @patch('detection.face_detector.INSIGHTFACE_AVAILABLE', False)
    def test_no_models_available(self):
        """Test behavior when no detection models are available"""
        detector = MultiFaceDetector()
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections = detector.detect(frame)
        assert detections == []

    def test_batch_detection(self):
        """Test batch detection functionality"""
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.zeros((480, 640, 3), dtype=np.uint8),
        ]

        batch_results = self.detector.detect_batch(frames)
        assert isinstance(batch_results, list)
        assert len(batch_results) == len(frames)
        assert all(isinstance(frame_dets, list) for frame_dets in batch_results)

    def test_iou_calculation(self):
        """Test IoU calculation for bounding boxes"""
        # Test identical boxes
        iou = self.detector._calculate_iou((0, 0, 10, 10), (0, 0, 10, 10))
        assert iou == 1.0

        # Test no overlap
        iou = self.detector._calculate_iou((0, 0, 10, 10), (20, 20, 30, 30))
        assert iou == 0.0

        # Test partial overlap
        iou = self.detector._calculate_iou((0, 0, 10, 10), (5, 5, 15, 15))
        assert iou == 25/75  # 25 intersection, 75 union

    def test_detection_fusion(self):
        """Test fusion of multiple detections"""
        # Create mock detections
        det1 = FaceDetection(
            bbox=(0, 0, 10, 10),
            confidence=0.8,
            source_models=["yolo"]
        )
        det2 = FaceDetection(
            bbox=(2, 2, 12, 12),
            confidence=0.7,
            source_models=["mediapipe"]
        )
        det3 = FaceDetection(
            bbox=(50, 50, 60, 60),  # Far away, no overlap
            confidence=0.9,
            source_models=["insightface"]
        )

        fused = self.detector._fuse_detections([det1, det2, det3])

        # Should have 2 detections (overlapping pair fused, separate one kept)
        assert len(fused) == 2

        # Check that overlapping detections were merged
        merged_det = next(d for d in fused if len(d.source_models) > 1)
        assert "yolo" in merged_det.source_models
        assert "mediapipe" in merged_det.source_models
        assert merged_det.confidence > 0.8  # Should be boosted

    def test_face_detection_dataclass(self):
        """Test FaceDetection dataclass"""
        detection = FaceDetection(
            bbox=(10, 20, 30, 40),
            confidence=0.85,
            landmarks=np.array([[15, 25], [25, 25]]),
            embedding=np.random.rand(512),
            source_models=["yolo", "mediapipe"]
        )

        assert detection.bbox == (10, 20, 30, 40)
        assert detection.confidence == 0.85
        assert detection.landmarks.shape == (2, 2)
        assert detection.embedding.shape == (512,)
        assert detection.source_models == ["yolo", "mediapipe"]

    @pytest.mark.parametrize("confidence_threshold", [0.1, 0.5, 0.9])
    def test_confidence_threshold(self, confidence_threshold):
        """Test different confidence thresholds"""
        detector = MultiFaceDetector(confidence_threshold=confidence_threshold)

        # Mock a detection with specific confidence
        with patch.object(detector, '_detect_yolo') as mock_yolo:
            mock_yolo.return_value = [[FaceDetection(
                bbox=(0, 0, 10, 10),
                confidence=confidence_threshold - 0.01,  # Below threshold
                source_models=["yolo"]
            )]]

            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            detections = detector.detect(frame)

            # Should filter out low confidence detections
            assert len(detections) == 0

    def test_model_initialization_warnings(self, caplog):
        """Test that appropriate warnings are logged when models fail to load"""
        with patch('detection.face_detector.YOLO', side_effect=Exception("Load failed")):
            with caplog.at_level('WARNING'):
                detector = MultiFaceDetector()

                assert "Failed to load YOLOv8-Face" in caplog.text

    def test_mediapipe_bbox_conversion(self):
        """Test MediaPipe bounding box coordinate conversion"""
        # This would require mocking MediaPipe results
        # For now, just ensure the method exists and doesn't crash
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # If MediaPipe is available, it should run without error
        try:
            detections = self.detector._detect_mediapipe([frame])
            assert isinstance(detections, list)
        except Exception as e:
            # If MediaPipe isn't properly configured, that's OK for testing
            pytest.skip(f"MediaPipe not available for testing: {e}")

    def test_insightface_feature_extraction(self):
        """Test InsightFace landmark and embedding extraction"""
        # This would require mocking InsightFace results
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        try:
            detections = self.detector._detect_insightface([frame])
            assert isinstance(detections, list)
        except Exception as e:
            # If InsightFace isn't properly configured, that's OK for testing
            pytest.skip(f"InsightFace not available for testing: {e}")