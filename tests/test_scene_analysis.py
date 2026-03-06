"""
Tests for scene analysis module
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from scene_analysis import (
    ShotDetector, ShotBoundary,
    MotionEstimator, MotionVector, TrackState,
    SceneInterpreter, SceneContext
)


class TestShotDetector:
    """Test shot detection functionality"""

    def test_initialization(self):
        """Test ShotDetector initialization"""
        detector = ShotDetector(sensitivity=0.8)
        assert detector.sensitivity == 0.8
        assert detector.histogram_threshold is None
        assert detector.edge_threshold is None

    def test_is_shot_change_no_change(self):
        """Test shot change detection with identical frames"""
        detector = ShotDetector()
        # Create identical frames
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)

        # Should not detect shot change
        assert not detector.is_shot_change(frame, frame)

    def test_is_shot_change_with_difference(self):
        """Test shot change detection with different frames"""
        detector = ShotDetector()
        frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
        frame2 = np.full((100, 100, 3), 255, dtype=np.uint8)

        # Should detect shot change
        assert detector.is_shot_change(frame1, frame2)

    @patch('cv2.VideoCapture')
    def test_detect_shots_empty_video(self, mock_cap):
        """Test shot detection on empty video"""
        mock_cap.return_value.isOpened.return_value = False

        detector = ShotDetector()
        boundaries = detector.detect_shots("nonexistent.mp4")

        assert len(boundaries) == 0


class TestMotionEstimator:
    """Test motion estimation functionality"""

    def test_initialization(self):
        """Test MotionEstimator initialization"""
        estimator = MotionEstimator()
        assert estimator.use_gpu is True  # Default
        assert estimator.prev_frame_gray is None
        assert isinstance(estimator.flow_history, dict)

    def test_estimate_motion_no_frames(self):
        """Test motion estimation with no frames"""
        estimator = MotionEstimator()
        motion_vectors = estimator.estimate_motion(None, None, [])
        assert len(motion_vectors) == 0

    def test_predict_next_position_no_history(self):
        """Test position prediction with no history"""
        estimator = MotionEstimator()
        bbox = estimator.predict_next_position([])
        assert bbox == (0, 0, 0, 0)

    def test_predict_next_position_with_history(self):
        """Test position prediction with track history"""
        estimator = MotionEstimator()

        # Create mock track history
        history = [
            TrackState(frame_number=0, bbox=(10, 10, 20, 20), confidence=0.9),
            TrackState(frame_number=1, bbox=(12, 12, 22, 22), confidence=0.9),
            TrackState(frame_number=2, bbox=(14, 14, 24, 24), confidence=0.9),
        ]

        predicted_bbox = estimator.predict_next_position(history)

        # Should predict continuation of motion (moving right and down)
        assert predicted_bbox[0] > 14  # x1 should be greater
        assert predicted_bbox[1] > 14  # y1 should be greater

    def test_smooth_motion_trajectory(self):
        """Test motion trajectory smoothing"""
        estimator = MotionEstimator()

        bboxes = [
            (10, 10, 20, 20),
            (12, 12, 22, 22),
            (14, 14, 24, 24),
            (16, 16, 26, 26),
            (18, 18, 28, 28),
        ]

        smoothed = estimator.smooth_motion_trajectory(bboxes, window_size=3)

        assert len(smoothed) == len(bboxes)
        # Smoothed trajectory should be less jerky
        for bbox in smoothed:
            assert len(bbox) == 4


class TestSceneInterpreter:
    """Test scene interpretation functionality"""

    def test_initialization(self):
        """Test SceneInterpreter initialization"""
        interpreter = SceneInterpreter(use_vlm=False)
        assert interpreter.use_vlm is False
        assert interpreter.device == "cuda"
        assert interpreter.frame_width is None
        assert interpreter.frame_height is None

    def test_analyze_frame_empty(self):
        """Test scene analysis with empty inputs"""
        interpreter = SceneInterpreter(use_vlm=False)

        # Create a mock frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        context = interpreter.analyze_frame(frame, [], [])

        assert isinstance(context, SceneContext)
        assert context.scene_type == "empty"
        assert context.num_persons_visible == 0
        assert not context.woman_visible
        assert not context.child_visible

    def test_classify_scene_type_both_visible(self):
        """Test scene type classification for both persons visible"""
        interpreter = SceneInterpreter()

        # Mock persons data
        persons = [
            {
                'bbox': (100, 100, 150, 200),
                'classification': Mock(is_adult=True, is_female=True, confidence=0.9),
                'confidence': 0.9,
                'is_target': True
            },
            {
                'bbox': (200, 100, 250, 180),
                'classification': Mock(is_adult=False, is_female=False, confidence=0.8),
                'confidence': 0.8,
                'is_target': False
            }
        ]

        scene_type = interpreter._classify_scene_type(persons, True, True)
        assert scene_type in ["both_visible", "both_separate"]

    def test_estimate_distance(self):
        """Test distance estimation"""
        interpreter = SceneInterpreter()
        interpreter.frame_height = 480
        interpreter.frame_width = 640

        # Mock persons with different sizes
        persons = [
            {'bbox': (100, 100, 200, 300)},  # Large person (close)
            {'bbox': (300, 150, 350, 200)},  # Small person (far)
        ]

        distance = interpreter._estimate_distance(persons)
        assert distance in ["close", "medium", "far"]

    def test_get_frame_region_hint(self):
        """Test frame region hint generation"""
        interpreter = SceneInterpreter()
        interpreter.frame_height = 480
        interpreter.frame_width = 640

        # Test center region
        region = interpreter._get_bbox_region((300, 200, 340, 280))
        assert region == "center"

        # Test edge regions
        region = interpreter._get_bbox_region((10, 10, 50, 50))
        assert "left" in region or "top" in region

    def test_should_invoke_vlm(self):
        """Test VLM invocation decision"""
        interpreter = SceneInterpreter(use_vlm=True)

        # Should invoke with low confidence
        assert interpreter._should_invoke_vlm([0.3, 0.2, 0.4])

        # Should not invoke with high confidence
        assert not interpreter._should_invoke_vlm([0.9, 0.8, 0.85])


class TestDataClasses:
    """Test dataclasses"""

    def test_shot_boundary(self):
        """Test ShotBoundary dataclass"""
        boundary = ShotBoundary(
            frame_number=100,
            timestamp=3.33,
            boundary_type="cut",
            confidence=0.95
        )

        assert boundary.frame_number == 100
        assert boundary.timestamp == 3.33
        assert boundary.boundary_type == "cut"
        assert boundary.confidence == 0.95

    def test_motion_vector(self):
        """Test MotionVector dataclass"""
        vector = MotionVector(
            track_id=1,
            velocity_x=2.5,
            velocity_y=-1.2,
            direction=45.0,
            speed=2.84,
            confidence=0.9
        )

        assert vector.track_id == 1
        assert vector.velocity_x == 2.5
        assert vector.direction == 45.0

    def test_scene_context(self):
        """Test SceneContext dataclass"""
        context = SceneContext(
            scene_type="both_visible",
            num_persons_visible=2,
            woman_visible=True,
            woman_confidence=0.85,
            child_visible=True,
            estimated_distance="medium",
            occlusion_detected=False,
            frame_region_hint="center"
        )

        assert context.scene_type == "both_visible"
        assert context.num_persons_visible == 2
        assert context.woman_visible
        assert context.woman_confidence == 0.85