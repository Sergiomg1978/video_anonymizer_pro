"""
Shot change detection for video analysis
Detects cuts and transitions between shots using histogram differences and edge detection
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class ShotBoundary:
    """Represents a shot boundary in the video"""
    frame_number: int
    timestamp: float
    boundary_type: str  # "cut", "fade", "dissolve"
    confidence: float

class ShotDetector:
    """
    Detects shot changes in video using multiple methods:
    1. Histogram difference between consecutive frames
    2. Edge detection for smooth transitions
    3. Adaptive thresholding based on video statistics
    """

    def __init__(self, sensitivity: float = 0.7):
        """
        Initialize shot detector

        Args:
            sensitivity: Detection sensitivity (0.0-1.0), higher = more sensitive
        """
        self.sensitivity = sensitivity
        self.histogram_threshold = None
        self.edge_threshold = None

    def detect_shots(self, video_path: str) -> List[ShotBoundary]:
        """
        Detect shot boundaries in the entire video

        Args:
            video_path: Path to the video file

        Returns:
            List of detected shot boundaries
        """
        logger.info(f"Detecting shots in video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # First pass: collect statistics for adaptive thresholding
        self._compute_adaptive_thresholds(cap)

        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        shot_boundaries = []
        prev_frame = None
        prev_hist = None
        prev_edges = None

        frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_hist = self._compute_histogram(frame)
            curr_edges = self._compute_edges(frame)

            if prev_frame is not None:
                # Check for shot change
                boundary = self._detect_shot_change(
                    prev_frame, frame, prev_hist, curr_hist,
                    prev_edges, curr_edges, frame_number
                )
                if boundary:
                    shot_boundaries.append(boundary)

            # Prepare for next iteration (reuse already-computed values)
            prev_frame = frame.copy()
            prev_hist = curr_hist
            prev_edges = curr_edges

            frame_number += 1

            if frame_number % 1000 == 0:
                logger.info(f"Processed {frame_number}/{total_frames} frames")

        cap.release()
        logger.info(f"Detected {len(shot_boundaries)} shot boundaries")

        return shot_boundaries

    def is_shot_change(self, frame_a: np.ndarray, frame_b: np.ndarray) -> bool:
        """
        Check if there's a shot change between two frames

        Args:
            frame_a: First frame
            frame_b: Second frame

        Returns:
            True if shot change detected
        """
        if self.histogram_threshold is None:
            # Use default thresholds if not computed
            self.histogram_threshold = 0.3
            self.edge_threshold = 0.2

        hist_diff = self._histogram_difference(
            self._compute_histogram(frame_a),
            self._compute_histogram(frame_b)
        )

        edge_diff = self._edge_difference(
            self._compute_edges(frame_a),
            self._compute_edges(frame_b)
        )

        # Combine metrics with sensitivity
        combined_score = (hist_diff + edge_diff) / 2.0
        threshold = (self.histogram_threshold + self.edge_threshold) / 2.0
        threshold *= (1.0 - self.sensitivity * 0.5)  # Adjust threshold based on sensitivity

        return combined_score > threshold

    def _compute_adaptive_thresholds(self, cap: cv2.VideoCapture):
        """Compute adaptive thresholds based on video statistics"""
        logger.info("Computing adaptive thresholds...")

        # Sample frames throughout the video
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_frames = []

        # Sample at regular intervals
        sample_count = min(100, total_frames // 10)  # Sample every 10th frame, max 100 samples
        step = max(1, total_frames // sample_count)

        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                sample_frames.append(frame)
                if len(sample_frames) >= sample_count:
                    break

        if len(sample_frames) < 2:
            # Fallback to default thresholds
            self.histogram_threshold = 0.3
            self.edge_threshold = 0.2
            return

        # Compute differences between consecutive samples
        hist_diffs = []
        edge_diffs = []

        prev_hist = self._compute_histogram(sample_frames[0])
        prev_edges = self._compute_edges(sample_frames[0])

        for frame in sample_frames[1:]:
            curr_hist = self._compute_histogram(frame)
            curr_edges = self._compute_edges(frame)

            hist_diffs.append(self._histogram_difference(prev_hist, curr_hist))
            edge_diffs.append(self._edge_difference(prev_edges, curr_edges))

            prev_hist = curr_hist
            prev_edges = curr_edges

        # Set thresholds as mean + 2*std to catch significant changes
        if hist_diffs:
            self.histogram_threshold = np.mean(hist_diffs) + 2 * np.std(hist_diffs)
        else:
            self.histogram_threshold = 0.3

        if edge_diffs:
            self.edge_threshold = np.mean(edge_diffs) + 2 * np.std(edge_diffs)
        else:
            self.edge_threshold = 0.2

        logger.info(f"Adaptive thresholds - Histogram: {self.histogram_threshold:.3f}, "
                   f"Edge: {self.edge_threshold:.3f}")

    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute color histogram for frame"""
        # Convert to HSV for better color representation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Compute histogram for each channel
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [8], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [8], [0, 256])

        # Normalize and concatenate
        hist_h = cv2.normalize(hist_h, hist_h).flatten()
        hist_s = cv2.normalize(hist_s, hist_s).flatten()
        hist_v = cv2.normalize(hist_v, hist_v).flatten()

        return np.concatenate([hist_h, hist_s, hist_v])

    def _compute_edges(self, frame: np.ndarray) -> np.ndarray:
        """Compute edge map for frame"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)

        # Compute histogram of edge pixels
        hist_edges = cv2.calcHist([edges], [0], None, [32], [0, 256])
        hist_edges = cv2.normalize(hist_edges, hist_edges).flatten()

        return hist_edges

    def _histogram_difference(self, hist_a: np.ndarray, hist_b: np.ndarray) -> float:
        """Compute histogram difference using Bhattacharyya distance"""
        return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_BHATTACHARYYA)

    def _edge_difference(self, edges_a: np.ndarray, edges_b: np.ndarray) -> float:
        """Compute edge difference"""
        return cv2.compareHist(edges_a, edges_b, cv2.HISTCMP_BHATTACHARYYA)

    def _detect_shot_change(self, frame_a: np.ndarray, frame_b: np.ndarray,
                           hist_a: np.ndarray, hist_b: np.ndarray,
                           edges_a: np.ndarray, edges_b: np.ndarray,
                           frame_number: int) -> Optional[ShotBoundary]:
        """Detect shot change and classify boundary type"""
        hist_diff = self._histogram_difference(hist_a, hist_b)
        edge_diff = self._edge_difference(edges_a, edges_b)

        # Check if difference exceeds threshold
        hist_trigger = hist_diff > self.histogram_threshold
        edge_trigger = edge_diff > self.edge_threshold

        if not (hist_trigger or edge_trigger):
            return None

        # Classify boundary type
        if hist_trigger and hist_diff > self.histogram_threshold * 1.5:
            boundary_type = "cut"
            confidence = min(1.0, hist_diff / (self.histogram_threshold * 2))
        elif edge_trigger:
            boundary_type = "fade"  # Gradual transitions affect edges more
            confidence = min(1.0, edge_diff / (self.edge_threshold * 2))
        else:
            boundary_type = "dissolve"
            confidence = min(1.0, (hist_diff + edge_diff) / 2 / self.histogram_threshold)

        # Estimate timestamp (assuming 30fps if not available)
        timestamp = frame_number / 30.0  # This should be passed from video metadata

        return ShotBoundary(
            frame_number=frame_number,
            timestamp=timestamp,
            boundary_type=boundary_type,
            confidence=confidence
        )