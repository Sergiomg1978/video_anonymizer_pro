"""
Motion estimation for video analysis
Estimates motion vectors and predicts positions using optical flow and tracking history
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class MotionVector:
    """Motion vector for a tracked person"""
    track_id: int
    velocity_x: float  # pixels per frame
    velocity_y: float
    direction: float   # angle in degrees (0 = right, 90 = down)
    speed: float       # pixels per frame
    confidence: float

@dataclass
class TrackState:
    """State of a track at a specific frame"""
    frame_number: int
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    keypoints: Optional[np.ndarray] = None  # pose keypoints if available

class MotionEstimator:
    """
    Estimates motion of tracked persons using:
    1. Optical flow (Farneback algorithm)
    2. Tracking history (Kalman filter integration)
    3. Pose keypoints displacement
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize motion estimator

        Args:
            use_gpu: Whether to use GPU acceleration for optical flow
        """
        self.use_gpu = use_gpu
        self.prev_frame_gray = None
        self.flow_history = {}  # track_id -> list of motion vectors

    def estimate_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray,
                       tracked_persons: List[object]) -> List[MotionVector]:
        """
        Estimate motion vectors for all tracked persons

        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            tracked_persons: List of TrackedPerson objects

        Returns:
            List of motion vectors for each tracked person
        """
        if prev_frame is None or curr_frame is None:
            return []

        # Convert frames to grayscale for optical flow
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow
        flow = self._compute_optical_flow(prev_gray, curr_gray)

        motion_vectors = []

        for person in tracked_persons:
            if not hasattr(person, 'track_id') or not hasattr(person, 'person_bbox'):
                continue

            track_id = person.track_id
            bbox = person.person_bbox

            # Extract motion vector for this person
            motion_vector = self._extract_person_motion(flow, bbox, track_id)
            if motion_vector:
                motion_vectors.append(motion_vector)

                # Update flow history
                if track_id not in self.flow_history:
                    self.flow_history[track_id] = []
                self.flow_history[track_id].append(motion_vector)

                # Keep only recent history (last 10 frames)
                if len(self.flow_history[track_id]) > 10:
                    self.flow_history[track_id] = self.flow_history[track_id][-10:]

        self.prev_frame_gray = curr_gray
        return motion_vectors

    def predict_next_position(self, track_history: List[TrackState]) -> Tuple[int, int, int, int]:
        """
        Predict the next position of a person based on tracking history

        Args:
            track_history: List of recent TrackState objects

        Returns:
            Predicted bbox (x1, y1, x2, y2)
        """
        if len(track_history) < 2:
            # Not enough history, return last known position
            return track_history[-1].bbox if track_history else (0, 0, 0, 0)

        # Calculate velocity from recent positions
        recent_positions = track_history[-5:]  # Use last 5 positions

        # Extract center points
        centers = []
        for state in recent_positions:
            x1, y1, x2, y2 = state.bbox
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            centers.append((center_x, center_y))

        # Calculate average velocity
        velocities = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            velocities.append((dx, dy))

        if not velocities:
            return track_history[-1].bbox

        # Average velocity
        avg_dx = np.mean([v[0] for v in velocities])
        avg_dy = np.mean([v[1] for v in velocities])

        # Predict next position
        last_bbox = track_history[-1].bbox
        x1, y1, x2, y2 = last_bbox

        # Calculate size change trend
        sizes = [(b[2] - b[0]) * (b[3] - b[1]) for b in [s.bbox for s in recent_positions]]
        size_trend = np.polyfit(range(len(sizes)), sizes, 1)[0] if len(sizes) > 1 else 0

        # Predict new size (with damping)
        curr_size = (x2 - x1) * (y2 - y1)
        new_size = curr_size + size_trend * 0.1  # Dampen size changes
        size_ratio = np.sqrt(new_size / curr_size) if curr_size > 0 else 1.0

        # New center position
        center_x = (x1 + x2) / 2 + avg_dx
        center_y = (y1 + y2) / 2 + avg_dy

        # New bbox dimensions
        half_width = (x2 - x1) / 2 * size_ratio
        half_height = (y2 - y1) / 2 * size_ratio

        new_x1 = int(center_x - half_width)
        new_y1 = int(center_y - half_height)
        new_x2 = int(center_x + half_width)
        new_y2 = int(center_y + half_height)

        return (new_x1, new_y1, new_x2, new_y2)

    def _compute_optical_flow(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> np.ndarray:
        """Compute optical flow between two frames"""
        try:
            # Use Farneback algorithm
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2,
                flags=0
            )
            return flow
        except Exception as e:
            logger.warning(f"Optical flow computation failed: {e}")
            # Return zero flow as fallback
            return np.zeros((prev_gray.shape[0], prev_gray.shape[1], 2), dtype=np.float32)

    def _extract_person_motion(self, flow: np.ndarray, bbox: Tuple[int, int, int, int],
                              track_id: int) -> Optional[MotionVector]:
        """Extract motion vector for a specific person from optical flow"""
        x1, y1, x2, y2 = bbox

        # Ensure bbox is within frame bounds
        h, w = flow.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Extract flow in the person's bounding box
        person_flow = flow[y1:y2, x1:x2]

        if person_flow.size == 0:
            return None

        # Calculate average flow vector
        mean_flow = np.mean(person_flow, axis=(0, 1))

        velocity_x, velocity_y = mean_flow

        # Calculate speed and direction
        speed = np.sqrt(velocity_x**2 + velocity_y**2)
        direction = np.degrees(np.arctan2(velocity_y, velocity_x)) % 360

        # Calculate confidence based on flow consistency
        flow_std = np.std(person_flow, axis=(0, 1))
        flow_consistency = 1.0 / (1.0 + np.mean(flow_std))  # Higher consistency = higher confidence

        # Also consider history consistency if available
        history_confidence = 1.0
        if track_id in self.flow_history and len(self.flow_history[track_id]) >= 3:
            recent_speeds = [mv.speed for mv in self.flow_history[track_id][-3:]]
            speed_std = np.std(recent_speeds)
            history_confidence = 1.0 / (1.0 + speed_std / max(1.0, np.mean(recent_speeds)))

        confidence = min(1.0, flow_consistency * history_confidence)

        return MotionVector(
            track_id=track_id,
            velocity_x=float(velocity_x),
            velocity_y=float(velocity_y),
            direction=float(direction),
            speed=float(speed),
            confidence=float(confidence)
        )

    def get_motion_history(self, track_id: int) -> List[MotionVector]:
        """Get motion history for a specific track"""
        return self.flow_history.get(track_id, [])

    def reset_track_history(self, track_id: int):
        """Reset motion history for a track (useful after re-identification)"""
        if track_id in self.flow_history:
            del self.flow_history[track_id]

    def smooth_motion_trajectory(self, track_history: List[TrackState],
                                window_size: int = 5) -> List[Tuple[int, int, int, int]]:
        """
        Smooth the motion trajectory using moving average

        Args:
            track_history: List of TrackState objects
            window_size: Size of smoothing window

        Returns:
            List of smoothed bboxes
        """
        if len(track_history) < window_size:
            return [state.bbox for state in track_history]

        smoothed_bboxes = []

        for i in range(len(track_history)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(track_history), i + window_size // 2 + 1)

            window_bboxes = [track_history[j].bbox for j in range(start_idx, end_idx)]

            # Average the bboxes in the window
            avg_bbox = self._average_bboxes(window_bboxes)
            smoothed_bboxes.append(avg_bbox)

        return smoothed_bboxes

    def _average_bboxes(self, bboxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
        """Compute average of multiple bounding boxes"""
        if not bboxes:
            return (0, 0, 0, 0)

        x1s = [b[0] for b in bboxes]
        y1s = [b[1] for b in bboxes]
        x2s = [b[2] for b in bboxes]
        y2s = [b[3] for b in bboxes]

        return (
            int(np.mean(x1s)),
            int(np.mean(y1s)),
            int(np.mean(x2s)),
            int(np.mean(y2s))
        )