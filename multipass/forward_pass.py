"""
Forward pass processing for video anonymization.
Processes video from start to end, performing detection, tracking, and classification.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from .types import FrameResult

class ForwardPass:
    """
    Processes video in forward direction (frame 0 to N).
    Performs detection, tracking, classification, and scene analysis.
    """

    def __init__(self, detector: Any, tracker: Any, classifier: Any,
                 scene_analyzer: Any, identity_manager: Any, anchor_frames: List[Any]):
        """
        Initialize forward pass processor.

        Args:
            detector: Person/face/head detector
            tracker: Multi-object tracker
            classifier: Adult/child classifier
            scene_analyzer: Scene analysis module
            identity_manager: Identity management system
            anchor_frames: Pre-annotated reference frames
        """
        self.detector = detector
        self.tracker = tracker
        self.classifier = classifier
        self.scene_analyzer = scene_analyzer
        self.identity_manager = identity_manager
        self.anchor_frames = anchor_frames

    def process(self, video_reader: Any) -> List[FrameResult]:
        """
        Process video from start to end.

        Args:
            video_reader: Video reader object with iterate_frames() method

        Returns:
            List of FrameResult for each frame
        """
        results = []
        frame_number = 0

        # Initialize with anchor frames if available
        if self.anchor_frames:
            self._initialize_from_anchors()

        for frame_number, (frame_num, frame) in enumerate(video_reader.iterate_frames()):
            start_time = time.time()

            # 1. Detection (face + head + person)
            detections = self.detector.detect(frame)

            # 2. Tracking update
            tracks = self.tracker.update(frame, detections)

            # 3. Classification (adult/niño)
            classifications = {}
            for track in tracks:
                if track.person_bbox:
                    classification = self.classifier.classify(
                        frame, track.person_bbox, track.face_bbox, track.keypoints
                    )
                    classifications[track.track_id] = classification

            # 4. Update identity manager and check for re-identification
            for track in tracks:
                if track.track_id not in self.identity_manager.get_all_identities():
                    # Try to match with existing identities
                    features = self._extract_features(track, frame)
                    identity_id = self.identity_manager.match_identity(features)
                    if identity_id is None:
                        # Register new identity
                        self.identity_manager.register_identity(track.track_id, features)
                    else:
                        # Re-identify
                        self.identity_manager.update_identity(identity_id, features)

            # 5. Scene analysis (periodically or when confidence is low)
            scene_context = None
            if frame_number % 30 == 0 or self._should_analyze_scene(tracks):  # Every second at 30fps
                scene_context = self.scene_analyzer.analyze_frame(frame, detections, tracks)

            # 6. Find woman head bbox
            woman_head_bbox, woman_confidence = self._find_woman_head(tracks, classifications)

            processing_time = time.time() - start_time

            result = FrameResult(
                frame_number=frame_num,
                timestamp=frame_num / video_reader.get_fps(),
                detections=detections,
                tracks=tracks,
                classifications=classifications,
                woman_head_bbox=woman_head_bbox,
                woman_confidence=woman_confidence,
                scene_context=scene_context,
                processing_time=processing_time
            )

            results.append(result)

        return results

    def _initialize_from_anchors(self):
        """Initialize identity manager with anchor frame embeddings."""
        for anchor in self.anchor_frames:
            if anchor.face_embedding is not None:
                features = {
                    'face_embedding': anchor.face_embedding,
                    'body_embedding': anchor.body_embedding,
                    'last_position': anchor.head_bbox,
                    'color_histogram': None  # Could extract from frame
                }
                self.identity_manager.register_identity(anchor.frame_number, features)

    def _extract_features(self, track: Any, frame: Any) -> Dict[str, Any]:
        """Extract features for identity matching."""
        features = {
            'face_embedding': getattr(track, 'face_embedding', None),
            'body_embedding': getattr(track, 'body_embedding', None),
            'last_position': track.person_bbox,
            'color_histogram': self._extract_color_histogram(frame, track.person_bbox) if track.person_bbox else None
        }
        return features

    def _extract_color_histogram(self, frame: Any, bbox: Tuple[int, int, int, int]) -> Any:
        """Extract color histogram from person region."""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        # Simple histogram extraction (placeholder)
        hist = np.histogram(roi.flatten(), bins=32, range=(0, 256))[0]
        return hist.astype(np.float32) / hist.sum()

    def _should_analyze_scene(self, tracks: List[Any]) -> bool:
        """Determine if scene analysis is needed."""
        # Analyze if confidence is low or scene changes detected
        low_confidence_tracks = [t for t in tracks if t.confidence < 0.5]
        return len(low_confidence_tracks) > 0

    def _find_woman_head(self, tracks: List[Any], classifications: Dict[int, Any]) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Find the woman's head bbox with highest confidence."""
        best_bbox = None
        best_confidence = 0.0

        for track in tracks:
            if track.track_id in classifications:
                classification = classifications[track.track_id]
                if classification.is_adult and classification.is_female:
                    # Check if this is the target identity
                    identity_id = self.identity_manager.get_target_identity()
                    if identity_id and self.identity_manager.get_identity(track.track_id) == identity_id:
                        confidence = min(track.confidence, classification.confidence)
                        if confidence > best_confidence:
                            best_bbox = track.head_bbox
                            best_confidence = confidence

        return best_bbox, best_confidence