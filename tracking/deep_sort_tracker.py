"""
Multi-object person tracker using DeepSORT with IoU-based fallback.

Tracks persons across frames, maintaining associations between
person bounding boxes, head bounding boxes, face bounding boxes,
embeddings, and classification results.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import deep_sort_realtime
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except (ImportError, Exception) as e:
    DEEPSORT_AVAILABLE = False
    logger.warning(f"deep_sort_realtime not available, using IoU-based fallback tracker: {e}")


@dataclass
class TrackedPerson:
    """A tracked person with associated bounding boxes, embeddings, and metadata."""
    track_id: int
    person_bbox: Optional[Tuple[int, int, int, int]] = None   # (x1, y1, x2, y2)
    head_bbox: Optional[Tuple[int, int, int, int]] = None      # (x1, y1, x2, y2)
    face_bbox: Optional[Tuple[int, int, int, int]] = None      # (x1, y1, x2, y2)
    confidence: float = 0.0
    face_embedding: Optional[np.ndarray] = None                 # 512-d face embedding
    body_embedding: Optional[np.ndarray] = None                 # body appearance embedding
    keypoints: Optional[np.ndarray] = None                      # pose keypoints
    is_confirmed: bool = False
    classification: Optional[Any] = None                        # PersonClassification


class _SimpleTrack:
    """A simple track for the IoU-based fallback tracker."""

    _next_id = 1

    def __init__(self, detection: Dict[str, Any], n_init: int):
        self.track_id: int = _SimpleTrack._next_id
        _SimpleTrack._next_id += 1

        self.person_bbox: Optional[Tuple[int, int, int, int]] = detection.get('person_bbox')
        self.head_bbox: Optional[Tuple[int, int, int, int]] = detection.get('head_bbox')
        self.face_bbox: Optional[Tuple[int, int, int, int]] = detection.get('face_bbox')
        self.confidence: float = detection.get('confidence', 0.0)
        self.face_embedding: Optional[np.ndarray] = detection.get('face_embedding')
        self.body_embedding: Optional[np.ndarray] = detection.get('body_embedding')
        self.keypoints: Optional[np.ndarray] = detection.get('keypoints')
        self.classification: Optional[Any] = detection.get('classification')

        self.hits: int = 1
        self.age: int = 0
        self.time_since_update: int = 0
        self.n_init: int = n_init

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= self.n_init

    def predict(self) -> None:
        """Advance internal age counter (no motion model in simple tracker)."""
        self.age += 1
        self.time_since_update += 1

    def update(self, detection: Dict[str, Any]) -> None:
        """Update this track with a matched detection."""
        self.person_bbox = detection.get('person_bbox', self.person_bbox)
        self.head_bbox = detection.get('head_bbox', self.head_bbox)
        self.face_bbox = detection.get('face_bbox', self.face_bbox)
        self.confidence = detection.get('confidence', self.confidence)

        if detection.get('face_embedding') is not None:
            self.face_embedding = detection['face_embedding']
        if detection.get('body_embedding') is not None:
            self.body_embedding = detection['body_embedding']
        if detection.get('keypoints') is not None:
            self.keypoints = detection['keypoints']
        if detection.get('classification') is not None:
            self.classification = detection['classification']

        self.hits += 1
        self.time_since_update = 0

    def to_tracked_person(self) -> TrackedPerson:
        """Convert to TrackedPerson dataclass."""
        return TrackedPerson(
            track_id=self.track_id,
            person_bbox=self.person_bbox,
            head_bbox=self.head_bbox,
            face_bbox=self.face_bbox,
            confidence=self.confidence,
            face_embedding=self.face_embedding,
            body_embedding=self.body_embedding,
            keypoints=self.keypoints,
            is_confirmed=self.is_confirmed,
            classification=self.classification,
        )


class _SimpleIoUTracker:
    """
    Fallback tracker using IoU-based association when DeepSORT is not available.

    Uses a greedy matching strategy based on bounding box IoU to associate
    detections with existing tracks across frames.
    """

    def __init__(self, max_age: int = 90, n_init: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.n_init = n_init
        self.iou_threshold = iou_threshold
        self.tracks: List[_SimpleTrack] = []

        # Reset the ID counter for fresh tracker instances
        _SimpleTrack._next_id = 1

    def update(self, detections: List[Dict[str, Any]]) -> List[_SimpleTrack]:
        """
        Update tracks with new detections.

        Args:
            detections: List of detection dicts, each containing at minimum
                        'person_bbox' with (x1, y1, x2, y2).

        Returns:
            List of active _SimpleTrack objects.
        """
        # Predict step: advance all tracks
        for track in self.tracks:
            track.predict()

        if not detections:
            # Remove dead tracks
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return [t for t in self.tracks if t.is_confirmed]

        # Build IoU cost matrix between existing tracks and new detections
        matched_tracks, matched_dets, unmatched_tracks, unmatched_dets = \
            self._match(detections)

        # Update matched tracks
        for track_idx, det_idx in zip(matched_tracks, matched_dets):
            self.tracks[track_idx].update(detections[det_idx])

        # Create new tracks from unmatched detections
        for det_idx in unmatched_dets:
            new_track = _SimpleTrack(detections[det_idx], n_init=self.n_init)
            self.tracks.append(new_track)

        # Remove dead tracks (exceeded max_age without update)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]

        # Return confirmed tracks
        return [t for t in self.tracks if t.is_confirmed]

    def _match(self, detections: List[Dict[str, Any]]):
        """
        Greedy IoU matching between existing tracks and new detections.

        Returns:
            matched_tracks: indices of matched tracks
            matched_dets: indices of matched detections
            unmatched_tracks: indices of unmatched tracks
            unmatched_dets: indices of unmatched detections
        """
        num_tracks = len(self.tracks)
        num_dets = len(detections)

        if num_tracks == 0:
            return [], [], [], list(range(num_dets))

        if num_dets == 0:
            return [], [], list(range(num_tracks)), []

        # Compute IoU matrix
        iou_matrix = np.zeros((num_tracks, num_dets), dtype=np.float32)
        for t_idx, track in enumerate(self.tracks):
            if track.person_bbox is None:
                continue
            for d_idx, det in enumerate(detections):
                det_bbox = det.get('person_bbox')
                if det_bbox is None:
                    continue
                iou_matrix[t_idx, d_idx] = self._compute_iou(track.person_bbox, det_bbox)

        # Greedy matching: pick highest IoU pairs
        matched_tracks = []
        matched_dets = []
        used_tracks = set()
        used_dets = set()

        # Flatten and sort by IoU descending
        if iou_matrix.size > 0:
            flat_indices = np.argsort(-iou_matrix.ravel())
            for flat_idx in flat_indices:
                t_idx = int(flat_idx // num_dets)
                d_idx = int(flat_idx % num_dets)

                if t_idx in used_tracks or d_idx in used_dets:
                    continue

                if iou_matrix[t_idx, d_idx] < self.iou_threshold:
                    break  # All remaining IoUs are below threshold

                matched_tracks.append(t_idx)
                matched_dets.append(d_idx)
                used_tracks.add(t_idx)
                used_dets.add(d_idx)

        unmatched_tracks = [i for i in range(num_tracks) if i not in used_tracks]
        unmatched_dets = [i for i in range(num_dets) if i not in used_dets]

        return matched_tracks, matched_dets, unmatched_tracks, unmatched_dets

    @staticmethod
    def _compute_iou(bbox_a: Tuple[int, int, int, int],
                     bbox_b: Tuple[int, int, int, int]) -> float:
        """Compute Intersection over Union between two bounding boxes."""
        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b

        x1_inter = max(x1_a, x1_b)
        y1_inter = max(y1_a, y1_b)
        x2_inter = min(x2_a, x2_b)
        y2_inter = min(y2_a, y2_b)

        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0

        inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = area_a + area_b - inter_area

        if union_area <= 0:
            return 0.0

        return inter_area / union_area

    def reset(self) -> None:
        """Clear all tracks and reset ID counter."""
        self.tracks.clear()
        _SimpleTrack._next_id = 1


class PersonTracker:
    """
    Multi-object person tracker.

    Wraps DeepSORT when available, falling back to a simple IoU-based tracker
    otherwise. Maintains associations between person, head, and face bounding
    boxes along with embeddings and classification data.
    """

    def __init__(self, max_age: int = 90, n_init: int = 3,
                 max_cosine_distance: float = 0.3, nn_budget: int = 150):
        """
        Initialize the person tracker.

        Args:
            max_age: Maximum number of frames a track is kept without updates
                     before deletion.
            n_init: Number of consecutive detections before a track is confirmed.
            max_cosine_distance: Maximum cosine distance for DeepSORT matching.
            nn_budget: Maximum number of feature samples stored per track in
                       DeepSORT's nearest-neighbor gallery.
        """
        self.max_age = max_age
        self.n_init = n_init
        self.max_cosine_distance = max_cosine_distance
        self.nn_budget = nn_budget

        # Store config for re-initialization (used by backward pass)
        self.config = {
            'max_age': max_age,
            'n_init': n_init,
            'max_cosine_distance': max_cosine_distance,
            'nn_budget': nn_budget,
        }

        # Internal state: map DeepSORT track IDs to auxiliary data
        self._aux_data: Dict[int, Dict[str, Any]] = {}

        # Active tracked persons from the last update
        self._active_tracks: List[TrackedPerson] = []

        # Initialize the underlying tracker
        self._tracker = None
        self._use_deepsort = False
        self._initialize_tracker()

    def _initialize_tracker(self) -> None:
        """Initialize the underlying tracking backend."""
        if DEEPSORT_AVAILABLE:
            try:
                self._tracker = DeepSort(
                    max_age=self.max_age,
                    n_init=self.n_init,
                    max_cosine_distance=self.max_cosine_distance,
                    nn_budget=self.nn_budget,
                )
                self._use_deepsort = True
                logger.info("PersonTracker initialized with DeepSORT backend")
            except Exception as e:
                logger.warning(f"Failed to initialize DeepSORT, falling back to IoU tracker: {e}")
                self._tracker = _SimpleIoUTracker(
                    max_age=self.max_age,
                    n_init=self.n_init,
                    iou_threshold=1.0 - self.max_cosine_distance,
                )
                self._use_deepsort = False
        else:
            self._tracker = _SimpleIoUTracker(
                max_age=self.max_age,
                n_init=self.n_init,
                iou_threshold=1.0 - self.max_cosine_distance,
            )
            self._use_deepsort = False
            logger.info("PersonTracker initialized with IoU fallback backend")

    def update(self, frame: np.ndarray,
               detections: List[Any]) -> List[TrackedPerson]:
        """
        Update the tracker with new detections for the current frame.

        Args:
            frame: The current video frame (BGR numpy array).
            detections: List of detection objects or dicts. Each detection should
                        provide at minimum a bounding box. Accepted formats:
                        - dict with keys: 'person_bbox', 'head_bbox', 'face_bbox',
                          'confidence', 'face_embedding', 'body_embedding',
                          'keypoints', 'classification'
                        - object with the same attributes

        Returns:
            List of TrackedPerson for all currently active (confirmed) tracks.
        """
        if frame is None:
            logger.warning("Received None frame, skipping tracker update")
            return self._active_tracks

        # Normalize detections to list of dicts
        normalized = self._normalize_detections(detections)

        if self._use_deepsort:
            tracked = self._update_deepsort(frame, normalized)
        else:
            tracked = self._update_simple(normalized)

        self._active_tracks = tracked
        return tracked

    def _normalize_detections(self, detections: List[Any]) -> List[Dict[str, Any]]:
        """Convert heterogeneous detection inputs into uniform dicts."""
        normalized = []
        for det in detections:
            if isinstance(det, dict):
                d = det
            else:
                d = {
                    'person_bbox': getattr(det, 'person_bbox', None),
                    'head_bbox': getattr(det, 'head_bbox', None),
                    'face_bbox': getattr(det, 'face_bbox',
                                         getattr(det, 'bbox', None)),
                    'confidence': getattr(det, 'confidence', 0.0),
                    'face_embedding': getattr(det, 'face_embedding',
                                              getattr(det, 'embedding', None)),
                    'body_embedding': getattr(det, 'body_embedding', None),
                    'keypoints': getattr(det, 'keypoints', None),
                    'classification': getattr(det, 'classification', None),
                }
            normalized.append(d)
        return normalized

    def _update_deepsort(self, frame: np.ndarray,
                         detections: List[Dict[str, Any]]) -> List[TrackedPerson]:
        """Update using the DeepSORT backend."""
        # DeepSORT expects detections as list of ([x1,y1,w,h], confidence, feature)
        ds_detections = []
        det_aux = {}  # index -> auxiliary data

        for idx, det in enumerate(detections):
            bbox = det.get('person_bbox')
            if bbox is None:
                # Fall back to face bbox if person bbox not available
                bbox = det.get('face_bbox')
            if bbox is None:
                continue

            x1, y1, x2, y2 = bbox
            w = x2 - x1
            h = y2 - y1
            confidence = det.get('confidence', 0.5)

            ds_detections.append(([x1, y1, w, h], confidence, None))
            det_aux[len(ds_detections) - 1] = det

        try:
            tracks = self._tracker.update_tracks(ds_detections, frame=frame)
        except Exception as e:
            logger.error(f"DeepSORT update failed: {e}")
            return self._active_tracks

        tracked_persons = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id

            # Get the bounding box from DeepSORT (x1, y1, x2, y2)
            try:
                ltrb = track.to_ltrb()
                person_bbox = (int(ltrb[0]), int(ltrb[1]),
                               int(ltrb[2]), int(ltrb[3]))
            except Exception:
                person_bbox = None

            # Retrieve auxiliary data if we have it stored
            aux = self._aux_data.get(track_id, {})

            # Try to match this track to a detection for auxiliary info
            best_det_aux = self._match_track_to_detection(person_bbox, detections)
            if best_det_aux:
                aux.update({
                    k: v for k, v in best_det_aux.items()
                    if v is not None and k != 'person_bbox'
                })
                self._aux_data[track_id] = aux

            tp = TrackedPerson(
                track_id=track_id,
                person_bbox=person_bbox,
                head_bbox=aux.get('head_bbox'),
                face_bbox=aux.get('face_bbox'),
                confidence=aux.get('confidence', 0.0),
                face_embedding=aux.get('face_embedding'),
                body_embedding=aux.get('body_embedding'),
                keypoints=aux.get('keypoints'),
                is_confirmed=True,
                classification=aux.get('classification'),
            )
            tracked_persons.append(tp)

        return tracked_persons

    def _match_track_to_detection(self, track_bbox: Optional[Tuple[int, int, int, int]],
                                  detections: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find the detection that best matches a track's bounding box via IoU."""
        if track_bbox is None or not detections:
            return None

        best_iou = 0.0
        best_det = None

        for det in detections:
            det_bbox = det.get('person_bbox') or det.get('face_bbox')
            if det_bbox is None:
                continue
            iou = _SimpleIoUTracker._compute_iou(track_bbox, det_bbox)
            if iou > best_iou:
                best_iou = iou
                best_det = det

        return best_det if best_iou > 0.2 else None

    def _update_simple(self, detections: List[Dict[str, Any]]) -> List[TrackedPerson]:
        """Update using the simple IoU fallback tracker."""
        active = self._tracker.update(detections)
        return [t.to_tracked_person() for t in active]

    def get_active_tracks(self) -> List[TrackedPerson]:
        """
        Return the list of currently active (confirmed) tracked persons.

        Returns:
            List of TrackedPerson from the most recent update call.
        """
        return list(self._active_tracks)

    def reset(self) -> None:
        """Clear all tracks and reset internal state."""
        self._active_tracks.clear()
        self._aux_data.clear()

        if self._use_deepsort:
            # Re-create the DeepSORT tracker from scratch
            try:
                self._tracker = DeepSort(
                    max_age=self.max_age,
                    n_init=self.n_init,
                    max_cosine_distance=self.max_cosine_distance,
                    nn_budget=self.nn_budget,
                )
            except Exception as e:
                logger.error(f"Failed to reset DeepSORT tracker: {e}")
        else:
            self._tracker.reset()

        logger.debug("PersonTracker reset")
