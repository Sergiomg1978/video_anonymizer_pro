"""
Re-identification module for recovering lost tracks after occlusion.

Combines multiple signals -- face embedding, body embedding, color histogram,
and spatial proximity -- to re-associate a newly detected person with a
previously known identity.
"""

import numpy as np
from typing import Dict, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class ReIdentifier:
    """
    Re-identifies persons after tracking loss (e.g. occlusion, temporary exit).

    Uses a weighted combination of four signals:
      1. Face embedding cosine similarity  (weight: 0.40)
      2. Body embedding cosine similarity  (weight: 0.25)
      3. Color histogram similarity        (weight: 0.20)
      4. Spatial proximity                 (weight: 0.15)
    """

    # Signal weights (must sum to 1.0)
    WEIGHT_FACE = 0.40
    WEIGHT_BODY = 0.25
    WEIGHT_COLOR = 0.20
    WEIGHT_SPATIAL = 0.15

    def __init__(self, embedding_threshold: float = 0.5,
                 body_threshold: float = 0.6,
                 max_time_gap: int = 90):
        """
        Initialize the re-identifier.

        Args:
            embedding_threshold: Minimum combined similarity score required
                                 to accept a re-identification match.
            body_threshold: Minimum body embedding similarity to consider
                            a match when face embedding is unavailable.
            max_time_gap: Maximum number of frames between last sighting
                          and current frame for a re-identification to
                          be attempted.
        """
        self.embedding_threshold = embedding_threshold
        self.body_threshold = body_threshold
        self.max_time_gap = max_time_gap

    def try_reidentify(self, features: Dict[str, Any],
                       identity_manager: Any) -> Optional[str]:
        """
        Attempt to re-identify a person by matching against known identities.

        Args:
            features: Feature dict for the current detection:
                      'face_embedding'   (np.ndarray or None)
                      'body_embedding'   (np.ndarray or None)
                      'color_histogram'  (np.ndarray or None)
                      'last_position'    (tuple(x1,y1,x2,y2) or None)
                      'frame_number'     (int or None)
            identity_manager: An IdentityManager instance providing
                              get_all_identities().

        Returns:
            The identity_id of the best match, or None if no match is found.
        """
        identities = identity_manager.get_all_identities()
        if not identities:
            return None

        current_frame = features.get('frame_number', -1)

        best_identity_id = None
        best_score = -1.0

        for identity_id, identity in identities.items():
            # Enforce maximum time gap
            if current_frame >= 0 and identity.last_frame >= 0:
                gap = abs(current_frame - identity.last_frame)
                if gap > self.max_time_gap:
                    continue

            # Build a features-dict from the stored identity
            identity_features = self._identity_to_features(identity)

            score = self.compute_similarity(features, identity_features)

            if score > best_score:
                best_score = score
                best_identity_id = identity_id

        if best_score < self.embedding_threshold:
            # If face is unavailable, try body-only with a separate threshold
            if features.get('face_embedding') is None:
                body_score = self._body_only_match(features, identities)
                if body_score is not None:
                    return body_score
            return None

        logger.debug(
            f"Re-identified as {best_identity_id} "
            f"with combined score {best_score:.3f}"
        )
        return best_identity_id

    def compute_similarity(self, features_a: Dict[str, Any],
                           features_b: Dict[str, Any]) -> float:
        """
        Compute a multi-signal weighted similarity between two feature sets.

        Signals:
          1. Face embedding cosine similarity
          2. Body embedding cosine similarity
          3. Color histogram correlation
          4. Spatial proximity (based on bounding box IoU / distance)

        Each signal is weighted and combined. Missing signals are excluded
        and the remaining weights are re-normalized.

        Args:
            features_a: First feature set dict.
            features_b: Second feature set dict.

        Returns:
            Combined similarity score in [0, 1].
        """
        scores = []
        weights = []

        # 1. Face embedding similarity
        face_sim = self._face_similarity(
            features_a.get('face_embedding'),
            features_b.get('face_embedding'),
        )
        if face_sim is not None:
            scores.append(face_sim)
            weights.append(self.WEIGHT_FACE)

        # 2. Body embedding similarity
        body_sim = self._body_similarity(
            features_a.get('body_embedding'),
            features_b.get('body_embedding'),
        )
        if body_sim is not None:
            scores.append(body_sim)
            weights.append(self.WEIGHT_BODY)

        # 3. Color histogram similarity
        color_sim = self._color_histogram_similarity(
            features_a.get('color_histogram'),
            features_b.get('color_histogram'),
        )
        if color_sim is not None:
            scores.append(color_sim)
            weights.append(self.WEIGHT_COLOR)

        # 4. Spatial proximity
        spatial_sim = self._spatial_similarity(
            features_a.get('last_position'),
            features_b.get('last_position'),
        )
        if spatial_sim is not None:
            scores.append(spatial_sim)
            weights.append(self.WEIGHT_SPATIAL)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        combined = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return float(np.clip(combined, 0.0, 1.0))

    # ------------------------------------------------------------------
    # Individual signal computations
    # ------------------------------------------------------------------

    def _face_similarity(self, emb_a: Optional[np.ndarray],
                         emb_b: Optional[np.ndarray]) -> Optional[float]:
        """Cosine similarity between two face embeddings."""
        if emb_a is None or emb_b is None:
            return None
        return self._cosine_similarity(emb_a, emb_b)

    def _body_similarity(self, emb_a: Optional[np.ndarray],
                         emb_b: Optional[np.ndarray]) -> Optional[float]:
        """Cosine similarity between two body embeddings."""
        if emb_a is None or emb_b is None:
            return None
        return self._cosine_similarity(emb_a, emb_b)

    def _color_histogram_similarity(
        self, hist_a: Optional[np.ndarray],
        hist_b: Optional[np.ndarray],
    ) -> Optional[float]:
        """
        Compute similarity between two color histograms using correlation.

        Uses the Bhattacharyya-like histogram intersection metric, which
        is robust to lighting changes.
        """
        if hist_a is None or hist_b is None:
            return None

        hist_a = np.asarray(hist_a, dtype=np.float32).ravel()
        hist_b = np.asarray(hist_b, dtype=np.float32).ravel()

        if hist_a.shape != hist_b.shape:
            return None

        # Normalize histograms
        sum_a = hist_a.sum()
        sum_b = hist_b.sum()
        if sum_a < 1e-10 or sum_b < 1e-10:
            return None

        hist_a = hist_a / sum_a
        hist_b = hist_b / sum_b

        # Histogram intersection (ranges from 0 to 1)
        intersection = float(np.minimum(hist_a, hist_b).sum())
        return intersection

    def _spatial_similarity(
        self,
        bbox_a: Optional[Tuple[int, int, int, int]],
        bbox_b: Optional[Tuple[int, int, int, int]],
    ) -> Optional[float]:
        """
        Compute spatial proximity between two bounding boxes.

        Uses a combination of IoU and center-distance normalized by the
        average bounding box diagonal. Returns a score in [0, 1].
        """
        if bbox_a is None or bbox_b is None:
            return None

        x1_a, y1_a, x2_a, y2_a = bbox_a
        x1_b, y1_b, x2_b, y2_b = bbox_b

        # Center distance
        cx_a = (x1_a + x2_a) / 2.0
        cy_a = (y1_a + y2_a) / 2.0
        cx_b = (x1_b + x2_b) / 2.0
        cy_b = (y1_b + y2_b) / 2.0

        center_dist = np.sqrt((cx_a - cx_b) ** 2 + (cy_a - cy_b) ** 2)

        # Average diagonal (for normalization)
        diag_a = np.sqrt((x2_a - x1_a) ** 2 + (y2_a - y1_a) ** 2)
        diag_b = np.sqrt((x2_b - x1_b) ** 2 + (y2_b - y1_b) ** 2)
        avg_diag = (diag_a + diag_b) / 2.0

        if avg_diag < 1e-10:
            return 0.0

        # Distance-based similarity: exponential decay
        # A center distance equal to the average diagonal gives ~0.37 similarity
        distance_sim = float(np.exp(-center_dist / avg_diag))

        # IoU component
        iou = self._compute_iou(bbox_a, bbox_b)

        # Combine: IoU is more informative when boxes overlap; distance
        # is more informative when they don't.
        if iou > 0:
            return 0.6 * iou + 0.4 * distance_sim
        else:
            return distance_sim

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Returns:
            Similarity in [0, 1] (negative similarities clamped to 0).
        """
        a = np.asarray(a, dtype=np.float32).ravel()
        b = np.asarray(b, dtype=np.float32).ravel()

        if a.shape != b.shape:
            logger.warning(
                f"Embedding shape mismatch: {a.shape} vs {b.shape}"
            )
            return 0.0

        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)

        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0

        sim = float(np.dot(a, b) / (norm_a * norm_b))
        return max(0.0, sim)

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

    @staticmethod
    def _identity_to_features(identity: Any) -> Dict[str, Any]:
        """
        Convert an Identity object to a features dict suitable for similarity
        comparison.

        Uses the most recent embedding from each gallery as the representative.
        """
        features: Dict[str, Any] = {
            'face_embedding': None,
            'body_embedding': None,
            'color_histogram': None,
            'last_position': identity.last_bbox,
        }

        if identity.face_embeddings:
            features['face_embedding'] = identity.face_embeddings[-1]

        if identity.body_embeddings:
            features['body_embedding'] = identity.body_embeddings[-1]

        if identity.color_histograms:
            features['color_histogram'] = identity.color_histograms[-1]

        return features

    def _body_only_match(self, features: Dict[str, Any],
                         identities: Dict[str, Any]) -> Optional[str]:
        """
        Fallback matching using only body embedding when face is unavailable.

        Args:
            features: Current detection features.
            identities: Dict of identity_id -> Identity.

        Returns:
            Best matching identity_id, or None.
        """
        body_emb = features.get('body_embedding')
        if body_emb is None:
            return None

        best_id = None
        best_sim = -1.0

        for identity_id, identity in identities.items():
            if not identity.body_embeddings:
                continue

            # Compare against the most recent body embedding
            sim = self._cosine_similarity(body_emb, identity.body_embeddings[-1])
            if sim > best_sim:
                best_sim = sim
                best_id = identity_id

        if best_sim >= self.body_threshold:
            logger.debug(
                f"Body-only re-identification matched {best_id} "
                f"with similarity {best_sim:.3f}"
            )
            return best_id

        return None
