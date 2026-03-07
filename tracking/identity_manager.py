"""
Persistent identity management for video anonymization.

Maintains a registry of identities across frames, using face embeddings,
body embeddings, and color histograms to match and track individuals
even after occlusion or tracking loss.
"""

import uuid
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


@dataclass
class Identity:
    """Persistent identity of a person across frames."""
    identity_id: str
    face_embeddings: List[np.ndarray] = field(default_factory=list)
    body_embeddings: List[np.ndarray] = field(default_factory=list)
    color_histograms: List[np.ndarray] = field(default_factory=list)
    last_bbox: Optional[Tuple[int, int, int, int]] = None
    last_frame: int = -1
    is_target: bool = False


class IdentityManager:
    """
    Manages persistent identities across video frames.

    Associates tracker track IDs with stable identity IDs and provides
    methods to register, match, and update identities using cosine
    similarity on face/body embeddings.
    """

    # Maximum number of stored embeddings per identity to limit memory use
    _MAX_EMBEDDINGS = 20

    def __init__(self, similarity_threshold: float = 0.5):
        """
        Initialize the identity manager.

        Args:
            similarity_threshold: Minimum cosine similarity required to
                                  consider two feature sets as the same identity.
        """
        self.similarity_threshold = similarity_threshold

        # Store config for re-initialization (used by backward pass)
        self.config = {
            'similarity_threshold': similarity_threshold,
        }

        # identity_id -> Identity
        self._identities: Dict[str, Identity] = {}

        # track_id -> identity_id (maps current tracker track IDs to stable identities)
        self._track_to_identity: Dict[int, str] = {}

        # The identity_id of the target person (e.g. the woman to anonymize)
        self._target_identity_id: Optional[str] = None

    def register_identity(self, track_id: int,
                          features: Dict[str, Any]) -> str:
        """
        Register a new identity from a track's features.

        Args:
            track_id: The tracker-assigned track ID.
            features: Dict with optional keys:
                      'face_embedding' (np.ndarray or None),
                      'body_embedding' (np.ndarray or None),
                      'color_histogram' (np.ndarray or None),
                      'last_position' (tuple or None),
                      'frame_number' (int or None).

        Returns:
            The newly created identity_id string.
        """
        identity_id = f"id_{uuid.uuid4().hex[:8]}"

        identity = Identity(identity_id=identity_id)

        # Store embeddings
        face_emb = features.get('face_embedding')
        if face_emb is not None:
            identity.face_embeddings.append(np.asarray(face_emb, dtype=np.float32))

        body_emb = features.get('body_embedding')
        if body_emb is not None:
            identity.body_embeddings.append(np.asarray(body_emb, dtype=np.float32))

        color_hist = features.get('color_histogram')
        if color_hist is not None:
            identity.color_histograms.append(np.asarray(color_hist, dtype=np.float32))

        identity.last_bbox = features.get('last_position')
        identity.last_frame = features.get('frame_number', -1)

        self._identities[identity_id] = identity
        self._track_to_identity[track_id] = identity_id

        logger.debug(f"Registered new identity {identity_id} for track {track_id}")
        return identity_id

    def match_identity(self, features: Dict[str, Any]) -> Optional[str]:
        """
        Try to match features against all existing identities.

        Args:
            features: Dict with optional keys:
                      'face_embedding', 'body_embedding',
                      'color_histogram', 'last_position'.

        Returns:
            The identity_id of the best matching identity, or None if no
            match exceeds the similarity threshold.
        """
        if not self._identities:
            return None

        face_emb = features.get('face_embedding')
        body_emb = features.get('body_embedding')

        best_identity_id = None
        best_similarity = -1.0

        for identity_id, identity in self._identities.items():
            similarity = self._compute_identity_similarity(
                face_emb, body_emb, identity
            )
            if similarity > best_similarity:
                best_similarity = similarity
                best_identity_id = identity_id

        if best_similarity >= self.similarity_threshold:
            logger.debug(
                f"Matched identity {best_identity_id} "
                f"with similarity {best_similarity:.3f}"
            )
            return best_identity_id

        return None

    def update_identity(self, identity_id: str,
                        features: Dict[str, Any]) -> None:
        """
        Update an existing identity with new feature observations.

        Args:
            identity_id: The identity to update.
            features: Dict with optional keys:
                      'face_embedding', 'body_embedding',
                      'color_histogram', 'last_position', 'frame_number'.
        """
        identity = self._identities.get(identity_id)
        if identity is None:
            logger.warning(f"Cannot update unknown identity: {identity_id}")
            return

        # Append new embeddings (with cap to prevent unbounded growth)
        face_emb = features.get('face_embedding')
        if face_emb is not None:
            identity.face_embeddings.append(np.asarray(face_emb, dtype=np.float32))
            if len(identity.face_embeddings) > self._MAX_EMBEDDINGS:
                identity.face_embeddings = identity.face_embeddings[-self._MAX_EMBEDDINGS:]

        body_emb = features.get('body_embedding')
        if body_emb is not None:
            identity.body_embeddings.append(np.asarray(body_emb, dtype=np.float32))
            if len(identity.body_embeddings) > self._MAX_EMBEDDINGS:
                identity.body_embeddings = identity.body_embeddings[-self._MAX_EMBEDDINGS:]

        color_hist = features.get('color_histogram')
        if color_hist is not None:
            identity.color_histograms.append(np.asarray(color_hist, dtype=np.float32))
            if len(identity.color_histograms) > self._MAX_EMBEDDINGS:
                identity.color_histograms = identity.color_histograms[-self._MAX_EMBEDDINGS:]

        if features.get('last_position') is not None:
            identity.last_bbox = features['last_position']

        if features.get('frame_number') is not None:
            identity.last_frame = features['frame_number']

    def get_identity(self, track_id: int) -> Optional[str]:
        """
        Get the identity_id associated with a tracker track ID.

        Args:
            track_id: The tracker-assigned track ID.

        Returns:
            The identity_id string, or None if not mapped.
        """
        return self._track_to_identity.get(track_id)

    def get_all_identities(self) -> Dict[str, Identity]:
        """
        Return all registered identities.

        Returns:
            Dict mapping identity_id to Identity dataclass.
        """
        return dict(self._identities)

    def set_target_identity(self, identity_id: str) -> None:
        """
        Mark an identity as the target person (the woman to protect/anonymize).

        Args:
            identity_id: The identity to mark as the target.

        Raises:
            ValueError: If the identity_id does not exist.
        """
        if identity_id not in self._identities:
            raise ValueError(f"Unknown identity: {identity_id}")

        # Clear previous target
        if self._target_identity_id is not None:
            prev = self._identities.get(self._target_identity_id)
            if prev is not None:
                prev.is_target = False

        self._target_identity_id = identity_id
        self._identities[identity_id].is_target = True
        logger.info(f"Set target identity to {identity_id}")

    def get_target_identity(self) -> Optional[str]:
        """
        Get the identity_id of the target person.

        Returns:
            The target identity_id, or None if not set.
        """
        return self._target_identity_id

    def link_track_to_identity(self, track_id: int, identity_id: str) -> None:
        """
        Explicitly associate a track ID with an identity.

        Args:
            track_id: The tracker-assigned track ID.
            identity_id: The identity_id to link.

        Raises:
            ValueError: If the identity_id does not exist.
        """
        if identity_id not in self._identities:
            raise ValueError(f"Unknown identity: {identity_id}")
        self._track_to_identity[track_id] = identity_id

    def unlink_track(self, track_id: int) -> None:
        """Remove the track-to-identity mapping for a given track."""
        self._track_to_identity.pop(track_id, None)

    # ------------------------------------------------------------------
    # Internal similarity computation
    # ------------------------------------------------------------------

    def _compute_identity_similarity(
        self,
        face_emb: Optional[np.ndarray],
        body_emb: Optional[np.ndarray],
        identity: Identity,
    ) -> float:
        """
        Compute a weighted similarity score between query embeddings and an identity.

        The score combines face embedding similarity (weight 0.7) and body
        embedding similarity (weight 0.3). If only one signal is available,
        it is used alone.

        Args:
            face_emb: Query face embedding or None.
            body_emb: Query body embedding or None.
            identity: The identity to compare against.

        Returns:
            Similarity score in [0, 1]. Returns 0.0 when no comparison is possible.
        """
        scores = []
        weights = []

        # Face embedding comparison
        if face_emb is not None and identity.face_embeddings:
            face_sim = self._best_cosine_similarity(face_emb, identity.face_embeddings)
            scores.append(face_sim)
            weights.append(0.7)

        # Body embedding comparison
        if body_emb is not None and identity.body_embeddings:
            body_sim = self._best_cosine_similarity(body_emb, identity.body_embeddings)
            scores.append(body_sim)
            weights.append(0.3)

        if not scores:
            return 0.0

        total_weight = sum(weights)
        weighted_score = sum(s * w for s, w in zip(scores, weights)) / total_weight
        return float(weighted_score)

    @staticmethod
    def _best_cosine_similarity(query: np.ndarray,
                                gallery: List[np.ndarray]) -> float:
        """
        Compute the maximum cosine similarity between a query vector and a
        gallery of vectors.

        Args:
            query: 1-D feature vector.
            gallery: List of 1-D feature vectors.

        Returns:
            Maximum cosine similarity in [-1, 1], clamped to [0, 1] since
            negative similarity indicates completely different identities.
        """
        query = np.asarray(query, dtype=np.float32).ravel()
        query_norm = np.linalg.norm(query)
        if query_norm < 1e-10:
            return 0.0

        query_normalized = query / query_norm

        best_sim = -1.0
        for g in gallery:
            g = np.asarray(g, dtype=np.float32).ravel()
            g_norm = np.linalg.norm(g)
            if g_norm < 1e-10:
                continue
            sim = float(np.dot(query_normalized, g / g_norm))
            if sim > best_sim:
                best_sim = sim

        # Clamp to [0, 1] - negative similarity means no match
        return max(0.0, best_sim)
