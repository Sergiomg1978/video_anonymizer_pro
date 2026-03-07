"""
Age and gender classification module.
Uses InsightFace face analysis when available; falls back to
heuristic estimation based on bounding-box geometry.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Import InsightFace
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except (ImportError, Exception) as e:
    INSIGHTFACE_AVAILABLE = False
    logger.warning(f"InsightFace not available, age/gender classification will use heuristics: {e}")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Config imports
try:
    from config import (
        AGE_CLASSIFICATION_ADULT_THRESHOLD,
        INSIGHTFACE_MODEL,
    )
except ImportError:
    AGE_CLASSIFICATION_ADULT_THRESHOLD = 14
    INSIGHTFACE_MODEL = "buffalo_l"


@dataclass
class PersonClassification:
    """Classification result for a detected person."""
    is_adult: bool
    is_female: bool
    age_estimate: float          # Estimated age in years
    gender_confidence: float     # Confidence for the predicted gender (0..1)
    confidence: float            # Overall classification confidence (0..1)

    @property
    def is_child(self) -> bool:
        return not self.is_adult

    @property
    def is_male(self) -> bool:
        return not self.is_female

    @property
    def age_group(self) -> str:
        """Coarse age group string for logging / debugging."""
        if self.age_estimate < 5:
            return "infant"
        elif self.age_estimate < 14:
            return "child"
        elif self.age_estimate < 18:
            return "teenager"
        elif self.age_estimate < 65:
            return "adult"
        else:
            return "senior"


class AgeGenderClassifier:
    """Classify age and gender from face / person crops.

    Primary method: InsightFace face analysis (buffalo_l model) which
    provides age and gender attributes from a face image.

    Fallback method: simple heuristic based on the ratio of face/head
    bounding box height to person bounding box height.  Children
    tend to have proportionally larger heads relative to body height.
    """

    def __init__(
        self,
        device: str = "cuda",
        model_name: str = INSIGHTFACE_MODEL,
        adult_age_threshold: float = AGE_CLASSIFICATION_ADULT_THRESHOLD,
    ):
        self.device = device
        self.model_name = model_name
        self.adult_age_threshold = adult_age_threshold

        self.face_analysis = None
        self._initialize_model()

    def _initialize_model(self):
        """Load InsightFace model for age/gender analysis."""
        if not INSIGHTFACE_AVAILABLE:
            logger.info(
                "InsightFace not installed. AgeGenderClassifier will use heuristic fallback. "
                "Install with: pip install insightface onnxruntime"
            )
            return

        try:
            ctx_id = -1 if self.device == "cpu" else 0
            self.face_analysis = FaceAnalysis(
                name=self.model_name,
                allowed_modules=["detection", "genderage"],
            )
            self.face_analysis.prepare(ctx_id=ctx_id, det_size=(640, 640))
            logger.info(
                f"InsightFace age/gender model loaded ({self.model_name}) "
                f"on {'CPU' if ctx_id == -1 else f'GPU:{ctx_id}'}"
            )
        except Exception as e:
            logger.warning(f"Failed to load InsightFace model: {e}. Using heuristic fallback.")
            self.face_analysis = None

    @property
    def is_available(self) -> bool:
        """Whether InsightFace-based classification is available."""
        return self.face_analysis is not None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        face_bbox: Optional[Tuple[int, int, int, int]] = None,
        keypoints: Optional[np.ndarray] = None,
    ) -> PersonClassification:
        """Classify a single person's age and gender.

        Tries InsightFace first (needs a visible face region).  If that
        fails or is unavailable, uses a heuristic fallback based on
        bounding-box proportions.

        Args:
            frame: Full BGR image (H, W, 3).
            person_bbox: (x1, y1, x2, y2) of the detected person.
            face_bbox: Optional (x1, y1, x2, y2) of a detected face
                within the person bbox.  Providing this improves both
                InsightFace accuracy and the heuristic fallback.
            keypoints: Optional body keypoints (not used by InsightFace
                but reserved for future skeleton-based classifiers).

        Returns:
            PersonClassification with age/gender estimates.
        """
        if frame is None or frame.size == 0:
            logger.warning("classify() received an empty frame")
            return self._default_classification()

        # Try InsightFace-based classification
        if self.face_analysis is not None:
            result = self._classify_insightface(frame, person_bbox, face_bbox)
            if result is not None:
                return result

        # Fallback to heuristic
        return self._classify_heuristic(frame, person_bbox, face_bbox)

    def classify_batch(
        self,
        frame: np.ndarray,
        person_bboxes: List[Tuple[int, int, int, int]],
        face_bboxes: Optional[List[Optional[Tuple[int, int, int, int]]]] = None,
    ) -> List[PersonClassification]:
        """Classify multiple persons in the same frame.

        Args:
            frame: Full BGR image.
            person_bboxes: List of person bounding boxes.
            face_bboxes: Optional parallel list of face bounding boxes.

        Returns:
            List of PersonClassification, one per person.
        """
        if not person_bboxes:
            return []

        if face_bboxes is None:
            face_bboxes = [None] * len(person_bboxes)
        while len(face_bboxes) < len(person_bboxes):
            face_bboxes.append(None)

        results: List[PersonClassification] = []
        for person_bb, face_bb in zip(person_bboxes, face_bboxes):
            results.append(self.classify(frame, person_bb, face_bbox=face_bb))
        return results

    # ------------------------------------------------------------------
    # InsightFace classification
    # ------------------------------------------------------------------

    def _classify_insightface(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> Optional[PersonClassification]:
        """Attempt classification via InsightFace.

        InsightFace needs a face crop.  We try:
        1. The provided face_bbox region.
        2. The upper portion of the person bbox (where the head usually is).
        3. The full person crop.
        """
        regions_to_try = []

        if face_bbox is not None:
            regions_to_try.append(("face_bbox", face_bbox))

        # Upper third of person bbox as a second attempt
        px1, py1, px2, py2 = person_bbox
        person_h = py2 - py1
        upper_third = (px1, py1, px2, py1 + max(1, person_h // 3))
        regions_to_try.append(("upper_person", upper_third))

        # Full person bbox as last resort
        regions_to_try.append(("full_person", person_bbox))

        height, width = frame.shape[:2]

        for region_name, bbox in regions_to_try:
            result = self._run_insightface_on_crop(frame, bbox, width, height)
            if result is not None:
                logger.debug(
                    f"InsightFace classification succeeded from {region_name}: "
                    f"age={result.age_estimate:.1f}, female={result.is_female}"
                )
                return result

        logger.debug("InsightFace could not find a face in any region")
        return None

    def _run_insightface_on_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
    ) -> Optional[PersonClassification]:
        """Run InsightFace on a cropped region of the frame.

        Returns a PersonClassification if a face with age/gender
        attributes is found, otherwise None.
        """
        x1, y1, x2, y2 = bbox
        # Add a small margin to help InsightFace detect the face
        margin_x = int((x2 - x1) * 0.1)
        margin_y = int((y2 - y1) * 0.1)

        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(frame_width, x2 + margin_x)
        y2 = min(frame_height, y2 + margin_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        try:
            faces = self.face_analysis.get(crop)
        except Exception as e:
            logger.debug(f"InsightFace analysis failed on crop: {e}")
            return None

        if not faces:
            return None

        # Use the face with the highest detection score
        best_face = max(faces, key=lambda f: float(getattr(f, "det_score", 0.0)))

        # Extract age
        age = getattr(best_face, "age", None)
        if age is None:
            return None
        age = float(age)

        # Extract gender (InsightFace: 0=female, 1=male)
        gender_raw = getattr(best_face, "gender", None)
        if gender_raw is None:
            return None
        is_female = int(gender_raw) == 0

        det_score = float(getattr(best_face, "det_score", 0.5))

        # Gender confidence: InsightFace does not expose a separate gender
        # probability, so we approximate from the detection score.
        gender_confidence = min(1.0, det_score + 0.1)

        return PersonClassification(
            is_adult=age >= self.adult_age_threshold,
            is_female=is_female,
            age_estimate=age,
            gender_confidence=gender_confidence,
            confidence=det_score,
        )

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _classify_heuristic(
        self,
        frame: np.ndarray,
        person_bbox: Tuple[int, int, int, int],
        face_bbox: Optional[Tuple[int, int, int, int]],
    ) -> PersonClassification:
        """Heuristic classification when InsightFace is unavailable.

        Uses the ratio of face/head bounding box height to person
        bounding box height.  In human anatomy, children have
        proportionally larger heads relative to their body height
        (roughly 1/4 to 1/5 of body height in young children vs 1/7 to
        1/8 in adults).

        Without a face bbox, the method defaults to ``is_adult=True``
        (conservative default for anonymization: assume adult).
        """
        px1, py1, px2, py2 = person_bbox
        person_height = py2 - py1

        if person_height <= 0:
            return self._default_classification()

        # ----------------------------------------------------------
        # Estimate age from head-to-body ratio
        # ----------------------------------------------------------
        age_estimate = 30.0  # default: assume mid-adult
        confidence = 0.2     # low confidence for heuristic

        if face_bbox is not None:
            fx1, fy1, fx2, fy2 = face_bbox
            face_height = fy2 - fy1

            if face_height > 0 and person_height > 0:
                head_body_ratio = face_height / person_height

                # Approximate mapping of head-to-body ratio to age:
                #   ratio > 0.30  -> likely young child  (~3-6 years)
                #   ratio  0.22-0.30 -> older child / teen (~7-14)
                #   ratio  0.15-0.22 -> teenager / young adult
                #   ratio < 0.15  -> adult
                if head_body_ratio > 0.30:
                    age_estimate = 5.0
                    confidence = 0.4
                elif head_body_ratio > 0.22:
                    age_estimate = 10.0
                    confidence = 0.35
                elif head_body_ratio > 0.15:
                    age_estimate = 16.0
                    confidence = 0.3
                else:
                    age_estimate = 30.0
                    confidence = 0.3
        else:
            # No face bbox: use absolute person height heuristic.
            # Very small bounding boxes (< 25% of frame height) *might*
            # indicate a child, but this is extremely unreliable because
            # of distance from camera.  Default to adult.
            frame_height = frame.shape[0] if frame is not None else 1080
            person_frame_ratio = person_height / frame_height

            if person_frame_ratio < 0.15:
                # Very small person - could be far away or a child
                age_estimate = 20.0   # assume young adult (conservative)
                confidence = 0.15
            else:
                age_estimate = 30.0
                confidence = 0.2

        is_adult = age_estimate >= self.adult_age_threshold

        # ----------------------------------------------------------
        # Gender: cannot determine from geometry alone
        # ----------------------------------------------------------
        is_female = False        # default (arbitrary; no signal available)
        gender_confidence = 0.0  # zero confidence = unknown

        return PersonClassification(
            is_adult=is_adult,
            is_female=is_female,
            age_estimate=age_estimate,
            gender_confidence=gender_confidence,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _default_classification() -> PersonClassification:
        """Return a safe default classification (adult, unknown gender)."""
        return PersonClassification(
            is_adult=True,
            is_female=False,
            age_estimate=30.0,
            gender_confidence=0.0,
            confidence=0.0,
        )
