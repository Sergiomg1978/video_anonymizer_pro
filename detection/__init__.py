"""
Detection module for Video Anonymizer Pro.
Provides face, person, head detection and age/gender classification.
"""

from detection.face_detector import MultiFaceDetector, FaceDetection
from detection.person_detector import PersonDetector, PersonDetection
from detection.head_detector import HeadDetector, HeadDetection
from detection.age_gender_classifier import AgeGenderClassifier, PersonClassification

__all__ = [
    "MultiFaceDetector",
    "FaceDetection",
    "PersonDetector",
    "PersonDetection",
    "HeadDetector",
    "HeadDetection",
    "AgeGenderClassifier",
    "PersonClassification",
]
