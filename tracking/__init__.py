"""
Tracking package for Video Anonymizer Pro.

Provides multi-object person tracking, persistent identity management,
and re-identification after occlusion.
"""

from .deep_sort_tracker import TrackedPerson, PersonTracker
from .identity_manager import Identity, IdentityManager
from .reidentification import ReIdentifier

__all__ = [
    "TrackedPerson",
    "PersonTracker",
    "Identity",
    "IdentityManager",
    "ReIdentifier",
]
