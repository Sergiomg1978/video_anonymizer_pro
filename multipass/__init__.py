"""
Multipass processing module for video anonymization.
Handles forward/backward passes, confidence merging, and gap filling.
"""

from .forward_pass import ForwardPass
from .backward_pass import BackwardPass
from .confidence_merger import ConfidenceMerger
from .gap_filler import GapFiller
from .types import (
    FrameResult, MergedFrameResult, FinalFrameResult,
    GapInfo, GapReport
)

__all__ = [
    'ForwardPass',
    'BackwardPass',
    'ConfidenceMerger',
    'GapFiller',
    'FrameResult',
    'MergedFrameResult',
    'FinalFrameResult',
    'GapInfo',
    'GapReport'
]