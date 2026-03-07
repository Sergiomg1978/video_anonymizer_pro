"""
Type definitions for multipass processing.
"""

from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

@dataclass
class FrameResult:
    """Result of processing a single frame in forward or backward pass."""
    frame_number: int
    timestamp: float
    detections: List[Any]  # List of person detections
    tracks: List[Any]  # List of tracked persons
    classifications: Dict[int, Any]  # track_id -> PersonClassification
    woman_head_bbox: Optional[Tuple[int, int, int, int]]  # (x1, y1, x2, y2) if found
    woman_confidence: float
    scene_context: Optional[Any]  # SceneContext
    processing_time: float  # seconds

@dataclass
class MergedFrameResult:
    """Result after merging forward and backward passes."""
    frame_number: int
    timestamp: float
    woman_head_bbox: Optional[Tuple[int, int, int, int]]
    woman_confidence: float
    merge_method: str  # "forward_only", "backward_only", "highest_confidence", "weighted_average", "gap"
    source_results: Dict[str, FrameResult]  # {"forward": FrameResult, "backward": FrameResult}

@dataclass
class FinalFrameResult:
    """Final result after gap filling."""
    frame_number: int
    timestamp: float
    woman_head_bbox: Optional[Tuple[int, int, int, int]]
    woman_confidence: float
    fill_method: str  # "detected", "interpolated", "no_action", "review_needed"
    mask: Optional[np.ndarray]  # Generated mask for anonymization

@dataclass
class GapInfo:
    """Information about a detection gap."""
    start_frame: int
    end_frame: int
    duration_frames: int
    reason: str  # "occlusion", "exit_scene", "low_confidence", etc.
    last_known_position: Optional[Tuple[int, int, int, int]]
    next_known_position: Optional[Tuple[int, int, int, int]]
    review_needed: bool

@dataclass
class GapReport:
    """Report of all gaps found during processing."""
    total_gaps: int
    gaps: List[GapInfo]
    total_frames_with_gaps: int
    max_gap_duration: int
    gaps_requiring_review: int