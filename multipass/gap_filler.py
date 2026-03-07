"""
Gap filler for multipass processing.
Fills detection gaps using interpolation or marks for manual review.
"""

from typing import List, Optional, Tuple
import numpy as np
from .types import FinalFrameResult, MergedFrameResult, GapReport, GapInfo

class GapFiller:
    """
    Fills gaps in detection results using interpolation or marks for review.
    """

    def __init__(self, max_interpolation_gap: int = 10):
        """
        Initialize gap filler.

        Args:
            max_interpolation_gap: Maximum gap size (in frames) to interpolate
        """
        self.max_interpolation_gap = max_interpolation_gap

    def fill_gaps(self, merged_results: List[MergedFrameResult]) -> List[FinalFrameResult]:
        """
        Fill gaps in merged results.

        Args:
            merged_results: Results after confidence merging

        Returns:
            Final results with gaps filled
        """
        final_results = []

        # Find all gaps
        gaps = self._find_gaps(merged_results)

        for result in merged_results:
            if result.merge_method == "gap":
                # This is a gap - try to fill it
                gap_info = self._get_gap_for_frame(result.frame_number, gaps)
                if gap_info:
                    filled_result = self._fill_single_gap(result, gap_info, merged_results)
                else:
                    filled_result = FinalFrameResult(
                        frame_number=result.frame_number,
                        timestamp=result.timestamp,
                        woman_head_bbox=None,
                        woman_confidence=0.0,
                        fill_method="gap_unfilled",
                        mask=None
                    )
            else:
                # No gap - use as is
                filled_result = FinalFrameResult(
                    frame_number=result.frame_number,
                    timestamp=result.timestamp,
                    woman_head_bbox=result.woman_head_bbox,
                    woman_confidence=result.woman_confidence,
                    fill_method="detected",
                    mask=None  # Will be generated later by mask_generator
                )

            final_results.append(filled_result)

        return final_results

    def _find_gaps(self, merged_results: List[MergedFrameResult]) -> List[GapInfo]:
        """Find all gaps in the results."""
        gaps = []
        current_gap_start = None
        last_known_position = None

        for i, result in enumerate(merged_results):
            if result.merge_method == "gap":
                if current_gap_start is None:
                    current_gap_start = result.frame_number
                    # Find last known position before gap
                    for j in range(i - 1, -1, -1):
                        if merged_results[j].woman_head_bbox is not None:
                            last_known_position = merged_results[j].woman_head_bbox
                            break
            else:
                if current_gap_start is not None:
                    # End of gap
                    gap_end = result.frame_number - 1
                    gap_duration = gap_end - current_gap_start + 1

                    # Find next known position after gap
                    next_known_position = None
                    for j in range(i, len(merged_results)):
                        if merged_results[j].woman_head_bbox is not None:
                            next_known_position = merged_results[j].woman_head_bbox
                            break

                    # Determine reason
                    reason = self._determine_gap_reason(last_known_position, next_known_position, gap_duration)

                    gap_info = GapInfo(
                        start_frame=current_gap_start,
                        end_frame=gap_end,
                        duration_frames=gap_duration,
                        reason=reason,
                        last_known_position=last_known_position,
                        next_known_position=next_known_position,
                        review_needed=gap_duration >= self.max_interpolation_gap
                    )

                    gaps.append(gap_info)
                    current_gap_start = None
                    last_known_position = None

        # Handle gap at the end
        if current_gap_start is not None:
            gap_end = merged_results[-1].frame_number
            gap_duration = gap_end - current_gap_start + 1
            reason = "exit_scene"  # Assume exited at end

            gap_info = GapInfo(
                start_frame=current_gap_start,
                end_frame=gap_end,
                duration_frames=gap_duration,
                reason=reason,
                last_known_position=last_known_position,
                next_known_position=None,
                review_needed=False  # No need to review if exited
            )
            gaps.append(gap_info)

        return gaps

    def _get_gap_for_frame(self, frame_number: int, gaps: List[GapInfo]) -> Optional[GapInfo]:
        """Find the gap that contains the given frame."""
        for gap in gaps:
            if gap.start_frame <= frame_number <= gap.end_frame:
                return gap
        return None

    def _fill_single_gap(self, result: MergedFrameResult, gap_info: GapInfo,
                        all_results: List[MergedFrameResult]) -> FinalFrameResult:
        """Fill a single gap frame."""
        if gap_info.duration_frames < self.max_interpolation_gap and gap_info.last_known_position and gap_info.next_known_position:
            # Interpolate
            progress = (result.frame_number - gap_info.start_frame) / gap_info.duration_frames
            interpolated_bbox = self._interpolate_bbox(
                gap_info.last_known_position,
                gap_info.next_known_position,
                progress
            )

            return FinalFrameResult(
                frame_number=result.frame_number,
                timestamp=result.timestamp,
                woman_head_bbox=interpolated_bbox,
                woman_confidence=0.5,  # Lower confidence for interpolated
                fill_method="interpolated",
                mask=None
            )

        elif gap_info.reason == "exit_scene":
            # No action needed
            return FinalFrameResult(
                frame_number=result.frame_number,
                timestamp=result.timestamp,
                woman_head_bbox=None,
                woman_confidence=0.0,
                fill_method="no_action",
                mask=None
            )

        else:
            # Mark for review
            return FinalFrameResult(
                frame_number=result.frame_number,
                timestamp=result.timestamp,
                woman_head_bbox=None,
                woman_confidence=0.0,
                fill_method="review_needed",
                mask=None
            )

    def _interpolate_bbox(self, bbox1: Tuple[int, int, int, int],
                         bbox2: Tuple[int, int, int, int], progress: float) -> Tuple[int, int, int, int]:
        """Linearly interpolate between two bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        x1_interp = int(x1_1 + progress * (x1_2 - x1_1))
        y1_interp = int(y1_1 + progress * (y1_2 - y1_1))
        x2_interp = int(x2_1 + progress * (x2_2 - x2_1))
        y2_interp = int(y2_1 + progress * (y2_2 - y2_1))

        return (x1_interp, y1_interp, x2_interp, y2_interp)

    def _determine_gap_reason(self, last_pos: Optional[Tuple[int, int, int, int]],
                            next_pos: Optional[Tuple[int, int, int, int]],
                            duration: int) -> str:
        """Determine the reason for a gap."""
        if last_pos is None and next_pos is None:
            return "no_detection"
        elif last_pos is None:
            return "entrance_scene"
        elif next_pos is None:
            return "exit_scene"
        else:
            # Check if positions are far apart (possible occlusion or scene change)
            # Simple distance check
            center1 = ((last_pos[0] + last_pos[2]) / 2, (last_pos[1] + last_pos[3]) / 2)
            center2 = ((next_pos[0] + next_pos[2]) / 2, (next_pos[1] + next_pos[3]) / 2)
            distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

            if distance > 100:  # Arbitrary threshold
                return "scene_change_or_occlusion"
            else:
                return "temporary_occlusion"

    def get_gap_report(self, gaps: List[GapInfo]) -> GapReport:
        """Generate a report of gaps."""
        total_frames_with_gaps = sum(gap.duration_frames for gap in gaps)
        max_gap_duration = max((gap.duration_frames for gap in gaps), default=0)
        gaps_requiring_review = sum(1 for gap in gaps if gap.review_needed)

        return GapReport(
            total_gaps=len(gaps),
            gaps=gaps,
            total_frames_with_gaps=total_frames_with_gaps,
            max_gap_duration=max_gap_duration,
            gaps_requiring_review=gaps_requiring_review
        )