"""
Confidence merger for multipass processing.
Combines results from forward and backward passes using confidence-based fusion.
"""

from typing import List, Optional, Tuple
import numpy as np
from .types import MergedFrameResult, FrameResult

class ConfidenceMerger:
    """
    Merges results from forward and backward passes based on confidence scores.
    """

    def __init__(self, iou_threshold: float = 0.5):
        """
        Initialize confidence merger.

        Args:
            iou_threshold: IoU threshold for considering bboxes as overlapping
        """
        self.iou_threshold = iou_threshold

    def merge(self, forward_results: List[FrameResult],
              backward_results: List[FrameResult]) -> List[MergedFrameResult]:
        """
        Merge forward and backward results for each frame.

        Args:
            forward_results: Results from forward pass
            backward_results: Results from backward pass

        Returns:
            List of merged results
        """
        if len(forward_results) != len(backward_results):
            raise ValueError("Forward and backward results must have same length")

        merged_results = []

        for forward_result, backward_result in zip(forward_results, backward_results):
            if forward_result.frame_number != backward_result.frame_number:
                raise ValueError("Frame numbers must match between forward and backward results")

            merged_result = self._merge_single_frame(forward_result, backward_result)
            merged_results.append(merged_result)

        return merged_results

    def _merge_single_frame(self, forward: FrameResult, backward: FrameResult) -> MergedFrameResult:
        """Merge results for a single frame."""
        forward_bbox = forward.woman_head_bbox
        backward_bbox = backward.woman_head_bbox
        forward_conf = forward.woman_confidence
        backward_conf = backward.woman_confidence

        # Case 1: Both passes detected with high confidence
        if forward_bbox is not None and backward_bbox is not None:
            if forward_conf > 0.7 and backward_conf > 0.7:
                # Both high confidence - use higher confidence one
                if forward_conf > backward_conf:
                    final_bbox, final_conf, method = forward_bbox, forward_conf, "forward_higher_confidence"
                else:
                    final_bbox, final_conf, method = backward_bbox, backward_conf, "backward_higher_confidence"
            else:
                # Check IoU
                iou = self._calculate_iou(forward_bbox, backward_bbox)
                if iou >= self.iou_threshold:
                    # Similar positions - use weighted average
                    final_bbox, final_conf, method = self._weighted_bbox_average(
                        forward_bbox, backward_bbox, forward_conf, backward_conf
                    )
                else:
                    # Different positions - use higher confidence
                    if forward_conf > backward_conf:
                        final_bbox, final_conf, method = forward_bbox, forward_conf, "forward_different_positions"
                    else:
                        final_bbox, final_conf, method = backward_bbox, backward_conf, "backward_different_positions"

        # Case 2: Only one pass detected
        elif forward_bbox is not None:
            final_bbox, final_conf, method = forward_bbox, forward_conf, "forward_only"
        elif backward_bbox is not None:
            final_bbox, final_conf, method = backward_bbox, backward_conf, "backward_only"

        # Case 3: Neither detected
        else:
            final_bbox, final_conf, method = None, 0.0, "gap"

        return MergedFrameResult(
            frame_number=forward.frame_number,
            timestamp=forward.timestamp,
            woman_head_bbox=final_bbox,
            woman_confidence=final_conf,
            merge_method=method,
            source_results={"forward": forward, "backward": backward}
        )

    def _calculate_iou(self, bbox1: Tuple[int, int, int, int],
                      bbox2: Tuple[int, int, int, int]) -> float:
        """Calculate Intersection over Union of two bboxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0

        intersection_area = (x2_i - x1_i) * (y2_i - y1_i)

        # Union
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area

        return intersection_area / union_area if union_area > 0 else 0.0

    def _weighted_bbox_average(self, bbox1: Tuple[int, int, int, int],
                              bbox2: Tuple[int, int, int, int],
                              conf1: float, conf2: float) -> Tuple[Tuple[int, int, int, int], float, str]:
        """Compute weighted average of two bboxes."""
        # Normalize weights
        total_conf = conf1 + conf2
        w1 = conf1 / total_conf
        w2 = conf2 / total_conf

        # Weighted average of coordinates
        x1_avg = int(w1 * bbox1[0] + w2 * bbox2[0])
        y1_avg = int(w1 * bbox1[1] + w2 * bbox2[1])
        x2_avg = int(w1 * bbox1[2] + w2 * bbox2[2])
        y2_avg = int(w1 * bbox1[3] + w2 * bbox2[3])

        final_bbox = (x1_avg, y1_avg, x2_avg, y2_avg)
        final_conf = (conf1 + conf2) / 2  # Average confidence

        return final_bbox, final_conf, "weighted_average"