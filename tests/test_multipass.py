import numpy as np
import pytest
from multipass import (
    ForwardPass, BackwardPass, ConfidenceMerger, GapFiller,
    FrameResult, MergedFrameResult, FinalFrameResult, GapReport
)

class TestMultipass:
    """Tests for multipass processing components."""

    @pytest.fixture
    def sample_frame_result(self):
        """Sample FrameResult for testing."""
        return FrameResult(
            frame_number=10,
            timestamp=0.33,
            detections=[],
            tracks=[],
            classifications={},
            woman_head_bbox=(100, 100, 150, 150),
            woman_confidence=0.8,
            scene_context=None,
            processing_time=0.05
        )

    def test_confidence_merger_both_high_confidence(self, sample_frame_result):
        """Test merging when both passes have high confidence."""
        forward = sample_frame_result
        backward = FrameResult(
            frame_number=10,
            timestamp=0.33,
            detections=[],
            tracks=[],
            classifications={},
            woman_head_bbox=(105, 105, 155, 155),  # Slightly different bbox
            woman_confidence=0.9,  # Higher confidence
            scene_context=None,
            processing_time=0.04
        )

        merger = ConfidenceMerger(iou_threshold=0.5)
        result = merger._merge_single_frame(forward, backward)

        assert result.woman_confidence == 0.9  # Should use higher confidence
        assert result.merge_method == "backward_higher_confidence"

    def test_confidence_merger_weighted_average(self, sample_frame_result):
        """Test weighted average merging."""
        forward = sample_frame_result
        backward = FrameResult(
            frame_number=10,
            timestamp=0.33,
            detections=[],
            tracks=[],
            classifications={},
            woman_head_bbox=(100, 100, 150, 150),  # Same bbox
            woman_confidence=0.9,
            scene_context=None,
            processing_time=0.04
        )

        merger = ConfidenceMerger(iou_threshold=0.5)
        result = merger._merge_single_frame(forward, backward)

        assert result.merge_method == "weighted_average"
        assert result.woman_confidence == 0.85  # Average of 0.8 and 0.9

    def test_confidence_merger_gap(self):
        """Test merging when neither pass detects anything."""
        forward = FrameResult(
            frame_number=10,
            timestamp=0.33,
            detections=[],
            tracks=[],
            classifications={},
            woman_head_bbox=None,
            woman_confidence=0.0,
            scene_context=None,
            processing_time=0.05
        )
        backward = forward

        merger = ConfidenceMerger()
        result = merger._merge_single_frame(forward, backward)

        assert result.merge_method == "gap"
        assert result.woman_head_bbox is None
        assert result.woman_confidence == 0.0

    def test_gap_filler_interpolation(self):
        """Test gap filling with interpolation."""
        # Create merged results with a small gap
        results = []
        for i in range(15):
            if 5 <= i <= 7:  # Gap frames
                bbox = None
                conf = 0.0
                method = "gap"
            else:
                bbox = (100 + i*5, 100 + i*5, 150 + i*5, 150 + i*5)
                conf = 0.8
                method = "detected"

            results.append(MergedFrameResult(
                frame_number=i,
                timestamp=i/30.0,
                woman_head_bbox=bbox,
                woman_confidence=conf,
                merge_method=method,
                source_results={}
            ))

        filler = GapFiller(max_interpolation_gap=10)
        final_results = filler.fill_gaps(results)

        # Check that gap frames are interpolated
        for i in range(5, 8):
            assert final_results[i].fill_method == "interpolated"
            assert final_results[i].woman_head_bbox is not None

    def test_gap_filler_no_interpolation(self):
        """Test gap filling when gap is too large."""
        # Create merged results with a large gap
        results = []
        for i in range(25):
            if 5 <= i <= 16:  # Large gap (12 frames)
                bbox = None
                conf = 0.0
                method = "gap"
            else:
                bbox = (100 + i*5, 100 + i*5, 150 + i*5, 150 + i*5)
                conf = 0.8
                method = "detected"

            results.append(MergedFrameResult(
                frame_number=i,
                timestamp=i/30.0,
                woman_head_bbox=bbox,
                woman_confidence=conf,
                merge_method=method,
                source_results={}
            ))

        filler = GapFiller(max_interpolation_gap=10)
        final_results = filler.fill_gaps(results)

        # Check that large gap frames are marked for review
        for i in range(5, 17):
            assert final_results[i].fill_method == "review_needed"

    def test_gap_report_generation(self):
        """Test gap report generation."""
        gaps = [
            GapInfo(
                start_frame=5, end_frame=7, duration_frames=3,
                reason="occlusion", last_known_position=(100, 100, 150, 150),
                next_known_position=(120, 120, 170, 170), review_needed=False
            ),
            GapInfo(
                start_frame=10, end_frame=15, duration_frames=6,
                reason="scene_change", last_known_position=(120, 120, 170, 170),
                next_known_position=(200, 200, 250, 250), review_needed=True
            )
        ]

        filler = GapFiller()
        report = filler.get_gap_report(gaps)

        assert report.total_gaps == 2
        assert report.total_frames_with_gaps == 9
        assert report.max_gap_duration == 6
        assert report.gaps_requiring_review == 1

    def test_iou_calculation(self):
        """Test IoU calculation in confidence merger."""
        merger = ConfidenceMerger()

        # Identical bboxes
        iou = merger._calculate_iou((100, 100, 150, 150), (100, 100, 150, 150))
        assert iou == 1.0

        # No overlap
        iou = merger._calculate_iou((100, 100, 150, 150), (200, 200, 250, 250))
        assert iou == 0.0

        # Partial overlap
        iou = merger._calculate_iou((100, 100, 150, 150), (125, 125, 175, 175))
        assert iou > 0 and iou < 1

if __name__ == "__main__":
    pytest.main([__file__])