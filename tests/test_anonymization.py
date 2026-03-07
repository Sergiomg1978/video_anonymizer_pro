"""
Tests for anonymization module
"""

import pytest
import numpy as np
import cv2
from unittest.mock import Mock, patch
from anonymization import MaskGenerator


class TestMaskGenerator:
    """Test mask generation functionality"""

    def test_initialization_fallback(self):
        """Test MaskGenerator initialization with fallback"""
        generator = MaskGenerator(model="fallback")
        assert generator.model_type == "fallback"
        assert generator.device == "cuda"
        assert not generator.propagation_initialized

    def test_generate_fallback_mask(self):
        """Test fallback mask generation"""
        generator = MaskGenerator(model="fallback")

        # Create test frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        bbox = (200, 150, 280, 230)  # Example head bbox

        mask = generator.generate_mask(frame, bbox)

        # Check mask properties
        assert mask.shape == (480, 640)
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() <= mask.max() <= 1.0

        # Check that mask has non-zero values in the bbox region
        x1, y1, x2, y2 = bbox
        expanded_x1 = max(0, x1 - int((x2-x1) * 0.15))
        expanded_y1 = max(0, y1 - int((y2-y1) * 0.15))
        expanded_x2 = min(640, x2 + int((x2-x1) * 0.15))
        expanded_y2 = min(480, y2 + int((y2-y1) * 0.15))

        # Should have some non-zero values in expanded region
        region_mask = mask[expanded_y1:expanded_y2, expanded_x1:expanded_x2]
        assert region_mask.max() > 0.0

    def test_refine_mask(self):
        """Test mask refinement"""
        generator = MaskGenerator()

        # Create a simple binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(mask, (50, 50), 20, 1, -1)
        mask = mask.astype(np.float32)

        # Refine with dilation and feathering
        refined = generator.refine_mask(mask, dilation_px=2, feather_px=3)

        assert refined.shape == mask.shape
        assert refined.dtype == np.float32
        assert 0.0 <= refined.min() <= refined.max() <= 1.0

        # Refined mask should be smoother (less binary)
        binary_pixels = np.sum((refined == 0.0) | (refined == 1.0))
        total_pixels = refined.size
        # Should have some intermediate values due to feathering
        assert binary_pixels < total_pixels

    def test_refine_mask_no_dilation(self):
        """Test mask refinement without dilation"""
        generator = MaskGenerator()

        mask = np.ones((50, 50), dtype=np.float32) * 0.5
        refined = generator.refine_mask(mask, dilation_px=0, feather_px=2)

        assert refined.shape == mask.shape
        # Should still be blurred but not dilated
        assert np.allclose(refined, mask, atol=0.1)  # Allow some blurring

    def test_get_model_info(self):
        """Test model info retrieval"""
        generator = MaskGenerator(model="fallback", device="cpu")

        info = generator.get_model_info()

        expected_keys = ["model_type", "device", "sam2_available",
                        "mobile_sam_available", "video_propagation_active"]

        for key in expected_keys:
            assert key in info

        assert info["model_type"] == "fallback"
        assert info["device"] == "cpu"
        assert info["video_propagation_active"] == False

    def test_reset_propagation(self):
        """Test propagation reset"""
        generator = MaskGenerator()
        generator.propagation_initialized = True
        generator.frame_count = 5

        generator.reset_propagation()

        assert not generator.propagation_initialized
        assert generator.frame_count == 0

    @patch('anonymization.mask_generator.SAM2_AVAILABLE', True)
    @patch('anonymization.mask_generator.SAM2ImagePredictor')
    def test_sam2_initialization(self, mock_predictor_class):
        """Test SAM2 initialization when available"""
        mock_predictor = Mock()
        mock_predictor_class.return_value = mock_predictor

        with patch('anonymization.mask_generator.build_sam2') as mock_build:
            mock_model = Mock()
            mock_build.return_value = mock_model

            generator = MaskGenerator(model="sam2")

            assert generator.model_type == "sam2"
            assert generator.sam_predictor == mock_predictor

    @patch('anonymization.mask_generator.SAM2_AVAILABLE', False)
    def test_sam2_fallback_when_unavailable(self):
        """Test fallback when SAM2 is not available"""
        generator = MaskGenerator(model="sam2")

        # Should fall back to fallback mode
        assert generator.model_type == "fallback"

    def test_bbox_bounds_checking(self):
        """Test that bbox coordinates are properly clamped"""
        generator = MaskGenerator(model="fallback")

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Test bbox extending beyond frame bounds
        bbox = (-10, -10, 50, 50)  # Negative coordinates

        mask = generator.generate_mask(frame, bbox)

        # Should not crash and should produce valid mask
        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32

    def test_empty_bbox(self):
        """Test handling of empty/invalid bbox"""
        generator = MaskGenerator(model="fallback")

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Invalid bbox (x2 < x1)
        bbox = (50, 50, 30, 70)

        mask = generator.generate_mask(frame, bbox)

        # Should handle gracefully
        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32

    def test_video_propagation_initialization_fallback(self):
        """Test video propagation initialization fallback"""
        generator = MaskGenerator(model="fallback")

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = (20, 20, 40, 40)

        # Should not crash
        generator.initialize_video_propagation(frame, bbox)

        assert not generator.propagation_initialized

    def test_propagate_mask_fallback(self):
        """Test mask propagation fallback"""
        generator = MaskGenerator(model="fallback")

        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        mask = generator.propagate_mask(frame)

        assert mask.shape == (100, 100)
        assert mask.dtype == np.float32
<parameter name="filePath">c:\rec3\video_anonymizer_pro\tests\test_anonymization.py