"""
Quality validation tests
Tests PSNR and SSIM metrics for video encoding/decoding quality preservation
"""

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from core.video_io import VideoReader, VideoWriter
from quality.codec_manager import CodecManager


class TestQualityValidation:
    """Test quality preservation in video encoding/decoding"""

    def setup_method(self):
        """Setup test fixtures"""
        self.codec_manager = CodecManager()

        # Create synthetic test frames
        self.test_frames = self._create_test_frames()

        # Create temporary directory for test files
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Cleanup test fixtures"""
        # Remove temporary files
        for file in self.temp_dir.glob("*"):
            file.unlink()
        self.temp_dir.rmdir()

    def _create_test_frames(self, num_frames: int = 5) -> list:
        """Create synthetic test frames with various patterns"""
        frames = []

        for i in range(num_frames):
            # Create a 1920x1080 RGB frame
            frame = np.zeros((1080, 1920, 3), dtype=np.uint8)

            # Add different patterns for each frame
            if i == 0:
                # Gradient pattern
                for y in range(1080):
                    for x in range(1920):
                        frame[y, x] = [x % 256, y % 256, (x + y) % 256]
            elif i == 1:
                # Checkerboard pattern
                for y in range(1080):
                    for x in range(1920):
                        frame[y, x] = 255 if (x // 50 + y // 50) % 2 else 0
            elif i == 2:
                # Color bars
                bar_width = 1920 // 8
                colors = [
                    [255, 255, 255], [255, 255, 0], [0, 255, 255], [0, 255, 0],
                    [255, 0, 255], [255, 0, 0], [0, 0, 255], [0, 0, 0]
                ]
                for bar in range(8):
                    start_x = bar * bar_width
                    end_x = (bar + 1) * bar_width
                    frame[:, start_x:end_x] = colors[bar]
            elif i == 3:
                # Random noise (but deterministic for testing)
                np.random.seed(42)
                frame = np.random.randint(0, 256, (1080, 1920, 3), dtype=np.uint8)
            else:
                # Fine details pattern
                for y in range(1080):
                    for x in range(1920):
                        frame[y, x] = [
                            (x * 255) // 1920,
                            (y * 255) // 1080,
                            ((x + y) * 255) // (1920 + 1080)
                        ]

        frames.append(frame)
        return frames

    def _create_test_video(self, frames: list, output_path: str, fps: str = "30/1"):
        """Create a test video from frames"""
        if not frames:
            raise ValueError("No frames provided")

        height, width = frames[0].shape[:2]

        # Create mock metadata
        metadata = {
            'width': width,
            'height': height,
            'fps': fps,
            'codec_name': 'h264',
            'pix_fmt': 'yuv420p',
            'profile': 'High',
            'level': '4.0',
        }

        # Write video
        writer = VideoWriter(str(output_path), metadata, quality_mode="lossless")

        for frame in frames:
            writer.write_frame(frame)

        writer.finalize()

        return metadata

    def _calculate_metrics(self, original: np.ndarray, reconstructed: np.ndarray) -> dict:
        """Calculate PSNR and SSIM metrics"""
        # Ensure same shape
        if original.shape != reconstructed.shape:
            raise ValueError(f"Shape mismatch: {original.shape} vs {reconstructed.shape}")

        # Convert to float for calculations
        original_float = original.astype(np.float64)
        reconstructed_float = reconstructed.astype(np.float64)

        # Calculate PSNR
        psnr = peak_signal_noise_ratio(original_float, reconstructed_float, data_range=255)

        # Calculate SSIM (using luminance channel for simplicity)
        if len(original.shape) == 3:
            # Convert to grayscale for SSIM
            original_gray = np.dot(original_float, [0.2989, 0.5870, 0.1140])
            reconstructed_gray = np.dot(reconstructed_float, [0.2989, 0.5870, 0.1140])
        else:
            original_gray = original_float
            reconstructed_gray = reconstructed_float

        ssim = structural_similarity(original_gray, reconstructed_gray, data_range=255)

        return {
            'psnr': psnr,
            'ssim': ssim,
        }

    @pytest.mark.parametrize("quality_mode", ["lossless", "high", "medium"])
    def test_frame_encoding_quality(self, quality_mode):
        """Test that frame encoding/decoding preserves quality"""
        # Create test video
        test_video_path = self.temp_dir / f"test_{quality_mode}.mp4"
        original_frames = self.test_frames[:1]  # Use first frame

        # Create video
        metadata = self._create_test_video(original_frames, test_video_path)

        # Read back the frame
        reader = VideoReader(str(test_video_path))
        reconstructed_frame = reader.read_frame(0)

        # Calculate metrics
        metrics = self._calculate_metrics(original_frames[0], reconstructed_frame)

        # Log metrics for debugging
        print(f"Quality mode: {quality_mode}")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")

        # Assert quality requirements
        if quality_mode == "lossless":
            # For lossless, we expect perfect reconstruction
            assert metrics['psnr'] > 80.0, f"PSNR too low for lossless: {metrics['psnr']}"
            assert metrics['ssim'] > 0.999, f"SSIM too low for lossless: {metrics['ssim']}"
        elif quality_mode == "high":
            # High quality should be very close to original
            assert metrics['psnr'] > 50.0, f"PSNR too low for high quality: {metrics['psnr']}"
            assert metrics['ssim'] > 0.99, f"SSIM too low for high quality: {metrics['ssim']}"
        else:  # medium
            # Medium quality should still be good
            assert metrics['psnr'] > 40.0, f"PSNR too low for medium quality: {metrics['psnr']}"
            assert metrics['ssim'] > 0.95, f"SSIM too low for medium quality: {metrics['ssim']}"

    def test_codec_manager_analysis(self):
        """Test codec manager analysis functionality"""
        # Create test metadata
        test_metadata = {
            'codec_name': 'h264',
            'profile': 'High',
            'level': '4.0',
            'pix_fmt': 'yuv420p',
            'bitrate': 5000000,
            'fps': '30/1',
            'width': 1920,
            'height': 1080,
            'duration': 10.0,
        }

        # Analyze video
        analysis = self.codec_manager.analyze_video(test_metadata)

        # Check analysis results
        assert analysis['source_codec'] == 'h264'
        assert analysis['recommended_codec'] == 'libx264'
        assert analysis['is_high_quality'] is True

        # Get encoding parameters
        params = self.codec_manager.get_encoding_params(analysis, "lossless")

        # Validate parameters
        assert params['c:v'] == 'libx264'
        assert params['crf'] == '0'
        assert params['preset'] == 'veryslow'

    def test_metadata_preservation(self):
        """Test that video metadata is properly extracted and preserved"""
        # Create test video
        test_video_path = self.temp_dir / "metadata_test.mp4"
        metadata = self._create_test_video(self.test_frames[:1], test_video_path)

        # Read metadata back
        reader = VideoReader(str(test_video_path))
        read_metadata = reader.get_metadata()

        # Check key metadata fields
        assert read_metadata['width'] == metadata['width']
        assert read_metadata['height'] == metadata['height']
        assert read_metadata['fps'] == metadata['fps']
        assert read_metadata['codec_name'] in ['h264', 'avc1']  # FFmpeg might report differently

    def test_video_reader_functionality(self):
        """Test VideoReader basic functionality"""
        # Create test video with multiple frames
        test_video_path = self.temp_dir / "reader_test.mp4"
        self._create_test_video(self.test_frames, test_video_path)

        reader = VideoReader(str(test_video_path))

        # Test metadata
        metadata = reader.get_metadata()
        assert metadata['width'] > 0
        assert metadata['height'] > 0
        assert metadata['total_frames'] == len(self.test_frames)

        # Test frame reading
        frame = reader.read_frame(0)
        assert frame.shape == (metadata['height'], metadata['width'], 3)
        assert frame.dtype == np.uint8

        # Test frame iteration
        frames_read = list(reader.iterate_frames())
        assert len(frames_read) == len(self.test_frames)

        for i, (frame_num, frame_data) in enumerate(frames_read):
            assert frame_num == i
            assert frame_data.shape == (metadata['height'], metadata['width'], 3)

    def test_video_writer_functionality(self):
        """Test VideoWriter basic functionality"""
        test_video_path = self.temp_dir / "writer_test.mp4"

        metadata = {
            'width': 1920,
            'height': 1080,
            'fps': '30/1',
            'codec_name': 'h264',
            'pix_fmt': 'yuv420p',
            'profile': 'High',
            'level': '4.0',
        }

        writer = VideoWriter(str(test_video_path), metadata, quality_mode="lossless")

        # Write frames
        for frame in self.test_frames:
            writer.write_frame(frame)

        writer.finalize()

        # Verify file was created
        assert test_video_path.exists()
        assert test_video_path.stat().st_size > 0

        # Verify we can read it back
        reader = VideoReader(str(test_video_path))
        assert reader.get_total_frames() == len(self.test_frames)