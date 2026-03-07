"""
Video I/O module for lossless video processing.
Handles reading and writing videos while preserving quality and metadata.
"""

import numpy as np
from fractions import Fraction
from pathlib import Path
from typing import Generator, Tuple, Dict
import logging

logger = logging.getLogger(__name__)

import ffmpeg


class VideoReader:
    """Video reader using FFmpeg for lossless frame extraction."""

    def __init__(self, video_path: str):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._probe = ffmpeg.probe(str(self.video_path))
        self._video_stream = next(
            (s for s in self._probe['streams'] if s['codec_type'] == 'video'),
            None
        )
        if not self._video_stream:
            raise ValueError("No video stream found in file")

        self._width = int(self._video_stream['width'])
        self._height = int(self._video_stream['height'])

        # Get total frames with fallback for formats that don't report nb_frames
        nb_frames = self._video_stream.get('nb_frames')
        if nb_frames and int(nb_frames) > 0:
            self._total_frames = int(nb_frames)
        else:
            duration = float(self._probe['format'].get('duration', 0))
            fps_float = self._parse_fps_float()
            self._total_frames = int(duration * fps_float) if duration > 0 else 0

        self._fps = self._parse_fps_fraction()

    def _parse_fps_float(self) -> float:
        fps_str = self._video_stream.get('r_frame_rate', '30/1')
        parts = fps_str.split('/')
        if len(parts) == 2 and int(parts[1]) != 0:
            return int(parts[0]) / int(parts[1])
        return float(parts[0])

    def _parse_fps_fraction(self) -> Fraction:
        fps_str = self._video_stream.get('r_frame_rate', '30/1')
        parts = fps_str.split('/')
        if len(parts) == 2:
            return Fraction(int(parts[0]), int(parts[1]))
        return Fraction(int(float(parts[0])), 1)

    def get_metadata(self) -> Dict:
        """Extract comprehensive metadata from video."""
        metadata = {
            'filename': str(self.video_path.name),
            'path': str(self.video_path),
            'duration': float(self._probe['format'].get('duration', 0)),
            'size_bytes': int(self._probe['format'].get('size', 0)),
            'bitrate': int(self._probe['format'].get('bit_rate', 0)),
            'format_name': self._probe['format'].get('format_name', ''),
            'format_long_name': self._probe['format'].get('format_long_name', ''),
            'width': self._width,
            'height': self._height,
            'codec_name': self._video_stream.get('codec_name', ''),
            'codec_long_name': self._video_stream.get('codec_long_name', ''),
            'profile': self._video_stream.get('profile', ''),
            'level': self._video_stream.get('level', ''),
            'pix_fmt': self._video_stream.get('pix_fmt', ''),
            'color_space': self._video_stream.get('color_space', ''),
            'color_transfer': self._video_stream.get('color_transfer', ''),
            'color_primaries': self._video_stream.get('color_primaries', ''),
            'fps': str(self._fps),
            'fps_float': float(self._fps),
            'total_frames': self._total_frames,
            'aspect_ratio': self._width / self._height if self._height > 0 else 0,
            'display_aspect_ratio': self._video_stream.get('display_aspect_ratio', ''),
            'sample_aspect_ratio': self._video_stream.get('sample_aspect_ratio', ''),
        }

        audio_stream = next(
            (s for s in self._probe['streams'] if s['codec_type'] == 'audio'),
            None
        )
        if audio_stream:
            metadata['audio'] = {
                'codec_name': audio_stream.get('codec_name', ''),
                'channels': int(audio_stream.get('channels', 0)),
                'sample_rate': int(audio_stream.get('sample_rate', 0)),
                'bit_rate': int(audio_stream.get('bit_rate', 0)),
            }

        if 'tags' in self._probe['format']:
            metadata['container_tags'] = self._probe['format']['tags']

        return metadata

    def read_frame(self, frame_number: int) -> np.ndarray:
        """Read a specific frame by number (0-based) using FFmpeg seeking."""
        if frame_number < 0 or (self._total_frames > 0 and frame_number >= self._total_frames):
            raise ValueError(
                f"Frame number {frame_number} out of range "
                f"[0, {self._total_frames - 1}]"
            )

        frame_size = self._width * self._height * 3

        # Use select filter with vsync to extract exact frame
        out, _ = (
            ffmpeg
            .input(str(self.video_path))
            .filter('select', f'eq(n,{frame_number})')
            .output('pipe:', format='rawvideo', pix_fmt='bgr24', vsync='0')
            .run(capture_stdout=True, capture_stderr=True, quiet=True)
        )

        if len(out) < frame_size:
            raise RuntimeError(f"Failed to read frame {frame_number}")

        frame = np.frombuffer(out[:frame_size], np.uint8).reshape(
            self._height, self._width, 3
        )
        return frame.copy()

    def iterate_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Iterate through all frames using a single FFmpeg streaming pipe."""
        process = (
            ffmpeg
            .input(str(self.video_path))
            .output('pipe:', format='rawvideo', pix_fmt='bgr24')
            .run_async(pipe_stdout=True, quiet=True)
        )

        frame_size = self._width * self._height * 3
        frame_num = 0

        try:
            while True:
                raw = process.stdout.read(frame_size)
                if not raw or len(raw) < frame_size:
                    break
                frame = np.frombuffer(raw, np.uint8).reshape(
                    self._height, self._width, 3
                ).copy()
                yield frame_num, frame
                frame_num += 1
        finally:
            process.stdout.close()
            process.wait()

        if self._total_frames == 0 or frame_num != self._total_frames:
            self._total_frames = frame_num

    def get_total_frames(self) -> int:
        return self._total_frames

    def get_fps(self) -> Fraction:
        return self._fps

    def get_resolution(self) -> Tuple[int, int]:
        return (self._width, self._height)


class VideoWriter:
    """Video writer using FFmpeg for quality preservation."""

    def __init__(self, output_path: str, source_metadata: Dict,
                 quality_mode: str = "lossless"):
        self.output_path = Path(output_path)
        self.source_metadata = source_metadata
        self.quality_mode = quality_mode

        self.width = source_metadata['width']
        self.height = source_metadata['height']
        self.fps = Fraction(source_metadata['fps'])
        self.codec = source_metadata.get('codec_name', 'h264')
        self.pix_fmt = source_metadata.get('pix_fmt', 'yuv420p')

        self._process = None
        self._frame_count = 0
        self._init_writer()

    def _get_encoding_params(self) -> Dict:
        """Determine encoding parameters based on codec and quality mode."""
        params = {}

        if self.quality_mode == "lossless":
            if self.codec in ('h264', 'avc1'):
                params.update({'vcodec': 'libx264', 'preset': 'veryslow', 'crf': '0'})
            elif self.codec in ('h265', 'hevc'):
                params.update({'vcodec': 'libx265', 'preset': 'veryslow', 'crf': '0'})
            else:
                params.update({'vcodec': 'libx264', 'preset': 'veryslow', 'crf': '0'})
        elif self.quality_mode == "high":
            params.update({'vcodec': 'libx264', 'preset': 'slow', 'crf': '18'})
        else:
            params.update({'vcodec': 'libx264', 'preset': 'medium', 'crf': '23'})

        params['pix_fmt'] = self.pix_fmt
        return params

    def _init_writer(self):
        """Initialize FFmpeg writer process with proper pipe setup."""
        encoding = self._get_encoding_params()
        encoding['an'] = None  # Remove audio

        self._process = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='bgr24',
                   s=f'{self.width}x{self.height}', r=str(self.fps))
            .output(str(self.output_path), **encoding)
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

    def write_frame(self, frame: np.ndarray):
        """Write a frame to the video."""
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Frame shape {frame.shape} does not match "
                f"video resolution {self.width}x{self.height}"
            )
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        self._process.stdin.write(frame.tobytes())
        self._frame_count += 1

    def finalize(self):
        """Finalize the video writing and close resources."""
        if self._process:
            self._process.stdin.close()
            self._process.wait()
            self._process = None

    def get_frame_count(self) -> int:
        return self._frame_count

    def __del__(self):
        if self._process:
            try:
                self._process.stdin.close()
                self._process.wait()
            except Exception:
                pass
