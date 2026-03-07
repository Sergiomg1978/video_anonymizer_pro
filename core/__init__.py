"""Core modules for video I/O and pipeline orchestration."""

from .video_io import VideoReader, VideoWriter

__all__ = ["VideoReader", "VideoWriter"]

try:
    from .pipeline import AnonymizationPipeline
    __all__.append("AnonymizationPipeline")
except ImportError:
    pass
