"""Annotation module for manual frame annotation."""

from .anchor_frames import AnchorFrame, AnchorFrameManager

__all__ = ["AnchorFrame", "AnchorFrameManager"]

try:
    from .manual_annotator import ManualAnnotator
    __all__.append("ManualAnnotator")
except ImportError:
    pass
