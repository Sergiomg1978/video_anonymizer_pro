"""
Video Anonymizer Pro
Professional video anonymization for adult women in videos with children
"""

__version__ = "1.0.0"

try:
    from .config import PipelineConfig
except ImportError:
    PipelineConfig = None

try:
    from .core.pipeline import AnonymizationPipeline
except ImportError:
    AnonymizationPipeline = None

__all__ = ["PipelineConfig", "AnonymizationPipeline"]
