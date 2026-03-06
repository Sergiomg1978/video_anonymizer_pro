"""
Configuration file for Video Anonymizer Pro
Contains all configurable constants and settings
"""

from dataclasses import dataclass
from typing import Optional

# Detection settings
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.3
HEAD_DETECTION_CONFIDENCE_THRESHOLD = 0.4
PERSON_DETECTION_CONFIDENCE_THRESHOLD = 0.5
AGE_CLASSIFICATION_ADULT_THRESHOLD = 14  # years

# Tracking settings
TRACKING_MAX_AGE = 90  # frames
TRACKING_N_INIT = 3
TRACKING_MAX_COSINE_DISTANCE = 0.3
TRACKING_NN_BUDGET = 150

# Re-identification settings
REID_EMBEDDING_SIMILARITY_THRESHOLD = 0.5
REID_BODY_SIMILARITY_THRESHOLD = 0.6

# Anonymization settings
BLUR_MODE = "gaussian"  # gaussian, pixelate, solid, mosaic
TEMPORAL_SMOOTHING_FRAMES = 5
MASK_DILATION_PX = 5
MASK_FEATHER_PX = 3

# Video processing settings
QUALITY_MODE = "lossless"  # lossless, high, medium
VIDEO_FPS_FRACTION_TOLERANCE = 0.01  # for exact FPS matching
MAX_INTERPOLATION_GAP = 10  # frames

# Scene analysis settings
SHOT_CHANGE_HISTOGRAM_THRESHOLD = 0.7
MOTION_ESTIMATION_OPTICAL_FLOW = True

# Pipeline settings
MULTIPASS_ENABLED = True
USE_MANUAL_ANNOTATION = True
USE_VLM_ANALYSIS = False
USE_SAM_SEGMENTATION = True
DEBUG_MODE = False

# Hardware settings
DEVICE = "auto"  # auto, cuda, cpu
GPU_MEMORY_FRACTION = 0.8  # fraction of GPU memory to use

# Logging settings
LOG_LEVEL = "INFO"  # DEBUG, INFO, WARNING, ERROR
LOG_FILE = "video_anonymizer.log"

# Model paths (will be downloaded if not present)
YOLO_FACE_MODEL = "yolov8n-face.pt"
YOLO_PERSON_MODEL = "yolov8x.pt"
YOLO_HEAD_MODEL = "yolov5-head.pt"
SAM_MODEL = "sam2_hiera_tiny.pt"
INSIGHTFACE_MODEL = "buffalo_l"

# GUI settings
GUI_THEME = "dark"  # dark, light
ANNOTATION_AUTO_EXTRACT_INTERVAL = 2.0  # seconds

# Output settings
PRESERVE_AUDIO = False  # always remove audio
GENERATE_DEBUG_VIDEO = False
DEBUG_VIDEO_SUFFIX = "_debug"

@dataclass
class PipelineConfig:
    """Configuration for the anonymization pipeline"""
    blur_mode: str = BLUR_MODE
    use_manual_annotation: bool = USE_MANUAL_ANNOTATION
    use_vlm: bool = USE_VLM_ANALYSIS
    use_sam: bool = USE_SAM_SEGMENTATION
    multipass: bool = MULTIPASS_ENABLED
    quality_mode: str = QUALITY_MODE
    device: str = DEVICE
    temporal_smoothing: int = TEMPORAL_SMOOTHING_FRAMES
    confidence_threshold: float = FACE_DETECTION_CONFIDENCE_THRESHOLD
    interpolation_max_gap: int = MAX_INTERPOLATION_GAP
    debug_output: bool = DEBUG_MODE

    def __post_init__(self):
        # Validation
        if self.blur_mode not in ["gaussian", "pixelate", "solid", "mosaic"]:
            raise ValueError(f"Invalid blur_mode: {self.blur_mode}")
        if self.quality_mode not in ["lossless", "high", "medium"]:
            raise ValueError(f"Invalid quality_mode: {self.quality_mode}")
        if self.device not in ["auto", "cuda", "cpu"]:
            raise ValueError(f"Invalid device: {self.device}")