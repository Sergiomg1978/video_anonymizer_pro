# Video Anonymizer Pro

**Video Anonymizer Pro** is a professional Python application designed to automatically anonymize the face/head of adult women in videos where they appear alongside children. The system preserves the original video quality while ensuring that only the adult female is anonymized, leaving children untouched.

## Features

- **Multi-Model Detection**: Combines YOLO, MediaPipe, and RetinaFace for robust face detection
- **Head Detection**: Detects complete heads even when faces are partially obscured
- **Age/Gender Classification**: Distinguishes adults from children using multiple signals
- **Advanced Tracking**: Uses Deep SORT for stable multi-object tracking with re-identification
- **Manual Annotation GUI**: Optional GUI for annotating reference frames to improve accuracy
- **Scene Analysis**: AI-powered scene understanding with optional Vision-Language Models
- **Precise Segmentation**: Uses SAM 2 for accurate head masking
- **Multi-Pass Processing**: Forward and backward passes with confidence merging
- **Quality Preservation**: Lossless video processing with metadata preservation
- **GPU Acceleration**: Optimized for NVIDIA GPUs with CUDA support

## System Requirements

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended for performance)
- **RAM**: 16GB+ recommended
- **VRAM**: 8GB+ for full model suite
- **FFmpeg**: Must be installed and in PATH

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/video-anonymizer-pro.git
cd video-anonymizer-pro
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Package (Optional)
```bash
pip install -e .
```

## Usage

### Command Line Interface

```bash
python main.py input_video.mp4 output_video.mp4 [options]
```

### Basic Usage
```bash
python main.py my_video.mp4 anonymized_video.mp4
```

### Full Processing with Manual Annotation
```bash
python main.py my_video.mp4 anonymized_video.mp4 --mode full --annotate
```

### Quick Processing (Single Pass)
```bash
python main.py my_video.mp4 anonymized_video.mp4 --mode quick --blur pixelate
```

### Debug Mode with Visualizations
```bash
python main.py my_video.mp4 anonymized_video.mp4 --debug --debug-video debug_output.mp4
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--mode` | Processing mode: `full`, `auto`, `quick`, `debug` | `auto` |
| `--blur` | Blur type: `gaussian`, `pixelate`, `solid`, `mosaic` | `gaussian` |
| `--quality` | Quality mode: `lossless`, `high`, `medium` | `lossless` |
| `--device` | Device: `auto`, `cuda`, `cpu` | `auto` |
| `--confidence-threshold` | Detection confidence threshold | `0.4` |
| `--temporal-smoothing` | Frames for temporal smoothing | `5` |
| `--use-sam` | Enable SAM 2 segmentation | `True` |
| `--use-vlm` | Enable Vision-Language analysis | `False` |
| `--annotate` | Launch manual annotation GUI | `False` |
| `--debug` | Enable debug logging | `False` |
| `--report` | Generate JSON report file | `None` |

## Examples

### Example 1: Basic Anonymization
```bash
python main.py family_video.mp4 family_video_anon.mp4
```
Processes the video with automatic detection and default settings.

### Example 2: High-Quality Processing with Manual Annotation
```bash
python main.py interview.mp4 interview_anon.mp4 --mode full --annotate --quality lossless --use-sam
```
Launches GUI for manual annotation, uses SAM 2 for precise masking, ensures lossless quality.

### Example 3: Batch Processing
```bash
for video in *.mp4; do
    python main.py "$video" "anon_$video" --mode quick
done
```

### Example 4: Debug Analysis
```bash
python main.py test_video.mp4 test_anon.mp4 --debug --debug-video test_debug.mp4 --report analysis.json
```
Generates debug video with bounding boxes and confidence scores, plus detailed JSON report.

## Project Structure

```
video_anonymizer_pro/
├── main.py                          # CLI entry point
├── config.py                        # Global configuration
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── setup.py                         # Package setup
│
├── core/
│   ├── __init__.py
│   ├── pipeline.py                  # Main pipeline orchestrator
│   ├── video_io.py                  # Lossless video I/O
│   └── frame_extractor.py           # Frame extraction for annotation
│
├── detection/
│   ├── __init__.py
│   ├── face_detector.py             # Multi-model face detection
│   ├── head_detector.py             # Head detection
│   ├── person_detector.py           # Person detection
│   └── age_gender_classifier.py     # Adult/child classification
│
├── tracking/
│   ├── __init__.py
│   ├── deep_sort_tracker.py         # Deep SORT tracking
│   ├── identity_manager.py          # Identity management
│   └── reidentification.py          # Re-identification
│
├── annotation/
│   ├── __init__.py
│   ├── manual_annotator.py          # Manual annotation GUI
│   └── anchor_frames.py             # Anchor frame management
│
├── scene_analysis/
│   ├── __init__.py
│   ├── scene_interpreter.py         # Scene understanding
│   ├── shot_detector.py             # Shot change detection
│   └── motion_estimator.py          # Motion estimation
│
├── anonymization/
│   ├── __init__.py
│   ├── blur_engine.py               # Blurring engine
│   ├── mask_generator.py            # SAM 2 mask generation
│   └── inpainting_engine.py         # Optional inpainting
│
├── multipass/
│   ├── __init__.py
│   ├── forward_pass.py              # Forward processing
│   ├── backward_pass.py             # Backward processing
│   ├── confidence_merger.py         # Result merging
│   └── gap_filler.py                # Gap filling
│
├── quality/
│   ├── __init__.py
│   ├── codec_manager.py             # Codec management
│   ├── frame_validator.py           # Quality validation
│   └── metadata_preserver.py        # Metadata preservation
│
├── utils/
│   ├── __init__.py
│   ├── logger.py                    # Professional logging
│   ├── gpu_manager.py               # GPU management
│   ├── progress_tracker.py          # Progress tracking
│   └── visualization.py             # Debug visualization
│
└── tests/
    ├── __init__.py
    ├── test_detection.py
    ├── test_tracking.py
    ├── test_pipeline.py
    └── test_quality.py
```

## Configuration

All configurable parameters are defined in `config.py`. Key settings include:

- Detection confidence thresholds
- Tracking parameters
- Anonymization modes
- Hardware settings
- Model paths

## Testing

Run the test suite:
```bash
pytest tests/
```

## Performance Notes

- **GPU Recommended**: CPU processing is 10-20x slower
- **Memory Usage**: ~8-12GB VRAM for full pipeline
- **Processing Speed**: ~1-5 FPS depending on video resolution and settings
- **Quality**: Lossless mode preserves original quality

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

Built with:
- Ultralytics YOLO
- Meta SAM 2
- InsightFace
- Deep SORT
- MediaPipe
- And many other open-source libraries
