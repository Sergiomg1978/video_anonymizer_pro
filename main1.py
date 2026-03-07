"""
Video Anonymizer Pro - Main CLI Entry Point
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import PipelineConfig
from core.pipeline import AnonymizationPipeline
from utils.logger import setup_logger

def main():
    parser = argparse.ArgumentParser(
        description="Video Anonymizer Pro - Automatically anonymize adult women in videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.mp4 output.mp4
  python main.py input.mp4 output.mp4 --mode full --annotate
  python main.py input.mp4 output.mp4 --debug --debug-video debug.mp4
        """
    )

    parser.add_argument('input', help='Input video file')
    parser.add_argument('output', help='Output video file')

    parser.add_argument('--mode', choices=['full', 'auto', 'quick', 'debug'],
                       default='auto', help='Processing mode')
    parser.add_argument('--blur', choices=['gaussian', 'pixelate', 'solid', 'mosaic'],
                       default='gaussian', help='Anonymization blur mode')
    parser.add_argument('--quality', choices=['lossless', 'high', 'medium'],
                       default='lossless', help='Video quality mode')
    parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'],
                       default='auto', help='Compute device')
    parser.add_argument('--confidence-threshold', type=float, default=0.4,
                       help='Detection confidence threshold')
    parser.add_argument('--temporal-smoothing', type=int, default=5,
                       help='Temporal smoothing frames')
    parser.add_argument('--use-sam', action='store_true', default=True,
                       help='Use SAM 2 for segmentation')
    parser.add_argument('--use-vlm', action='store_true', default=False,
                       help='Use Vision-Language Models')
    parser.add_argument('--annotate', action='store_true', default=False,
                       help='Launch manual annotation GUI')
    parser.add_argument('--debug', action='store_true', default=False,
                       help='Enable debug mode')
    parser.add_argument('--debug-video', help='Output debug video file')
    parser.add_argument('--report', help='Output JSON report file')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')

    args = parser.parse_args()

    # Setup logging
    setup_logger(level=args.log_level)

    # Create configuration
    config = PipelineConfig(
        blur_mode=args.blur,
        use_manual_annotation=args.annotate,
        use_vlm=args.use_vlm,
        use_sam=args.use_sam,
        multipass=(args.mode in ['full', 'debug']),
        quality_mode=args.quality,
        device=args.device,
        temporal_smoothing=args.temporal_smoothing,
        confidence_threshold=args.confidence_threshold,
        debug_output=args.debug
    )

    # Create pipeline
    pipeline = AnonymizationPipeline(config)

    # Run pipeline
    try:
        pipeline.run(args.input, args.output)
        print("Anonymization completed successfully!")
    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()