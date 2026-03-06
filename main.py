#!/usr/bin/env python3
"""
Video Anonymizer Pro - CLI Entry Point
Anonymizes faces/heads of adult women in videos while preserving children.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import List

# Rich imports with fallback
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        TimeElapsedColumn, TimeRemainingColumn,
    )
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Minimal fallbacks when Rich is not installed
if not RICH_AVAILABLE:
    import re as _re

    class Console:
        def print(self, *args, **kwargs):
            text = " ".join(str(a) for a in args)
            text = _re.sub(r'\[/?[^\]]*\]', '', text)
            print(text)
        def log(self, *args, **kwargs):
            self.print(*args, **kwargs)

    class Table:
        def __init__(self, **kw):
            self.title = kw.get("title", "")
            self._rows = []
        def add_column(self, name, **kw): pass
        def add_row(self, *vals):
            self._rows.append(vals)
        def __str__(self):
            lines = [self.title] if self.title else []
            for row in self._rows:
                lines.append("  ".join(str(v) for v in row))
            return "\n".join(lines)

    class Panel:
        def __init__(self, content, **kw):
            self.content = content
        @classmethod
        def fit(cls, content, **kw):
            return cls(content, **kw)
        def __str__(self):
            return str(self.content)

    class Progress:
        def __init__(self, *a, **kw): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def add_task(self, *a, **kw): return 0
        def update(self, *a, **kw): pass

    SpinnerColumn = TextColumn = BarColumn = TimeElapsedColumn = TimeRemainingColumn = lambda *a, **k: None
    RichHandler = None


VERSION = "1.0.0"
PROGRAM_NAME = "Video Anonymizer Pro"

console = Console()


def setup_logging(log_level: str) -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    if RICH_AVAILABLE:
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=console, rich_tracebacks=True)],
        )
    else:
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )


def print_banner(config) -> None:
    blur = getattr(config, "blur_mode", "gaussian")
    quality = getattr(config, "quality_mode", "lossless")
    device = getattr(config, "device", "auto")
    multipass = getattr(config, "multipass", False)
    use_sam = getattr(config, "use_sam", False)
    use_vlm = getattr(config, "use_vlm", False)

    if RICH_AVAILABLE:
        panel = Panel.fit(
            f"[bold blue]{PROGRAM_NAME} v{VERSION}[/bold blue]\n"
            f"[dim]Advanced video anonymization[/dim]\n\n"
            f"[bold]Configuration:[/bold]\n"
            f"  Blur: {blur}  |  Quality: {quality}  |  Device: {device}\n"
            f"  Multi-pass: {multipass}  |  SAM: {use_sam}  |  VLM: {use_vlm}",
            title="[bold green]Starting[/bold green]",
        )
        console.print(panel)
    else:
        console.print(
            f"{PROGRAM_NAME} v{VERSION}\n"
            f"  Blur: {blur}  |  Quality: {quality}  |  Device: {device}\n"
            f"  Multi-pass: {multipass}  |  SAM: {use_sam}  |  VLM: {use_vlm}"
        )


def detect_hardware() -> dict:
    try:
        from utils.gpu_manager import GPUManager
        mgr = GPUManager()
        device = mgr.get_device("auto")
        info = {
            "device": device,
            "gpu_available": device.startswith("cuda"),
            "cpu_count": mgr.get_cpu_count(),
            "memory_gb": mgr.get_memory_gb(),
        }
    except Exception:
        info = {"device": "cpu", "gpu_available": False, "cpu_count": 1, "memory_gb": 8}

    if RICH_AVAILABLE:
        table = Table(title="Hardware Detection")
        table.add_column("Component", style="cyan")
        table.add_column("Status", style="green")
        table.add_column("Details", style="yellow")
        table.add_row("GPU",
                       "Available" if info["gpu_available"] else "Not available",
                       f"Using {info['device']}")
        table.add_row("CPU Cores", "Available", f"{info['cpu_count']} cores")
        table.add_row("Memory", "Available", f"{info['memory_gb']:.1f} GB")
        console.print(table)
    else:
        console.print(
            f"  Device: {info['device']}  CPU: {info['cpu_count']} cores  "
            f"RAM: {info['memory_gb']:.1f} GB"
        )
    return info


def analyze_video(input_path: str) -> dict:
    try:
        from core.video_io import VideoReader
        reader = VideoReader(input_path)
        meta = reader.get_metadata()
        video_info = {
            "path": input_path,
            "duration": f"{meta['duration']:.1f}s",
            "resolution": f"{meta['width']}x{meta['height']}",
            "fps": f"{meta['fps_float']:.2f}",
            "codec": meta.get("codec_name", "unknown"),
            "total_frames": meta["total_frames"],
            "size_mb": round(meta["size_bytes"] / (1024 * 1024), 2),
        }
    except Exception as e:
        console.print(f"Warning: Could not analyze video with FFmpeg: {e}")
        video_info = {
            "path": input_path,
            "size_mb": round(Path(input_path).stat().st_size / (1024 * 1024), 2),
        }

    if RICH_AVAILABLE:
        table = Table(title="Video Analysis")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        for k, v in video_info.items():
            table.add_row(k.replace("_", " ").title(), str(v))
        console.print(table)
    else:
        for k, v in video_info.items():
            console.print(f"  {k}: {v}")
    return video_info


def process_videos(pipeline, input_paths: List[str],
                   output_paths: List[str], args) -> None:
    for idx, (inp, outp) in enumerate(zip(input_paths, output_paths)):
        console.print(f"\nProcessing video {idx + 1}/{len(input_paths)}: {Path(inp).name}")
        try:
            start = time.time()
            pipeline.run(inp, outp, args.mode)
            elapsed = time.time() - start

            report = pipeline.get_report()
            console.print(f"  Done in {elapsed:.1f}s")
            console.print(
                f"  Frames: {report.total_frames} total, "
                f"{report.frames_with_blur} anonymized, "
                f"{report.gaps_detected} gaps"
            )
        except Exception as e:
            console.print(f"Error processing {inp}: {e}")
            if args.debug:
                raise


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description="Advanced video anonymization for adult women while preserving children",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py input.mp4 output.mp4
  python main.py input.mp4 output.mp4 --mode full --blur gaussian --use-sam
  python main.py input.mp4 output.mp4 --mode quick
        """,
    )
    parser.add_argument("input", help="Input video file")
    parser.add_argument("output", help="Output video file")
    parser.add_argument("--mode", choices=["full", "auto", "quick", "debug"],
                        default="auto")
    parser.add_argument("--blur", choices=["gaussian", "pixelate", "solid", "mosaic"],
                        default="gaussian")
    parser.add_argument("--quality", choices=["lossless", "high", "medium"],
                        default="lossless")
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--temporal-smoothing", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.4)
    parser.add_argument("--use-sam", action="store_true")
    parser.add_argument("--use-vlm", action="store_true")
    parser.add_argument("--annotate", action="store_true",
                        help="Launch manual annotation GUI")
    parser.add_argument("--annotation-file", type=str)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug-video", type=str)
    parser.add_argument("--report", type=str, help="Save report to JSON")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        default="INFO")
    parser.add_argument("--batch", action="store_true",
                        help="Process multiple videos (glob pattern)")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    setup_logging(args.log_level)

    try:
        from core.pipeline import AnonymizationPipeline
        from config import PipelineConfig
    except ImportError as e:
        console.print(f"Error: Could not import pipeline modules: {e}")
        sys.exit(1)

    config = PipelineConfig(
        blur_mode=args.blur,
        use_manual_annotation=args.annotate,
        use_vlm=args.use_vlm,
        use_sam=args.use_sam,
        multipass=(args.mode in ("full", "debug")),
        quality_mode=args.quality,
        device=args.device,
        temporal_smoothing=args.temporal_smoothing,
        confidence_threshold=args.confidence_threshold,
        debug_output=args.debug_video is not None,
    )

    print_banner(config)
    detect_hardware()

    if args.batch:
        from glob import glob
        input_paths = sorted(glob(args.input))
        if not input_paths:
            console.print(f"No files match: {args.input}")
            sys.exit(1)
        output_paths = [
            f"{Path(p).stem}_anonymized{Path(p).suffix}" for p in input_paths
        ]
    else:
        input_paths = [args.input]
        output_paths = [args.output]

    for p in input_paths:
        if not Path(p).exists():
            console.print(f"Error: Input file not found: {p}")
            sys.exit(1)

    analyze_video(input_paths[0])

    try:
        pipeline = AnonymizationPipeline(config)
    except Exception as e:
        console.print(f"Error initializing pipeline: {e}")
        sys.exit(1)

    process_videos(pipeline, input_paths, output_paths, args)

    if args.report:
        report = pipeline.get_report()
        try:
            with open(args.report, "w") as f:
                json.dump(report.__dict__, f, indent=2, default=str)
            console.print(f"Report saved to: {args.report}")
        except Exception as e:
            console.print(f"Error saving report: {e}")

    console.print("\nAll processing complete!")


if __name__ == "__main__":
    main()
