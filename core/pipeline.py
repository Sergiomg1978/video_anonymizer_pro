"""
Main pipeline orchestrator for Video Anonymizer Pro.
Coordinates detection, tracking, classification, and anonymization.
"""

import time
import logging
from types import SimpleNamespace
from typing import Optional, Callable, List

logger = logging.getLogger(__name__)

# All imports are guarded so the module loads even when deps are missing.
try:
    from core.video_io import VideoReader, VideoWriter
except ImportError as e:
    logger.warning(f"core.video_io not available: {e}")
    VideoReader = VideoWriter = None

try:
    from config import PipelineConfig
except ImportError:
    PipelineConfig = None

try:
    from detection.face_detector import MultiFaceDetector
except ImportError:
    MultiFaceDetector = None

try:
    from detection.person_detector import PersonDetector
except ImportError:
    PersonDetector = None

try:
    from detection.head_detector import HeadDetector
except ImportError:
    HeadDetector = None

try:
    from detection.age_gender_classifier import AgeGenderClassifier
except ImportError:
    AgeGenderClassifier = None

try:
    from tracking.deep_sort_tracker import PersonTracker
except ImportError:
    PersonTracker = None

try:
    from tracking.identity_manager import IdentityManager
except ImportError:
    IdentityManager = None

try:
    from anonymization.mask_generator import MaskGenerator
except ImportError:
    MaskGenerator = None

try:
    from anonymization.blur_engine import BlurEngine
except ImportError:
    BlurEngine = None

try:
    from multipass.forward_pass import ForwardPass
except ImportError:
    ForwardPass = None

try:
    from multipass.backward_pass import BackwardPass
except ImportError:
    BackwardPass = None

try:
    from multipass.confidence_merger import ConfidenceMerger
except ImportError:
    ConfidenceMerger = None

try:
    from multipass.gap_filler import GapFiller
except ImportError:
    GapFiller = None

try:
    from annotation.manual_annotator import ManualAnnotator
except ImportError:
    ManualAnnotator = None

try:
    from annotation.anchor_frames import AnchorFrameManager
except ImportError:
    AnchorFrameManager = None

try:
    from scene_analysis.scene_interpreter import SceneInterpreter
except ImportError:
    SceneInterpreter = None

try:
    from utils.gpu_manager import GPUManager
except ImportError:
    GPUManager = None


class AnonymizationPipeline:
    """Orchestrates the full video anonymization workflow."""

    def __init__(self, config):
        self.config = config
        self.device = "cpu"
        self._report = None

        # Models (lazy loaded)
        self._face_detector = None
        self._person_detector = None
        self._head_detector = None
        self._classifier = None
        self._tracker = None
        self._identity_manager = None
        self._mask_generator = None
        self._blur_engine = None
        self._scene_interpreter = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        input_path: str,
        output_path: str,
        mode: str = "auto",
        progress_callback: Optional[Callable] = None,
    ):
        """Run the full anonymization pipeline.

        Args:
            input_path: Path to input video.
            output_path: Path to write the anonymized video.
            mode: Processing mode (quick / auto / full / debug).
            progress_callback: ``callback(current, total, stage)``
        """
        start_time = time.time()

        # 1. Open video
        logger.info(f"Opening video: {input_path}")
        reader = VideoReader(input_path)
        metadata = reader.get_metadata()
        total_frames = reader.get_total_frames()
        logger.info(
            f"Video: {metadata['width']}x{metadata['height']} "
            f"@ {metadata['fps_float']:.2f} fps, {total_frames} frames"
        )

        # 2. Resolve device
        self._resolve_device()

        # 3. Initialize models
        self._init_models(mode)

        # 4. Optional manual annotation (full mode)
        anchor_manager = AnchorFrameManager() if AnchorFrameManager else None
        if (
            mode == "full"
            and getattr(self.config, "use_manual_annotation", False)
            and ManualAnnotator is not None
        ):
            logger.info("Launching manual annotation GUI...")
            annotator = ManualAnnotator(input_path)
            anchors = annotator.launch_gui()
            if anchor_manager:
                for a in anchors:
                    anchor_manager.add_anchor(a)
            logger.info(f"Received {len(anchors)} anchor frames from annotation")

        anchor_frames = anchor_manager.get_anchors() if anchor_manager else []

        # 5. Forward pass
        logger.info("Running forward detection pass...")
        forward_results = self._run_forward_pass(
            reader, anchor_frames, progress_callback
        )

        # 6. Optional backward pass + merge
        if getattr(self.config, "multipass", False) and mode in ("full", "auto", "debug"):
            logger.info("Running backward detection pass...")
            backward_results = self._run_backward_pass(
                reader, anchor_frames, forward_results, progress_callback
            )
            logger.info("Merging forward and backward results...")
            merged = self._merge_results(forward_results, backward_results)
        else:
            merged = self._forward_to_merged(forward_results)

        # 7. Gap filling
        logger.info("Filling detection gaps...")
        final_results = self._fill_gaps(merged)

        # 8-9. Apply anonymization and write output
        logger.info("Applying anonymization and writing output...")
        writer = VideoWriter(output_path, metadata, self.config.quality_mode)
        self._apply_anonymization(reader, writer, final_results, progress_callback)
        writer.finalize()

        # 10. Build report
        elapsed = time.time() - start_time
        frames_with_blur = sum(
            1 for r in final_results if r.get("bbox") is not None
        )
        confidences = [
            r["confidence"] for r in final_results if r.get("confidence", 0) > 0
        ]
        self._report = SimpleNamespace(
            total_frames=total_frames,
            frames_with_blur=frames_with_blur,
            gaps_detected=sum(
                1 for r in final_results if r.get("fill_method") == "gap"
            ),
            average_confidence=(
                sum(confidences) / len(confidences) if confidences else 0.0
            ),
            processing_time=elapsed,
            quality_psnr=0.0,
            quality_ssim=0.0,
        )
        logger.info(
            f"Done in {elapsed:.1f}s — "
            f"{frames_with_blur}/{total_frames} frames anonymized"
        )

    def get_report(self):
        if self._report is None:
            return SimpleNamespace(
                total_frames=0,
                frames_with_blur=0,
                gaps_detected=0,
                average_confidence=0.0,
                processing_time=0.0,
                quality_psnr=0.0,
                quality_ssim=0.0,
            )
        return self._report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_device(self):
        device_pref = getattr(self.config, "device", "auto")
        if GPUManager is not None:
            mgr = GPUManager()
            self.device = mgr.get_device(device_pref)
        else:
            self.device = "cpu"
        logger.info(f"Using device: {self.device}")

    def _init_models(self, mode: str):
        """Initialize AI models based on mode and config."""
        conf_thresh = getattr(self.config, "confidence_threshold", 0.4)

        if MultiFaceDetector is not None:
            try:
                self._face_detector = MultiFaceDetector(
                    device=self.device, confidence_threshold=conf_thresh
                )
            except Exception as e:
                logger.warning(f"Failed to init face detector: {e}")

        if PersonDetector is not None:
            try:
                self._person_detector = PersonDetector(
                    device=self.device, confidence_threshold=0.5
                )
            except Exception as e:
                logger.warning(f"Failed to init person detector: {e}")

        if HeadDetector is not None:
            try:
                self._head_detector = HeadDetector(
                    device=self.device, confidence_threshold=0.4
                )
            except Exception as e:
                logger.warning(f"Failed to init head detector: {e}")

        if AgeGenderClassifier is not None:
            try:
                self._classifier = AgeGenderClassifier(device=self.device)
            except Exception as e:
                logger.warning(f"Failed to init classifier: {e}")

        if PersonTracker is not None:
            try:
                self._tracker = PersonTracker()
            except Exception as e:
                logger.warning(f"Failed to init tracker: {e}")

        if IdentityManager is not None:
            try:
                self._identity_manager = IdentityManager()
            except Exception as e:
                logger.warning(f"Failed to init identity manager: {e}")

        if BlurEngine is not None:
            blur_mode = getattr(self.config, "blur_mode", "gaussian")
            temporal = getattr(self.config, "temporal_smoothing", 5)
            self._blur_engine = BlurEngine(mode=blur_mode, temporal_smoothing=temporal)

        use_sam = getattr(self.config, "use_sam", False) and mode != "quick"
        if MaskGenerator is not None and use_sam:
            try:
                self._mask_generator = MaskGenerator(
                    model="sam2", device=self.device
                )
            except Exception as e:
                logger.warning(f"Failed to init SAM mask generator: {e}")

        if SceneInterpreter is not None:
            use_vlm = getattr(self.config, "use_vlm", False)
            try:
                self._scene_interpreter = SceneInterpreter(
                    use_vlm=use_vlm, device=self.device
                )
            except Exception as e:
                logger.warning(f"Failed to init scene interpreter: {e}")

    # ------------------------------------------------------------------
    # Forward / backward passes
    # ------------------------------------------------------------------

    def _run_forward_pass(self, reader, anchor_frames, progress_callback):
        """Iterate frames forward, detect and track persons."""
        results = []
        total = reader.get_total_frames()

        for frame_num, frame in reader.iterate_frames():
            result = self._process_single_frame(frame_num, frame)
            results.append(result)

            if progress_callback and frame_num % 10 == 0:
                progress_callback(frame_num, total, "forward_pass")

        return results

    def _run_backward_pass(self, reader, anchor_frames, forward_results, cb):
        """Iterate frames backward for a second pass."""
        # Reset tracker for backward pass
        if self._tracker:
            self._tracker.reset()

        results_reversed = []
        total = reader.get_total_frames()

        for frame_num in range(total - 1, -1, -1):
            try:
                frame = reader.read_frame(frame_num)
            except Exception:
                results_reversed.append(
                    {"frame": frame_num, "bbox": None, "confidence": 0.0}
                )
                continue

            result = self._process_single_frame(frame_num, frame)
            results_reversed.append(result)

            if cb and frame_num % 10 == 0:
                cb(total - frame_num, total, "backward_pass")

        results_reversed.reverse()
        return results_reversed

    def _process_single_frame(self, frame_num, frame):
        """Run detection + tracking + classification on a single frame."""
        bbox = None
        confidence = 0.0

        # Face detection
        face_detections = []
        if self._face_detector:
            try:
                face_detections = self._face_detector.detect(frame)
            except Exception as e:
                logger.debug(f"Face detection failed frame {frame_num}: {e}")

        # Person detection
        person_detections = []
        if self._person_detector:
            try:
                person_detections = self._person_detector.detect(frame)
            except Exception as e:
                logger.debug(f"Person detection failed frame {frame_num}: {e}")

        # Head detection
        head_detections = []
        if self._head_detector:
            try:
                head_detections = self._head_detector.detect(
                    frame, face_detections=face_detections
                )
            except Exception as e:
                logger.debug(f"Head detection failed frame {frame_num}: {e}")

        # Use tracking if available
        if self._tracker and (face_detections or person_detections):
            try:
                all_dets = []
                for fd in face_detections:
                    all_dets.append(fd)
                for pd in person_detections:
                    all_dets.append(pd)
                tracks = self._tracker.update(frame, all_dets)

                # Classify and find the target woman
                for track in tracks:
                    if self._classifier and getattr(track, "person_bbox", None):
                        try:
                            cls = self._classifier.classify(
                                frame,
                                track.person_bbox,
                                getattr(track, "face_bbox", None),
                            )
                            if cls.is_adult and cls.is_female and cls.confidence > confidence:
                                # Use head bbox if available, else face, else person
                                bbox = (
                                    getattr(track, "head_bbox", None)
                                    or getattr(track, "face_bbox", None)
                                    or track.person_bbox
                                )
                                confidence = cls.confidence
                        except Exception:
                            pass
                    elif not self._classifier:
                        # Without classifier, use largest detection
                        tb = getattr(track, "person_bbox", None) or getattr(track, "face_bbox", None)
                        if tb:
                            area = (tb[2] - tb[0]) * (tb[3] - tb[1])
                            if area > 0 and track.confidence > confidence:
                                bbox = tb
                                confidence = track.confidence
            except Exception as e:
                logger.debug(f"Tracking failed frame {frame_num}: {e}")

        # Fallback: use raw detections if tracking unavailable
        if bbox is None and (face_detections or head_detections):
            best = None
            best_conf = 0.0
            for det in head_detections or face_detections:
                det_bbox = getattr(det, "bbox", None)
                det_conf = getattr(det, "confidence", 0.0)
                if det_bbox and det_conf > best_conf:
                    best = det_bbox
                    best_conf = det_conf
            if best:
                bbox = best
                confidence = best_conf

        return {
            "frame": frame_num,
            "bbox": tuple(bbox) if bbox else None,
            "confidence": confidence,
        }

    # ------------------------------------------------------------------
    # Merge & gap fill
    # ------------------------------------------------------------------

    def _merge_results(self, forward, backward):
        """Merge forward and backward pass results."""
        merged = []
        for fwd, bwd in zip(forward, backward):
            fb = fwd.get("bbox")
            bb = bwd.get("bbox")
            fc = fwd.get("confidence", 0)
            bc = bwd.get("confidence", 0)

            if fb and bb:
                if fc >= bc:
                    merged.append({**fwd, "fill_method": "detected"})
                else:
                    merged.append({**bwd, "fill_method": "detected"})
            elif fb:
                merged.append({**fwd, "fill_method": "detected"})
            elif bb:
                merged.append({**bwd, "fill_method": "detected"})
            else:
                merged.append({
                    "frame": fwd["frame"],
                    "bbox": None,
                    "confidence": 0.0,
                    "fill_method": "gap",
                })
        return merged

    def _forward_to_merged(self, forward):
        """Convert forward-only results to merged format."""
        merged = []
        for r in forward:
            method = "detected" if r.get("bbox") else "gap"
            merged.append({**r, "fill_method": method})
        return merged

    def _fill_gaps(self, merged):
        """Fill short gaps via linear interpolation."""
        max_gap = getattr(self.config, "interpolation_max_gap", 10)
        results = list(merged)
        n = len(results)

        i = 0
        while i < n:
            if results[i].get("fill_method") == "gap":
                gap_start = i
                while i < n and results[i].get("fill_method") == "gap":
                    i += 1
                gap_end = i  # exclusive

                gap_len = gap_end - gap_start
                before = results[gap_start - 1] if gap_start > 0 else None
                after = results[gap_end] if gap_end < n else None

                if (
                    gap_len <= max_gap
                    and before
                    and before.get("bbox")
                    and after
                    and after.get("bbox")
                ):
                    b1, b2 = before["bbox"], after["bbox"]
                    for j in range(gap_start, gap_end):
                        t = (j - gap_start + 1) / (gap_len + 1)
                        interp = tuple(
                            int(b1[k] + t * (b2[k] - b1[k])) for k in range(4)
                        )
                        results[j] = {
                            "frame": results[j]["frame"],
                            "bbox": interp,
                            "confidence": 0.5,
                            "fill_method": "interpolated",
                        }
            else:
                i += 1

        return results

    # ------------------------------------------------------------------
    # Anonymization pass
    # ------------------------------------------------------------------

    def _apply_anonymization(self, reader, writer, final_results, cb):
        """Read frames, apply blur where needed, write output."""
        total = len(final_results)
        result_map = {r["frame"]: r for r in final_results}

        for frame_num, frame in reader.iterate_frames():
            result = result_map.get(frame_num)
            if result and result.get("bbox"):
                bbox = result["bbox"]
                try:
                    if self._mask_generator:
                        mask = self._mask_generator.generate_mask(frame, bbox)
                    else:
                        mask = self._fallback_mask(frame, bbox)

                    if self._blur_engine:
                        frame = self._blur_engine.anonymize_frame(frame, mask)
                    else:
                        frame = self._simple_blur(frame, mask)
                except Exception as e:
                    logger.warning(
                        f"Anonymization failed frame {frame_num}: {e}"
                    )

            writer.write_frame(frame)

            if cb and frame_num % 10 == 0:
                cb(frame_num, total, "writing")

    def _fallback_mask(self, frame, bbox):
        """Generate a simple elliptical mask from a bbox."""
        import numpy as np
        import cv2

        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.float32)
        x1, y1, x2, y2 = bbox

        # Expand by 15%
        bw, bh = x2 - x1, y2 - y1
        ex, ey = int(bw * 0.15), int(bh * 0.15)
        x1, y1 = max(0, x1 - ex), max(0, y1 - ey)
        x2, y2 = min(w, x2 + ex), min(h, y2 + ey)

        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ax, ay = (x2 - x1) // 2, (y2 - y1) // 2
        cv2.ellipse(mask, (cx, cy), (ax, ay), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (11, 11), 0)
        return np.clip(mask, 0.0, 1.0)

    def _simple_blur(self, frame, mask):
        """Apply gaussian blur using only the mask (no BlurEngine)."""
        import numpy as np
        import cv2

        blurred = cv2.GaussianBlur(frame, (51, 51), 30)
        mask_3d = mask[:, :, np.newaxis]
        result = (frame * (1 - mask_3d) + blurred * mask_3d).astype(np.uint8)
        return result
