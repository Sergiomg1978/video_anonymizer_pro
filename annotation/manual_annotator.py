"""
Manual annotation GUI for selecting the target woman in anchor frames.
Uses PyQt6 for the interface with fallback when not available.
"""

import logging
import numpy as np
from typing import List, Optional, Tuple
from pathlib import Path

from .anchor_frames import AnchorFrame, AnchorFrameManager

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QSlider, QStatusBar, QGroupBox, QMessageBox,
    )
    from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor, QFont
    from PyQt6.QtCore import Qt, QRect, QPoint
    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    logger.warning("PyQt6 not available. Manual annotation GUI disabled.")


class ManualAnnotator:
    """
    GUI for manually annotating anchor frames to identify the target woman.

    Workflow:
    1. Extract representative keyframes from the video.
    2. Pre-run face/person detectors on each keyframe.
    3. Show the user each frame with overlaid detections.
    4. User clicks a detection to confirm it as the woman, draws a bbox,
       or marks her as absent.
    5. Returns a list of AnchorFrames with embeddings.
    """

    def __init__(self, video_path: str, auto_extract_interval: float = 2.0):
        self.video_path = video_path
        self.auto_extract_interval = auto_extract_interval
        self._reader = None
        self._keyframe_numbers: List[int] = []
        self._keyframes: List[Tuple[int, np.ndarray]] = []
        self._detections: dict = {}

    def _get_reader(self):
        if self._reader is None:
            from core.video_io import VideoReader
            self._reader = VideoReader(self.video_path)
        return self._reader

    def auto_extract_keyframes(self) -> List[int]:
        """Extract representative frame numbers at regular intervals."""
        reader = self._get_reader()
        total = reader.get_total_frames()
        fps = float(reader.get_fps())
        interval_frames = max(1, int(fps * self.auto_extract_interval))

        frames = [0]
        for i in range(interval_frames, total, interval_frames):
            frames.append(i)
        if total > 1 and frames[-1] != total - 1:
            frames.append(total - 1)

        self._keyframe_numbers = frames
        logger.info(f"Extracted {len(frames)} keyframe positions")
        return frames

    def _pre_detect(self, frames: List[Tuple[int, np.ndarray]]) -> dict:
        """Run detectors on extracted frames."""
        detections = {}
        try:
            from detection.face_detector import MultiFaceDetector
            face_det = MultiFaceDetector(device="cpu", confidence_threshold=0.3)
        except Exception:
            face_det = None

        try:
            from detection.person_detector import PersonDetector
            person_det = PersonDetector(device="cpu", confidence_threshold=0.4)
        except Exception:
            person_det = None

        for frame_num, frame in frames:
            entry = {"faces": [], "persons": []}
            if face_det:
                try:
                    faces = face_det.detect(frame)
                    entry["faces"] = [
                        {"bbox": f.bbox, "confidence": f.confidence} for f in faces
                    ]
                except Exception:
                    pass
            if person_det:
                try:
                    persons = person_det.detect(frame)
                    entry["persons"] = [
                        {"bbox": p.bbox, "confidence": p.confidence} for p in persons
                    ]
                except Exception:
                    pass
            detections[frame_num] = entry

        self._detections = detections
        return detections

    def launch_gui(self) -> List[AnchorFrame]:
        """Launch the PyQt6 annotation window. Returns anchors when closed."""
        if not PYQT6_AVAILABLE:
            logger.error(
                "PyQt6 is not installed. Cannot launch annotation GUI. "
                "Install with: pip install PyQt6"
            )
            return self._quick_annotate_fallback()

        # Extract keyframes
        if not self._keyframe_numbers:
            self.auto_extract_keyframes()

        reader = self._get_reader()
        self._keyframes = []
        for fn in self._keyframe_numbers:
            try:
                frame = reader.read_frame(fn)
                self._keyframes.append((fn, frame))
            except Exception as e:
                logger.warning(f"Failed to read frame {fn}: {e}")

        if not self._keyframes:
            logger.error("No keyframes could be read")
            return []

        # Pre-detect
        self._pre_detect(self._keyframes)

        # Launch Qt app
        app = QApplication.instance() or QApplication([])
        fps = float(reader.get_fps())
        window = _AnnotationWindow(
            self._keyframes, self._detections, fps
        )
        window.show()
        app.exec()

        return window.get_anchors()

    def _quick_annotate_fallback(self) -> List[AnchorFrame]:
        """Automatic quick annotation: assume largest person is the woman."""
        if not self._keyframe_numbers:
            self.auto_extract_keyframes()

        reader = self._get_reader()
        fps = float(reader.get_fps())
        anchors = []

        for fn in self._keyframe_numbers:
            try:
                frame = reader.read_frame(fn)
            except Exception:
                continue

            dets = self._detections.get(fn, {})
            persons = dets.get("persons", [])

            if persons:
                # Pick largest person by area
                largest = max(
                    persons,
                    key=lambda p: (p["bbox"][2] - p["bbox"][0])
                    * (p["bbox"][3] - p["bbox"][1]),
                )
                anchor = AnchorFrame(
                    frame_number=fn,
                    timestamp=fn / fps,
                    head_bbox=tuple(largest["bbox"]),
                    woman_present=True,
                    annotated_by="auto_confirmed",
                )
            else:
                anchor = AnchorFrame(
                    frame_number=fn,
                    timestamp=fn / fps,
                    head_bbox=None,
                    woman_present=False,
                    annotated_by="auto_confirmed",
                )
            anchors.append(anchor)

        logger.info(f"Quick annotation produced {len(anchors)} anchors")
        return anchors

    def quick_annotate(
        self,
        frames: List[Tuple[int, np.ndarray]],
        detections: dict,
    ) -> List[AnchorFrame]:
        """Public quick-annotate entry point."""
        fps = float(self._get_reader().get_fps())
        anchors = []
        for fn, frame in frames:
            dets = detections.get(fn, {})
            persons = dets.get("persons", [])
            if persons:
                largest = max(
                    persons,
                    key=lambda p: (p["bbox"][2] - p["bbox"][0])
                    * (p["bbox"][3] - p["bbox"][1]),
                )
                anchors.append(
                    AnchorFrame(
                        frame_number=fn,
                        timestamp=fn / fps,
                        head_bbox=tuple(largest["bbox"]),
                        woman_present=True,
                        annotated_by="auto_confirmed",
                    )
                )
            else:
                anchors.append(
                    AnchorFrame(
                        frame_number=fn,
                        timestamp=fn / fps,
                        head_bbox=None,
                        woman_present=False,
                        annotated_by="auto_confirmed",
                    )
                )
        return anchors


# ---------------------------------------------------------------------------
# PyQt6 annotation window
# ---------------------------------------------------------------------------

if PYQT6_AVAILABLE:

    class _AnnotationWindow(QMainWindow):
        """Main annotation window."""

        def __init__(
            self,
            keyframes: List[Tuple[int, np.ndarray]],
            detections: dict,
            fps: float,
        ):
            super().__init__()
            self.keyframes = keyframes
            self.detections = detections
            self.fps = fps
            self.current_idx = 0
            self.annotations: dict = {}  # frame_number -> dict
            self._draw_mode = False
            self._draw_start: Optional[QPoint] = None
            self._draw_rect: Optional[QRect] = None

            self.setWindowTitle("Video Anonymizer Pro - Manual Annotation")
            self.setMinimumSize(1024, 700)

            self._build_ui()
            self._show_frame(0)

        # -- UI construction --------------------------------------------------

        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QHBoxLayout(central)

            # Left: image
            left = QVBoxLayout()
            self.image_label = QLabel()
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setMinimumSize(640, 480)
            self.image_label.mousePressEvent = self._on_image_click
            self.image_label.mouseMoveEvent = self._on_image_move
            self.image_label.mouseReleaseEvent = self._on_image_release
            left.addWidget(self.image_label, stretch=1)

            # Navigation bar
            nav = QHBoxLayout()
            self.btn_prev = QPushButton("< Prev")
            self.btn_prev.clicked.connect(self._prev_frame)
            nav.addWidget(self.btn_prev)

            self.slider = QSlider(Qt.Orientation.Horizontal)
            self.slider.setMinimum(0)
            self.slider.setMaximum(max(0, len(self.keyframes) - 1))
            self.slider.valueChanged.connect(self._on_slider)
            nav.addWidget(self.slider, stretch=1)

            self.btn_next = QPushButton("Next >")
            self.btn_next.clicked.connect(self._next_frame)
            nav.addWidget(self.btn_next)
            left.addLayout(nav)

            main_layout.addLayout(left, stretch=3)

            # Right: info + actions
            right = QVBoxLayout()

            info_box = QGroupBox("Info")
            info_layout = QVBoxLayout()
            self.lbl_frame = QLabel("Frame: 0 / 0")
            self.lbl_timestamp = QLabel("Time: 0.00s")
            self.lbl_annotated = QLabel("Annotated: 0 / 0")
            info_layout.addWidget(self.lbl_frame)
            info_layout.addWidget(self.lbl_timestamp)
            info_layout.addWidget(self.lbl_annotated)
            info_box.setLayout(info_layout)
            right.addWidget(info_box)

            actions_box = QGroupBox("Actions")
            actions_layout = QVBoxLayout()

            self.btn_draw = QPushButton("Draw Mode (D)")
            self.btn_draw.setCheckable(True)
            self.btn_draw.clicked.connect(self._toggle_draw)
            actions_layout.addWidget(self.btn_draw)

            btn_not_present = QPushButton("Woman Not Present (N)")
            btn_not_present.clicked.connect(self._mark_not_present)
            actions_layout.addWidget(btn_not_present)

            btn_quick = QPushButton("Quick Annotate All")
            btn_quick.clicked.connect(self._quick_annotate_all)
            actions_layout.addWidget(btn_quick)

            btn_save = QPushButton("Save && Exit (Q)")
            btn_save.clicked.connect(self._save_and_exit)
            actions_layout.addWidget(btn_save)

            actions_box.setLayout(actions_layout)
            right.addWidget(actions_box)
            right.addStretch()

            main_layout.addLayout(right, stretch=1)

            self.statusBar().showMessage("Click a detection to select, or press D to draw.")

        # -- Display -----------------------------------------------------------

        def _show_frame(self, idx: int):
            if idx < 0 or idx >= len(self.keyframes):
                return
            self.current_idx = idx
            fn, frame = self.keyframes[idx]

            # Draw detections
            display = frame.copy()
            dets = self.detections.get(fn, {})

            for face in dets.get("faces", []):
                x1, y1, x2, y2 = face["bbox"]
                cv2 = _get_cv2()
                if cv2:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 100, 0), 2)

            for person in dets.get("persons", []):
                x1, y1, x2, y2 = person["bbox"]
                cv2 = _get_cv2()
                if cv2:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 200, 0), 2)

            # Draw current annotation if exists
            ann = self.annotations.get(fn)
            if ann and ann.get("head_bbox"):
                x1, y1, x2, y2 = ann["head_bbox"]
                cv2 = _get_cv2()
                if cv2:
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 3)

            # Convert to QPixmap
            h, w, ch = display.shape
            bytes_per_line = ch * w
            img = QImage(display.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(img)

            # Scale to fit label
            label_size = self.image_label.size()
            scaled = pixmap.scaled(
                label_size,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.image_label.setPixmap(scaled)
            self._pixmap_scale = (
                w / scaled.width() if scaled.width() > 0 else 1,
                h / scaled.height() if scaled.height() > 0 else 1,
            )
            self._pixmap_offset = (
                (label_size.width() - scaled.width()) // 2,
                (label_size.height() - scaled.height()) // 2,
            )

            # Update UI
            self.slider.blockSignals(True)
            self.slider.setValue(idx)
            self.slider.blockSignals(False)
            self.lbl_frame.setText(
                f"Frame: {idx + 1} / {len(self.keyframes)}  (#{fn})"
            )
            self.lbl_timestamp.setText(f"Time: {fn / self.fps:.2f}s")
            annotated = sum(1 for a in self.annotations.values() if a)
            self.lbl_annotated.setText(
                f"Annotated: {annotated} / {len(self.keyframes)}"
            )

        # -- Events ------------------------------------------------------------

        def _to_frame_coords(self, pos) -> Tuple[int, int]:
            """Convert widget coords to frame pixel coords."""
            ox, oy = self._pixmap_offset
            sx, sy = self._pixmap_scale
            x = int((pos.x() - ox) * sx)
            y = int((pos.y() - oy) * sy)
            return x, y

        def _on_image_click(self, event):
            fn, frame = self.keyframes[self.current_idx]
            x, y = self._to_frame_coords(event.pos())

            if self._draw_mode:
                self._draw_start = event.pos()
                self._draw_rect = None
                return

            # Check if click is inside a detection bbox
            dets = self.detections.get(fn, {})
            all_dets = dets.get("faces", []) + dets.get("persons", [])

            for det in all_dets:
                bx1, by1, bx2, by2 = det["bbox"]
                if bx1 <= x <= bx2 and by1 <= y <= by2:
                    self.annotations[fn] = {
                        "head_bbox": tuple(det["bbox"]),
                        "woman_present": True,
                    }
                    self.statusBar().showMessage(
                        f"Selected bbox at ({bx1},{by1})-({bx2},{by2})"
                    )
                    self._show_frame(self.current_idx)
                    return

            self.statusBar().showMessage("No detection at click position")

        def _on_image_move(self, event):
            if self._draw_mode and self._draw_start:
                self._draw_rect = QRect(self._draw_start, event.pos())

        def _on_image_release(self, event):
            if self._draw_mode and self._draw_start:
                fn, frame = self.keyframes[self.current_idx]
                x1, y1 = self._to_frame_coords(self._draw_start)
                x2, y2 = self._to_frame_coords(event.pos())
                # Normalize
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    self.annotations[fn] = {
                        "head_bbox": (x1, y1, x2, y2),
                        "woman_present": True,
                    }
                    self.statusBar().showMessage(
                        f"Drew bbox ({x1},{y1})-({x2},{y2})"
                    )
                    self._show_frame(self.current_idx)

                self._draw_start = None
                self._draw_rect = None

        def keyPressEvent(self, event):
            key = event.key()
            if key == Qt.Key.Key_Right:
                self._next_frame()
            elif key == Qt.Key.Key_Left:
                self._prev_frame()
            elif key == Qt.Key.Key_Return:
                self.statusBar().showMessage("Annotation confirmed")
                self._next_frame()
            elif key == Qt.Key.Key_N:
                self._mark_not_present()
            elif key == Qt.Key.Key_D:
                self._toggle_draw()
            elif key == Qt.Key.Key_Q:
                self._save_and_exit()

        # -- Actions -----------------------------------------------------------

        def _next_frame(self):
            if self.current_idx < len(self.keyframes) - 1:
                self._show_frame(self.current_idx + 1)

        def _prev_frame(self):
            if self.current_idx > 0:
                self._show_frame(self.current_idx - 1)

        def _on_slider(self, val):
            self._show_frame(val)

        def _toggle_draw(self):
            self._draw_mode = not self._draw_mode
            self.btn_draw.setChecked(self._draw_mode)
            mode_str = "ON" if self._draw_mode else "OFF"
            self.statusBar().showMessage(f"Draw mode {mode_str}")

        def _mark_not_present(self):
            fn, _ = self.keyframes[self.current_idx]
            self.annotations[fn] = {
                "head_bbox": None,
                "woman_present": False,
            }
            self.statusBar().showMessage("Marked: woman not present")
            self._show_frame(self.current_idx)
            self._next_frame()

        def _quick_annotate_all(self):
            """Auto-annotate: largest person detection = the woman."""
            for idx, (fn, frame) in enumerate(self.keyframes):
                if fn in self.annotations:
                    continue
                dets = self.detections.get(fn, {})
                persons = dets.get("persons", [])
                if persons:
                    largest = max(
                        persons,
                        key=lambda p: (p["bbox"][2] - p["bbox"][0])
                        * (p["bbox"][3] - p["bbox"][1]),
                    )
                    self.annotations[fn] = {
                        "head_bbox": tuple(largest["bbox"]),
                        "woman_present": True,
                    }
                else:
                    self.annotations[fn] = {
                        "head_bbox": None,
                        "woman_present": False,
                    }
            self.statusBar().showMessage(
                f"Quick annotated {len(self.keyframes)} frames"
            )
            self._show_frame(self.current_idx)

        def _save_and_exit(self):
            self.close()

        # -- Results -----------------------------------------------------------

        def get_anchors(self) -> List[AnchorFrame]:
            anchors = []
            for fn, _ in self.keyframes:
                ann = self.annotations.get(fn)
                if ann is None:
                    continue
                anchors.append(
                    AnchorFrame(
                        frame_number=fn,
                        timestamp=fn / self.fps,
                        head_bbox=ann.get("head_bbox"),
                        woman_present=ann.get("woman_present", False),
                        annotated_by="manual",
                    )
                )
            return anchors


def _get_cv2():
    """Lazy import cv2."""
    try:
        import cv2
        return cv2
    except ImportError:
        return None
