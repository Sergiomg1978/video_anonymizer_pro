from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict
import numpy as np
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class AnchorFrame:
    frame_number: int
    timestamp: float
    head_bbox: Optional[Tuple[int, int, int, int]]  # None if woman not present
    woman_present: bool
    face_embedding: Optional[np.ndarray] = None
    body_embedding: Optional[np.ndarray] = None
    annotated_by: str = "manual"  # "manual" or "auto_confirmed"

class AnchorFrameManager:
    """Manages manually annotated anchor frames."""

    def __init__(self):
        self.anchors: List[AnchorFrame] = []

    def add_anchor(self, anchor: AnchorFrame):
        """Add anchor, maintaining order by frame_number."""
        self.anchors.append(anchor)
        self.anchors.sort(key=lambda a: a.frame_number)

    def get_anchors(self) -> List[AnchorFrame]:
        return list(self.anchors)

    def get_reference_embeddings(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Average all face embeddings from anchors where woman_present=True."""
        face_embs = [a.face_embedding for a in self.anchors
                     if a.woman_present and a.face_embedding is not None]
        body_embs = [a.body_embedding for a in self.anchors
                     if a.woman_present and a.body_embedding is not None]

        avg_face = np.mean(face_embs, axis=0) if face_embs else None
        avg_body = np.mean(body_embs, axis=0) if body_embs else None
        return avg_face, avg_body

    def get_nearest_anchors(self, frame_number: int) -> Tuple[Optional[AnchorFrame], Optional[AnchorFrame]]:
        """Get closest anchors before and after frame_number."""
        before = None
        after = None
        for a in self.anchors:
            if a.frame_number <= frame_number:
                before = a
            elif after is None:
                after = a
                break
        return before, after

    def interpolate_bbox(self, frame_number: int) -> Optional[Tuple[int, int, int, int]]:
        """Linearly interpolate head bbox between nearest anchors."""
        before, after = self.get_nearest_anchors(frame_number)
        if not before or not after:
            return None
        if not before.woman_present or not after.woman_present:
            return None
        if not before.head_bbox or not after.head_bbox:
            return None

        total = after.frame_number - before.frame_number
        if total <= 0:
            return before.head_bbox

        progress = (frame_number - before.frame_number) / total
        b1, b2 = before.head_bbox, after.head_bbox
        return tuple(int(b1[i] + progress * (b2[i] - b1[i])) for i in range(4))

    def save_to_file(self, filepath: str):
        """Save anchors to JSON."""
        data = []
        for a in self.anchors:
            entry = {
                'frame_number': a.frame_number,
                'timestamp': a.timestamp,
                'head_bbox': list(a.head_bbox) if a.head_bbox else None,
                'woman_present': a.woman_present,
                'face_embedding': a.face_embedding.tolist() if a.face_embedding is not None else None,
                'body_embedding': a.body_embedding.tolist() if a.body_embedding is not None else None,
                'annotated_by': a.annotated_by,
            }
            data.append(entry)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(data)} anchors to {filepath}")

    def load_from_file(self, filepath: str):
        """Load anchors from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.anchors = []
        for entry in data:
            anchor = AnchorFrame(
                frame_number=entry['frame_number'],
                timestamp=entry['timestamp'],
                head_bbox=tuple(entry['head_bbox']) if entry['head_bbox'] else None,
                woman_present=entry['woman_present'],
                face_embedding=np.array(entry['face_embedding']) if entry['face_embedding'] else None,
                body_embedding=np.array(entry['body_embedding']) if entry['body_embedding'] else None,
                annotated_by=entry.get('annotated_by', 'manual'),
            )
            self.anchors.append(anchor)
        logger.info(f"Loaded {len(self.anchors)} anchors from {filepath}")

    def get_woman_present_ratio(self) -> float:
        if not self.anchors:
            return 0.0
        return sum(1 for a in self.anchors if a.woman_present) / len(self.anchors)

    def get_annotated_count(self) -> int:
        return len(self.anchors)
