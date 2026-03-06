"""GPU and system resource management for Video Anonymizer Pro."""

from __future__ import annotations

import logging
import os
from typing import Dict

log = logging.getLogger(__name__)


class GPUManager:
    """Detect and report GPU / CPU / memory resources."""

    # ------------------------------------------------------------------
    # Device selection
    # ------------------------------------------------------------------
    @staticmethod
    def get_device(preference: str = "auto") -> str:
        """Return the best available torch device string.

        Args:
            preference: ``"auto"`` (pick GPU when available), ``"cuda"``, or ``"cpu"``.

        Returns:
            ``"cuda"`` or ``"cpu"``.
        """
        if preference == "cpu":
            return "cpu"

        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                log.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
                return device
        except ImportError:
            log.debug("PyTorch not installed; falling back to CPU.")

        if preference == "cuda":
            log.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"

    # ------------------------------------------------------------------
    # CPU info
    # ------------------------------------------------------------------
    @staticmethod
    def get_cpu_count() -> int:
        """Return the number of logical CPU cores."""
        try:
            import psutil

            return psutil.cpu_count(logical=True) or os.cpu_count() or 1
        except ImportError:
            return os.cpu_count() or 1

    # ------------------------------------------------------------------
    # System RAM
    # ------------------------------------------------------------------
    @staticmethod
    def get_memory_gb() -> float:
        """Return total system RAM in gigabytes."""
        try:
            import psutil

            return round(psutil.virtual_memory().total / (1024**3), 2)
        except ImportError:
            log.debug("psutil not available; cannot determine system RAM.")
            return 0.0

    # ------------------------------------------------------------------
    # GPU VRAM
    # ------------------------------------------------------------------
    @staticmethod
    def get_vram_info() -> Dict[str, object]:
        """Return GPU VRAM information when CUDA is available.

        Returns:
            A dict with keys ``available`` (bool), and when True also
            ``device_name``, ``total_mb``, ``used_mb``, ``free_mb``.
        """
        try:
            import torch

            if not torch.cuda.is_available():
                return {"available": False}

            idx = torch.cuda.current_device()
            total = torch.cuda.get_device_properties(idx).total_mem
            reserved = torch.cuda.memory_reserved(idx)
            allocated = torch.cuda.memory_allocated(idx)
            free = total - reserved

            return {
                "available": True,
                "device_name": torch.cuda.get_device_name(idx),
                "total_mb": round(total / (1024**2), 1),
                "used_mb": round(allocated / (1024**2), 1),
                "free_mb": round(free / (1024**2), 1),
            }
        except ImportError:
            return {"available": False}
