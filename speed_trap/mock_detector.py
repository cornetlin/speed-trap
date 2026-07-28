from __future__ import annotations

import numpy as np

from speed_trap.clock import wall_clock_ns
from speed_trap.tracker import Detection


class MockDetector:
    """Deterministic detector for offline replay on PCs without Hailo.

    Simulates a single vehicle whose bbox center moves top -> bottom by
    a fixed step per frame. The same track_id is reported every frame
    until the bbox fully exits the frame.
    """

    def __init__(
        self,
        track_id: int = 1,
        label: str = "car",
        start_y_center: float = 0.1,
        step_per_frame: float = 0.02,
        bbox_w: float = 0.2,
        bbox_h: float = 0.15,
        confidence: float = 0.9,
    ) -> None:
        if step_per_frame <= 0:
            raise ValueError("step_per_frame must be positive")
        if not 0.0 < bbox_w <= 1.0 or not 0.0 < bbox_h <= 1.0:
            raise ValueError("bbox_w and bbox_h must be in (0, 1]")

        self._track_id = track_id
        self._label = label
        self._start_y = start_y_center
        self._step = step_per_frame
        self._bw = bbox_w
        self._bh = bbox_h
        self._confidence = confidence

    def detect(
        self,
        frame: np.ndarray,
        frame_index: int,
        frame_ns: int,
    ) -> list[Detection]:
        cy = self._start_y + frame_index * self._step

        if cy - self._bh / 2.0 > 1.0:
            return []

        cx = 0.5
        x1 = max(0.0, cx - self._bw / 2.0)
        x2 = min(1.0, cx + self._bw / 2.0)
        y1 = max(0.0, cy - self._bh / 2.0)
        y2 = min(1.0, cy + self._bh / 2.0)

        return [
            Detection(
                track_id=self._track_id,
                label=self._label,
                bbox=(x1, y1, x2, y2),
                confidence=self._confidence,
                frame_ns=frame_ns,
                capture_wall_ns=wall_clock_ns(),
                frame_jpeg=None,
            )
        ]
