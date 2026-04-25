from __future__ import annotations

import math
from dataclasses import dataclass

STALE_TIMEOUT_NS: int = 5 * 1_000_000_000

_AREA_WEIGHT: float = 0.6
_CENTER_WEIGHT: float = 0.3
_SHARPNESS_WEIGHT: float = 0.1
_MAX_CENTER_DIST: float = math.hypot(0.5, 0.5)


@dataclass(frozen=True)
class Detection:
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    frame_ns: int
    frame_jpeg: bytes | None


@dataclass
class VehicleTrack:
    track_id: int
    label: str
    first_seen_ns: int
    last_seen_ns: int
    best_detection: Detection
    best_score: float
    emitted: bool = False


def score_detection(det: Detection) -> float:
    x1, y1, x2, y2 = det.bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dist = math.hypot(cx - 0.5, cy - 0.5)
    center_closeness = max(0.0, 1.0 - dist / _MAX_CENTER_DIST)

    sharpness_placeholder = 0.0

    return (
        _AREA_WEIGHT * area
        + _CENTER_WEIGHT * center_closeness
        + _SHARPNESS_WEIGHT * sharpness_placeholder
    )


class VehicleTracker:
    def __init__(self, stale_timeout_ns: int = STALE_TIMEOUT_NS) -> None:
        self._stale_timeout_ns = stale_timeout_ns
        self._tracks: dict[int, VehicleTrack] = {}

    def update(self, detections: list[Detection]) -> None:
        for det in detections:
            score = score_detection(det)
            existing = self._tracks.get(det.track_id)
            if existing is None:
                self._tracks[det.track_id] = VehicleTrack(
                    track_id=det.track_id,
                    label=det.label,
                    first_seen_ns=det.frame_ns,
                    last_seen_ns=det.frame_ns,
                    best_detection=det,
                    best_score=score,
                )
                continue

            existing.last_seen_ns = det.frame_ns
            if score > existing.best_score:
                existing.best_detection = det
                existing.best_score = score

    def get_track(self, track_id: int) -> VehicleTrack | None:
        return self._tracks.get(track_id)

    def prune_stale(self, now_ns: int) -> list[int]:
        stale_ids = [
            tid
            for tid, track in self._tracks.items()
            if now_ns - track.last_seen_ns > self._stale_timeout_ns
        ]
        for tid in stale_ids:
            del self._tracks[tid]
        return stale_ids

    def all_active(self) -> list[VehicleTrack]:
        return list(self._tracks.values())
