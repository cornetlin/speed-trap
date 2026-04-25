from __future__ import annotations

from dataclasses import dataclass

from speed_trap.tracker import Detection, VehicleTrack


@dataclass
class _CrossingState:
    last_bottom_y: float
    triggered: bool = False


class TriggerLineDetector:
    def __init__(self, line_y: float) -> None:
        if not 0.0 <= line_y <= 1.0:
            raise ValueError(f"line_y must be in [0, 1], got {line_y}")
        self._line_y = line_y
        self._states: dict[int, _CrossingState] = {}

    @property
    def line_y(self) -> float:
        return self._line_y

    def check_crossing(self, track: VehicleTrack, latest_det: Detection) -> bool:
        bottom_y = latest_det.bbox[3]
        state = self._states.get(track.track_id)
        if state is None:
            self._states[track.track_id] = _CrossingState(last_bottom_y=bottom_y)
            return False

        crossed = state.last_bottom_y < self._line_y <= bottom_y
        state.last_bottom_y = bottom_y
        return crossed

    def mark_triggered(self, track_id: int) -> None:
        state = self._states.get(track_id)
        if state is None:
            self._states[track_id] = _CrossingState(last_bottom_y=0.0, triggered=True)
            return
        state.triggered = True

    def was_triggered(self, track_id: int) -> bool:
        state = self._states.get(track_id)
        return state is not None and state.triggered
