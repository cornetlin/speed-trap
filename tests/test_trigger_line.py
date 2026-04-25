from __future__ import annotations

import pytest

from speed_trap.tracker import Detection, VehicleTrack
from speed_trap.trigger_line import TriggerLineDetector


def _det(track_id: int, y2: float, *, frame_ns: int = 0) -> Detection:
    return Detection(
        track_id=track_id,
        label="car",
        bbox=(0.4, max(0.0, y2 - 0.1), 0.6, y2),
        confidence=0.9,
        frame_ns=frame_ns,
        frame_jpeg=None,
    )


def _track(track_id: int, det: Detection) -> VehicleTrack:
    return VehicleTrack(
        track_id=track_id,
        label="car",
        first_seen_ns=det.frame_ns,
        last_seen_ns=det.frame_ns,
        best_detection=det,
        best_score=1.0,
    )


def test_first_detection_returns_false_no_history() -> None:
    detector = TriggerLineDetector(line_y=0.5)
    det = _det(1, y2=0.4)
    assert detector.check_crossing(_track(1, det), det) is False


def test_top_to_bottom_crossing_detected() -> None:
    detector = TriggerLineDetector(line_y=0.5)
    above = _det(1, y2=0.3)
    below = _det(1, y2=0.7)

    detector.check_crossing(_track(1, above), above)
    crossed = detector.check_crossing(_track(1, below), below)

    assert crossed is True


def test_bottom_to_top_does_not_trigger() -> None:
    detector = TriggerLineDetector(line_y=0.5)
    below = _det(1, y2=0.7)
    above = _det(1, y2=0.3)

    detector.check_crossing(_track(1, below), below)
    crossed = detector.check_crossing(_track(1, above), above)

    assert crossed is False


def test_already_past_line_does_not_re_trigger() -> None:
    detector = TriggerLineDetector(line_y=0.5)
    seq = [0.55, 0.6, 0.7, 0.8]
    crossings = []
    for y in seq:
        det = _det(1, y2=y)
        crossings.append(detector.check_crossing(_track(1, det), det))

    assert crossings == [False, False, False, False]


def test_crossing_exactly_on_line_counts() -> None:
    detector = TriggerLineDetector(line_y=0.5)
    above = _det(1, y2=0.3)
    on_line = _det(1, y2=0.5)

    detector.check_crossing(_track(1, above), above)
    crossed = detector.check_crossing(_track(1, on_line), on_line)

    assert crossed is True


def test_mark_and_was_triggered() -> None:
    detector = TriggerLineDetector(line_y=0.5)
    det = _det(1, y2=0.3)
    detector.check_crossing(_track(1, det), det)

    assert detector.was_triggered(1) is False
    detector.mark_triggered(1)
    assert detector.was_triggered(1) is True


def test_was_triggered_false_for_unknown_track() -> None:
    detector = TriggerLineDetector(line_y=0.5)
    assert detector.was_triggered(999) is False


def test_mark_triggered_for_unknown_track_creates_state() -> None:
    detector = TriggerLineDetector(line_y=0.5)
    detector.mark_triggered(42)
    assert detector.was_triggered(42) is True


def test_invalid_line_y_above_one_raises() -> None:
    with pytest.raises(ValueError, match="line_y"):
        TriggerLineDetector(line_y=1.5)


def test_invalid_line_y_below_zero_raises() -> None:
    with pytest.raises(ValueError, match="line_y"):
        TriggerLineDetector(line_y=-0.1)


def test_independent_state_per_track() -> None:
    detector = TriggerLineDetector(line_y=0.5)
    a_above = _det(1, y2=0.3)
    b_above = _det(2, y2=0.3)
    detector.check_crossing(_track(1, a_above), a_above)
    detector.check_crossing(_track(2, b_above), b_above)

    a_below = _det(1, y2=0.6)
    assert detector.check_crossing(_track(1, a_below), a_below) is True
    assert detector.was_triggered(2) is False


def test_line_y_property_exposes_value() -> None:
    detector = TriggerLineDetector(line_y=0.42)
    assert detector.line_y == pytest.approx(0.42)
