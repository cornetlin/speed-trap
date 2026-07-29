from __future__ import annotations

from speed_trap.tracker import (
    STALE_TIMEOUT_NS,
    Detection,
    VehicleTracker,
    score_detection,
)

_NS_PER_SEC = 1_000_000_000


def _det(
    track_id: int,
    bbox: tuple[float, float, float, float],
    *,
    label: str = "car",
    confidence: float = 0.9,
    frame_ns: int = 0,
    frame_jpeg: bytes | None = None,
) -> Detection:
    return Detection(
        track_id=track_id,
        label=label,
        bbox=bbox,
        confidence=confidence,
        frame_ns=frame_ns,
        frame_jpeg=frame_jpeg,
    )


def test_empty_tracker_has_no_tracks() -> None:
    tracker = VehicleTracker()
    assert tracker.get_track(1) is None
    assert tracker.all_active() == []


def test_update_creates_track_with_initial_detection() -> None:
    tracker = VehicleTracker()
    det = _det(7, (0.4, 0.4, 0.6, 0.6), frame_ns=1_000)

    tracker.update([det])
    track = tracker.get_track(7)

    assert track is not None
    assert track.track_id == 7
    assert track.label == "car"
    assert track.first_seen_ns == 1_000
    assert track.last_seen_ns == 1_000
    assert track.best_detection is det
    assert track.emitted is False


def test_higher_score_detection_replaces_best_detection() -> None:
    tracker = VehicleTracker()
    corner = _det(1, (0.0, 0.0, 0.05, 0.05), frame_ns=1)
    centered = _det(1, (0.3, 0.3, 0.7, 0.7), frame_ns=2, frame_jpeg=b"jpg")

    tracker.update([corner])
    tracker.update([centered])

    track = tracker.get_track(1)
    assert track is not None
    assert track.best_detection is centered
    assert track.last_seen_ns == 2


def test_lower_score_detection_keeps_best_but_updates_last_seen() -> None:
    tracker = VehicleTracker()
    centered = _det(1, (0.3, 0.3, 0.7, 0.7), frame_ns=10)
    corner = _det(1, (0.0, 0.0, 0.05, 0.05), frame_ns=20)

    tracker.update([centered])
    tracker.update([corner])

    track = tracker.get_track(1)
    assert track is not None
    assert track.best_detection is centered
    assert track.last_seen_ns == 20
    assert track.first_seen_ns == 10


def test_prune_stale_removes_old_tracks() -> None:
    tracker = VehicleTracker()
    tracker.update([_det(1, (0.4, 0.4, 0.6, 0.6), frame_ns=0)])
    tracker.update([_det(2, (0.4, 0.4, 0.6, 0.6), frame_ns=4 * _NS_PER_SEC)])

    pruned = tracker.prune_stale(now_ns=6 * _NS_PER_SEC)

    assert pruned == [1]
    assert tracker.get_track(1) is None
    assert tracker.get_track(2) is not None


def test_prune_stale_keeps_fresh_tracks() -> None:
    tracker = VehicleTracker()
    tracker.update([_det(1, (0.4, 0.4, 0.6, 0.6), frame_ns=10 * _NS_PER_SEC)])

    pruned = tracker.prune_stale(now_ns=12 * _NS_PER_SEC)

    assert pruned == []
    assert len(tracker.all_active()) == 1


def test_prune_stale_boundary_exactly_at_timeout_keeps_track() -> None:
    tracker = VehicleTracker()
    tracker.update([_det(1, (0.4, 0.4, 0.6, 0.6), frame_ns=0)])

    pruned = tracker.prune_stale(now_ns=STALE_TIMEOUT_NS)

    assert pruned == []


def test_score_detection_orders_centered_above_corner() -> None:
    centered = _det(1, (0.4, 0.4, 0.6, 0.6))
    corner = _det(2, (0.0, 0.0, 0.2, 0.2))
    full = _det(3, (0.0, 0.0, 1.0, 1.0))

    assert score_detection(full) > score_detection(centered) > score_detection(corner)


def test_score_detection_handles_inverted_bbox_as_zero_area() -> None:
    inverted = _det(1, (0.6, 0.6, 0.4, 0.4))
    score = score_detection(inverted)
    assert score >= 0.0


def test_multiple_tracks_are_independent() -> None:
    tracker = VehicleTracker()
    tracker.update(
        [
            _det(1, (0.1, 0.1, 0.2, 0.2), frame_ns=1),
            _det(2, (0.4, 0.4, 0.6, 0.6), frame_ns=1, label="truck"),
        ]
    )

    assert len(tracker.all_active()) == 2
    track2 = tracker.get_track(2)
    assert track2 is not None and track2.label == "truck"


def test_custom_stale_timeout() -> None:
    tracker = VehicleTracker(stale_timeout_ns=1_000)
    tracker.update([_det(1, (0.4, 0.4, 0.6, 0.6), frame_ns=0)])

    pruned = tracker.prune_stale(now_ns=2_000)
    assert pruned == [1]
