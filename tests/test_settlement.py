"""結算流程:結算時機、重複抑制、雜訊過濾、狀態清理。

實測 54 台車卻記錄到 61 筆事件(多 13%),以及最佳幀選在車子出框那一刻,
都是這一層的問題。
"""

from __future__ import annotations

import time

from tests.helpers import (
    NS_PER_SEC,
    EchoOCR,
    FakeSource,
    FixedOCR,
    ListSink,
    approaching_car,
    make_config,
    make_detection,
)

from apps.run_station import (
    REVIEW_ALL_FRAMES_EDGE,
    REVIEW_DUPLICATE,
    _DuplicateGuard,
    _settle_finished,
    _StopFlag,
    run_station,
)
from speed_trap.tracker import ScoringParams, VehicleTracker
from speed_trap.trigger_line import TriggerLineDetector


def _build(config):
    """組出結算需要的 tracker + trigger 組合。"""
    tracker = VehicleTracker(
        stale_timeout_ns=int(config.track_idle_timeout_s * NS_PER_SEC),
        scoring=ScoringParams.from_config(config),
    )
    return tracker, TriggerLineDetector(line_y=config.trigger_line_y)


def _feed(tracker, trigger, detections) -> None:
    for detection in detections:
        tracker.update([detection])
        track = tracker.get_track(detection.track_id)
        if trigger.check_crossing(track, detection) and not trigger.was_triggered(
            detection.track_id
        ):
            trigger.mark_triggered(detection.track_id)


# --- 結算時機 -----------------------------------------------------------


def test_crossing_the_line_does_not_settle_immediately() -> None:
    """車子朝相機開來,過線之後還會變大變清楚 —— 不能在過線當下定案。"""
    config = make_config()
    tracker, trigger = _build(config)
    sink = ListSink()
    car = approaching_car(1, 0, frames=30)

    for detection in car:
        tracker.update([detection])
        track = tracker.get_track(1)
        if trigger.check_crossing(track, detection) and not trigger.was_triggered(1):
            trigger.mark_triggered(1)
        settled = _settle_finished(
            config,
            tracker.prune_stale(detection.frame_ns),
            trigger,
            sink,
            EchoOCR(),
        )
        assert settled == 0

    assert trigger.was_triggered(1), "這台車應該有過線"
    assert sink.events == []


def test_settles_only_after_idle_timeout() -> None:
    config = make_config(track_idle_timeout_s=2.0)
    tracker, trigger = _build(config)
    sink = ListSink()
    car = approaching_car(1, 0, frames=30)
    _feed(tracker, trigger, car)
    last_ns = car[-1].frame_ns

    # 還沒超過 idle timeout
    assert (
        _settle_finished(
            config, tracker.prune_stale(last_ns + NS_PER_SEC), trigger, sink, EchoOCR()
        )
        == 0
    )
    # 超過了
    assert (
        _settle_finished(
            config,
            tracker.prune_stale(last_ns + 3 * NS_PER_SEC),
            trigger,
            sink,
            EchoOCR(),
        )
        == 1
    )
    assert len(sink.events) == 1


def test_run_station_settles_leftover_tracks_on_shutdown() -> None:
    """收工時畫面上還在、但已過線的車不能漏掉。"""
    config = make_config()
    sink = ListSink()
    source = FakeSource(approaching_car(1, time.monotonic_ns(), frames=30))

    emitted = run_station(config, sink, source, _StopFlag(), recognizer=EchoOCR())

    assert emitted == 1
    assert len(sink.events) == 1
    assert source.started and source.stopped


def test_all_edge_track_still_emitted_but_flagged() -> None:
    """保護機制:不放棄,但標記需人工複核。"""
    config = make_config()
    tracker, trigger = _build(config)
    sink = ListSink()
    boxes = [
        (0.62, 0.40, 0.999, 0.55),
        (0.62, 0.45, 0.999, 0.60),
        (0.62, 0.60, 0.999, 0.85),   # 底緣越過 0.7 觸發線
    ]
    _feed(
        tracker,
        trigger,
        [
            make_detection(5, bbox, index * NS_PER_SEC // 30)
            for index, bbox in enumerate(boxes)
        ],
    )

    assert _settle_finished(config, tracker.drain(), trigger, sink, EchoOCR()) == 1
    event = sink.events[0]
    assert event.needs_review
    assert REVIEW_ALL_FRAMES_EDGE in event.review_reason


# --- 重複計數 -----------------------------------------------------------


def test_non_vehicle_labels_never_produce_events() -> None:
    """行人與盆栽拿得到 track id,而且也會過線,但不是車。"""
    config = make_config(vehicle_classes=("car",))
    sink = ListSink()
    base = time.monotonic_ns()
    detections = (
        approaching_car(1, base, frames=30)
        + approaching_car(2, base, frames=30, label="person")
        + approaching_car(3, base, frames=30, label="pottedplant")
    )

    emitted = run_station(
        config, sink, FakeSource(detections), _StopFlag(), recognizer=EchoOCR()
    )

    assert emitted == 1
    assert [event.label for event in sink.events] == ["car"]


def test_same_plate_from_two_track_ids_is_suppressed() -> None:
    """同一台車被 tracker 斷開重新編號 —— 兩段都過線,但只算一次。"""
    config = make_config(duplicate_window_s=10.0)
    tracker, trigger = _build(config)
    sink = ListSink()
    guard = _DuplicateGuard(config.duplicate_window_s)
    ocr = FixedOCR("ABC1234")
    base = 0

    for track_id, start in ((7, base), (8, base + 2 * NS_PER_SEC)):
        _feed(tracker, trigger, approaching_car(track_id, start, frames=15))
        _settle_finished(
            config,
            tracker.prune_stale(start + 10 * NS_PER_SEC),
            trigger,
            sink,
            ocr,
            guard,
        )

    assert len(sink.events) == 1


def test_same_plate_outside_window_is_kept() -> None:
    config = make_config(duplicate_window_s=10.0)
    tracker, trigger = _build(config)
    sink = ListSink()
    guard = _DuplicateGuard(config.duplicate_window_s)
    ocr = FixedOCR("ABC1234")

    for track_id, start in ((9, 0), (10, 60 * NS_PER_SEC)):
        _feed(tracker, trigger, approaching_car(track_id, start, frames=15))
        _settle_finished(
            config,
            tracker.prune_stale(start + 20 * NS_PER_SEC),
            trigger,
            sink,
            ocr,
            guard,
        )

    assert len(sink.events) == 2


def test_duplicate_guard_ignores_missing_plate_text() -> None:
    """讀不到車牌時無從比對 —— 寧可多一筆,也不要把兩台車併成一筆。"""
    guard = _DuplicateGuard(10.0)
    guard.remember(None, 0, 1)
    assert guard.duplicate_of(None, NS_PER_SEC, 2) is None


def test_duplicate_guard_ignores_same_track_id() -> None:
    guard = _DuplicateGuard(10.0)
    guard.remember("ABC1234", 0, 1)
    assert guard.duplicate_of("ABC1234", NS_PER_SEC, 1) is None
    assert guard.duplicate_of("ABC1234", NS_PER_SEC, 2) == 1


def test_duplicate_guard_disabled_by_zero_window() -> None:
    guard = _DuplicateGuard(0.0)
    guard.remember("ABC1234", 0, 1)
    assert not guard.enabled
    assert guard.duplicate_of("ABC1234", NS_PER_SEC, 2) is None


def test_duplicate_is_flagged_in_review_reason() -> None:
    config = make_config()
    tracker, trigger = _build(config)
    guard = _DuplicateGuard(10.0)
    guard.remember("ABC1234", 0, 99)
    sink = ListSink()

    _feed(tracker, trigger, approaching_car(7, 0, frames=15))
    settled = _settle_finished(
        config,
        tracker.drain(),
        trigger,
        sink,
        FixedOCR("ABC1234"),
        guard,
    )

    assert settled == 0
    assert sink.events == []  # 判為重複就不送出


def test_short_tracks_are_discarded_as_noise() -> None:
    config = make_config(min_track_frames=5)
    tracker, trigger = _build(config)
    sink = ListSink()

    _feed(tracker, trigger, approaching_car(11, 0, frames=3))
    trigger.mark_triggered(11)

    assert _settle_finished(config, tracker.drain(), trigger, sink, EchoOCR()) == 0


# --- 狀態清理 -----------------------------------------------------------


def test_trigger_states_are_cleared_when_tracks_finish() -> None:
    """_states 原本只增不減,而且 track id 會被 hailotracker 回收再利用。"""
    config = make_config()
    tracker, trigger = _build(config)
    sink = ListSink()

    for track_id in range(1, 21):
        _feed(tracker, trigger, approaching_car(track_id, track_id * 3 * NS_PER_SEC, frames=10))

    assert trigger.tracked_count == 20
    _settle_finished(
        config, tracker.prune_stale(200 * NS_PER_SEC), trigger, sink, EchoOCR()
    )
    assert trigger.tracked_count == 0


def test_reused_track_id_does_not_inherit_triggered_state() -> None:
    trigger = TriggerLineDetector(line_y=0.7)
    trigger.mark_triggered(5)
    assert trigger.was_triggered(5)

    trigger.forget(5)

    assert not trigger.was_triggered(5)
    assert trigger.tracked_count == 0
