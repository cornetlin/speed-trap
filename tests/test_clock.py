"""兩種時鐘:單調時鐘量間隔,真實時鐘做對外時間戳。

單調時鐘的起點是各台機器自己的開機時刻 —— 重開機後前後兩批資料對不起來,
A、B 兩站相減算速度也沒有意義。但也不能全面改用真實時間:NTP 往回校正時,
用它量間隔會算短甚至變成負數。
"""

from __future__ import annotations

import subprocess
import time
from datetime import datetime

import pytest

from tests.helpers import (
    NS_PER_SEC,
    FakeSource,
    FixedOCR,
    ListSink,
    approaching_car,
    make_config,
)

from apps.run_station import _StopFlag, run_station
from speed_trap import clock

_YEAR_2020_NS = 1_577_836_800 * NS_PER_SEC


def test_wall_clock_is_a_real_epoch_timestamp() -> None:
    now = clock.wall_clock_ns()
    assert now > _YEAR_2020_NS
    # 單調時鐘不是 epoch —— 拿它當時間戳就是原本的 bug
    assert time.monotonic_ns() < _YEAR_2020_NS


def test_wall_clock_tracks_time_dot_time() -> None:
    before = time.time_ns()
    sample = clock.wall_clock_ns()
    after = time.time_ns()
    assert before <= sample <= after


def test_wall_ns_to_iso_round_trips() -> None:
    sample = clock.wall_clock_ns()
    text = clock.wall_ns_to_iso(sample)
    parsed = datetime.fromisoformat(text)
    # 到毫秒為止,所以誤差上限是 1 ms
    assert abs(parsed.timestamp() - sample / NS_PER_SEC) < 0.001


def test_wall_ns_to_iso_treats_zero_as_unknown() -> None:
    assert clock.wall_ns_to_iso(0) == ""


@pytest.mark.parametrize(
    ("stdout", "expected"),
    [
        ("NTPSynchronized=yes\n", True),
        ("NTPSynchronized=no\n", False),
        ("NTPSynchronized=true\n", True),
        ("NTPSynchronized=false\n", False),
    ],
)
def test_ntp_status_parses_timedatectl(
    monkeypatch: pytest.MonkeyPatch, stdout: str, expected: bool
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert clock.ntp_status().synchronized is expected


def test_ntp_status_unknown_when_command_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError("timedatectl")

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = clock.ntp_status()
    assert status.synchronized is None
    assert "FileNotFoundError" in status.detail


def test_ntp_status_unknown_on_nonzero_exit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="not systemd"
        )

    monkeypatch.setattr(subprocess, "run", fake_run)
    status = clock.ntp_status()
    assert status.synchronized is None
    assert "not systemd" in status.detail


def test_event_timestamp_uses_wall_clock() -> None:
    config = make_config()
    sink = ListSink()
    before = clock.wall_clock_ns()
    run_station(
        config,
        sink,
        FakeSource(approaching_car(1, time.monotonic_ns(), frames=30)),
        _StopFlag(),
        recognizer=FixedOCR("ABC1234"),
    )
    after = clock.wall_clock_ns()

    assert len(sink.events) == 1
    assert before <= sink.events[0].timestamp_ns <= after


def test_tracker_still_uses_monotonic_for_intervals() -> None:
    """真實時鐘會被 NTP 往回調,拿去量間隔會算錯 —— 這幾處必須維持單調。"""
    from speed_trap.tracker import VehicleTracker

    from tests.helpers import make_detection

    tracker = VehicleTracker(stale_timeout_ns=2 * NS_PER_SEC)
    # frame_ns 用的是小數值(模擬開機後幾秒),若實作誤用 capture_wall_ns
    # (epoch,約 1.8e18)這裡的 prune 判斷就會整個錯掉。
    tracker.update([make_detection(1, (0.4, 0.4, 0.6, 0.6), frame_ns=0)])

    assert tracker.prune_stale(NS_PER_SEC) == []
    assert [track.track_id for track in tracker.prune_stale(3 * NS_PER_SEC)] == [1]
