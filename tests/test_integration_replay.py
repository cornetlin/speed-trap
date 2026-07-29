from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from apps.replay_video import main, run_replay
from speed_trap.config import StationConfig, load_config
from speed_trap.event import EventSink, PassageEvent


class _RecordingSink(EventSink):
    def __init__(self) -> None:
        self.events: list[PassageEvent] = []
        self.closed = False

    def emit(self, event: PassageEvent) -> None:
        self.events.append(event)

    def close(self) -> None:
        self.closed = True


def _synthesize_video(
    path: Path,
    *,
    frames: int = 100,
    width: int = 320,
    height: int = 240,
    fps: int = 30,
) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        pytest.skip("cv2.VideoWriter (mp4v) unavailable in this environment")
    try:
        for i in range(frames):
            shade = (i * 2) % 256
            frame = np.full((height, width, 3), shade, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()

    if not path.exists() or path.stat().st_size == 0:
        pytest.skip("video writer produced empty output (codec missing)")


def _config(video_path: Path, *, trigger_line_y: float = 0.5) -> StationConfig:
    return StationConfig(
        station_id="test_station",
        camera_source=str(video_path),
        frame_width=320,
        frame_height=240,
        frame_fps=30,
        hef_path=Path("dummy.hef"),
        hailofilter_so_path=Path("/tmp/dummy.so"),
        vehicle_classes=("car",),
        trigger_line_y=trigger_line_y,
        mqtt_broker=None,
        mqtt_topic="test/topic",
        log_level="INFO",
    )


def test_replay_emits_passage_event_for_synthesized_video(tmp_path: Path) -> None:
    video_path = tmp_path / "synth.mp4"
    _synthesize_video(video_path, frames=100)

    sink = _RecordingSink()
    events = run_replay(video_path, _config(video_path), sink, show_display=False)

    assert len(events) >= 1
    event = events[0]
    assert event.station_id == "test_station"
    assert event.label == "car"
    assert event.track_id == 1
    assert events == sink.events


def test_replay_emits_only_once_per_track(tmp_path: Path) -> None:
    video_path = tmp_path / "synth.mp4"
    _synthesize_video(video_path, frames=100)

    sink = _RecordingSink()
    events = run_replay(video_path, _config(video_path), sink, show_display=False)

    track_ids = [e.track_id for e in events]
    assert len(track_ids) == len(set(track_ids))


def test_replay_raises_when_video_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        run_replay(missing, _config(missing), _RecordingSink(), show_display=False)


def test_main_returns_zero_with_synthesized_video(tmp_path: Path) -> None:
    video_path = tmp_path / "synth.mp4"
    _synthesize_video(video_path, frames=60)

    cfg_path = tmp_path / "station.yaml"
    cfg_data = {
        "station_id": "main_test",
        "camera_source": str(video_path),
        "frame_width": 320,
        "frame_height": 240,
        "frame_fps": 30,
        "hef_path": "dummy.hef",
        "hailofilter_so_path": "/tmp/dummy.so",
        "vehicle_classes": ["car"],
        "trigger_line_y": 0.5,
        "mqtt_broker": None,
        "mqtt_topic": "test/topic",
        "log_level": "WARNING",
    }
    cfg_path.write_text(yaml.safe_dump(cfg_data), encoding="utf-8")

    rc = main(["--video", str(video_path), "--config", str(cfg_path), "--no-display"])
    assert rc == 0
    # sanity: config we wrote is loadable
    assert load_config(cfg_path).station_id == "main_test"
