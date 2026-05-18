from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from speed_trap.config import StationConfig
from speed_trap.event import (
    ConsoleEventSink,
    EventSink,
    MqttEventSink,
    PassageEvent,
    SQLiteEventSink,
    make_sink,
)


def _event(**overrides: object) -> PassageEvent:
    defaults: dict[str, object] = dict(
        station_id="station_a",
        track_id=11,
        label="car",
        timestamp_ns=1_700_000_000_000_000_000,
        confidence=0.92,
        image_sha256="a" * 64,
    )
    defaults.update(overrides)
    return PassageEvent(**defaults)  # type: ignore[arg-type]


def _config(broker: str | None) -> StationConfig:
    return StationConfig(
        station_id="station_a",
        camera_source="cam",
        frame_width=640,
        frame_height=480,
        frame_fps=30,
        hef_path=Path("m.hef"),
        hailofilter_so_path=Path("/tmp/dummy.so"),
        vehicle_classes=("car",),
        trigger_line_y=0.5,
        mqtt_broker=broker,
        mqtt_topic="speedtrap/a",
        log_level="INFO",
    )


def test_passage_event_to_json_roundtrips() -> None:
    event = _event()
    decoded = json.loads(event.to_json())

    assert decoded["station_id"] == "station_a"
    assert decoded["track_id"] == 11
    assert decoded["label"] == "car"
    assert decoded["timestamp_ns"] == 1_700_000_000_000_000_000
    assert decoded["confidence"] == pytest.approx(0.92)
    assert decoded["image_sha256"] == "a" * 64


def test_passage_event_to_json_keys_sorted() -> None:
    event = _event()
    text = event.to_json()
    keys = list(json.loads(text))
    assert keys == sorted(keys)


def test_event_sink_is_abstract() -> None:
    with pytest.raises(TypeError):
        EventSink()  # type: ignore[abstract]


def test_console_sink_logs_event(caplog: pytest.LogCaptureFixture) -> None:
    sink = ConsoleEventSink(logger=logging.getLogger("test.console"))
    with caplog.at_level(logging.INFO, logger="test.console"):
        sink.emit(_event())

    assert any("passage event" in r.message for r in caplog.records)
    sink.close()


def test_console_sink_default_logger_emits(caplog: pytest.LogCaptureFixture) -> None:
    sink = ConsoleEventSink()
    with caplog.at_level(logging.INFO, logger="speed_trap.event"):
        sink.emit(_event())
    assert any("passage event" in r.message for r in caplog.records)


def test_sqlite_sink_persists_event(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    sink = SQLiteEventSink(db_path)
    sink.emit(_event(track_id=1))
    sink.emit(_event(track_id=2, label="truck"))
    sink.close()

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT track_id, label FROM events ORDER BY track_id"
        ).fetchall()

    assert rows == [(1, "car"), (2, "truck")]


def test_sqlite_sink_creates_schema_idempotently(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    SQLiteEventSink(db_path).close()
    sink = SQLiteEventSink(db_path)
    sink.emit(_event())
    sink.close()

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 1


def test_mqtt_sink_logs_stub(caplog: pytest.LogCaptureFixture) -> None:
    sink = MqttEventSink(
        broker="mqtt.local:1883",
        topic="speedtrap/a",
        logger=logging.getLogger("test.mqtt"),
    )
    with caplog.at_level(logging.INFO, logger="test.mqtt"):
        sink.emit(_event())

    messages = [r.message for r in caplog.records]
    assert any("[mqtt-stub]" in m and "mqtt.local:1883" in m for m in messages)


def test_make_sink_returns_mqtt_when_broker_set() -> None:
    sink = make_sink(_config(broker="mqtt.local:1883"))
    assert isinstance(sink, MqttEventSink)


def test_make_sink_returns_console_when_broker_none() -> None:
    sink = make_sink(_config(broker=None))
    assert isinstance(sink, ConsoleEventSink)


def test_make_sink_returns_console_when_broker_empty_string() -> None:
    sink = make_sink(_config(broker=""))
    assert isinstance(sink, ConsoleEventSink)


def test_passage_event_includes_plate_fields_when_set() -> None:
    event = _event(plate_text="ABC1234", plate_confidence=0.93)
    decoded = json.loads(event.to_json())
    assert decoded["plate_text"] == "ABC1234"
    assert decoded["plate_confidence"] == pytest.approx(0.93)


def test_passage_event_plate_fields_default_to_none() -> None:
    event = _event()
    decoded = json.loads(event.to_json())
    assert decoded["plate_text"] is None
    assert decoded["plate_confidence"] is None


def test_sqlite_sink_persists_plate_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "events.db"
    sink = SQLiteEventSink(db_path)
    sink.emit(_event(track_id=1, plate_text="ABC1234", plate_confidence=0.88))
    sink.emit(_event(track_id=2))  # no plate
    sink.close()

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT track_id, plate_text, plate_confidence FROM events ORDER BY track_id"
        ).fetchall()
    assert rows == [(1, "ABC1234", pytest.approx(0.88)), (2, None, None)]
