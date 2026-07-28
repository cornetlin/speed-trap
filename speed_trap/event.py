from __future__ import annotations

import json
import logging
import sqlite3
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path

from speed_trap.config import StationConfig

_logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PassageEvent:
    station_id: str
    track_id: int
    label: str
    timestamp_ns: int
    confidence: float
    image_sha256: str
    plate_text: str | None = None
    plate_confidence: float | None = None
    plate_is_taiwan_format: bool | None = None
    # 這筆紀錄的可信度有疑慮,下游應該讓人看過。原因見 review_reason
    # (例如整條 track 每一幀都碰到畫面邊界)。
    needs_review: bool = False
    review_reason: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


class EventSink(ABC):
    @abstractmethod
    def emit(self, event: PassageEvent) -> None: ...

    def close(self) -> None:
        return None


class ConsoleEventSink(EventSink):
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or _logger

    def emit(self, event: PassageEvent) -> None:
        self._logger.info("passage event: %s", event.to_json())


class SQLiteEventSink(EventSink):
    _SCHEMA = """
        CREATE TABLE IF NOT EXISTS events (
            station_id              TEXT    NOT NULL,
            track_id                INTEGER NOT NULL,
            label                   TEXT    NOT NULL,
            timestamp_ns            INTEGER NOT NULL,
            confidence              REAL    NOT NULL,
            image_sha256            TEXT    NOT NULL,
            plate_text              TEXT,
            plate_confidence        REAL,
            plate_is_taiwan_format  INTEGER,
            needs_review            INTEGER,
            review_reason           TEXT
        )
    """

    # 舊資料庫沒有這兩欄,開啟時補上,免得 INSERT 欄位數對不起來。
    _ADDED_COLUMNS = (
        ("needs_review", "INTEGER"),
        ("review_reason", "TEXT"),
    )

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute(self._SCHEMA)
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(events)")
        }
        for name, sql_type in self._ADDED_COLUMNS:
            if name not in existing:
                self._conn.execute(
                    f"ALTER TABLE events ADD COLUMN {name} {sql_type}"
                )
        self._conn.commit()

    def emit(self, event: PassageEvent) -> None:
        self._conn.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event.station_id,
                event.track_id,
                event.label,
                event.timestamp_ns,
                event.confidence,
                event.image_sha256,
                event.plate_text,
                event.plate_confidence,
                event.plate_is_taiwan_format,
                int(event.needs_review),
                event.review_reason,
            ),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()


class MqttEventSink(EventSink):
    """Stub: logs payload only. Real broker connection comes later."""

    def __init__(
        self,
        broker: str,
        topic: str,
        logger: logging.Logger | None = None,
    ) -> None:
        self._broker = broker
        self._topic = topic
        self._logger = logger or _logger

    def emit(self, event: PassageEvent) -> None:
        self._logger.info(
            "[mqtt-stub] %s -> %s: %s", self._broker, self._topic, event.to_json()
        )


def make_sink(config: StationConfig) -> EventSink:
    if config.mqtt_broker:
        return MqttEventSink(config.mqtt_broker, config.mqtt_topic)
    return ConsoleEventSink()
