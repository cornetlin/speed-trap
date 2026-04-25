from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class StationConfig:
    station_id: str
    camera_source: str
    frame_width: int
    frame_height: int
    frame_fps: int
    hef_path: Path
    hailofilter_so_path: Path
    vehicle_classes: tuple[str, ...]
    trigger_line_y: float
    mqtt_broker: str | None
    mqtt_topic: str
    log_level: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.trigger_line_y <= 1.0:
            raise ValueError(
                f"trigger_line_y must be in [0, 1], got {self.trigger_line_y}"
            )
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame_width and frame_height must be positive")
        if self.frame_fps <= 0:
            raise ValueError("frame_fps must be positive")


def load_config(path: Path) -> StationConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")

    return StationConfig(
        station_id=str(data["station_id"]),
        camera_source=str(data["camera_source"]),
        frame_width=int(data["frame_width"]),
        frame_height=int(data["frame_height"]),
        frame_fps=int(data["frame_fps"]),
        hef_path=Path(data["hef_path"]),
        hailofilter_so_path=Path(data["hailofilter_so_path"]),
        vehicle_classes=tuple(data["vehicle_classes"]),
        trigger_line_y=float(data["trigger_line_y"]),
        mqtt_broker=data.get("mqtt_broker"),
        mqtt_topic=str(data["mqtt_topic"]),
        log_level=str(data.get("log_level", "INFO")),
    )
