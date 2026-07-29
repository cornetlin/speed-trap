"""PC-side smoke tests for the Hailo adapter.

The actual GStreamer pipeline can only run on a Pi with the Hailo HAT, but the
module must (a) import cleanly on a PC and (b) raise a clear error if someone
tries to instantiate it without the runtime available.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from speed_trap import hailo_source
from speed_trap.config import StationConfig


def _config() -> StationConfig:
    return StationConfig(
        station_id="station_a",
        camera_source="libcamera",
        frame_width=1280,
        frame_height=720,
        frame_fps=30,
        hef_path=Path("models/yolov8n.hef"),
        hailofilter_so_path=Path(
            "/usr/lib/hailo-post-processes/libyolo_hailortpp_postprocess.so"
        ),
        vehicle_classes=("car", "truck"),
        trigger_line_y=0.7,
        mqtt_broker=None,
        mqtt_topic="speedtrap/a",
        log_level="INFO",
    )


def test_module_imports_on_pc() -> None:
    assert hasattr(hailo_source, "HailoDetectionSource")


def test_instantiation_raises_when_runtime_missing() -> None:
    if hailo_source._IMPORT_ERROR is None:
        pytest.skip("Hailo runtime is available; PC-side error path not testable here")

    with pytest.raises(RuntimeError, match="Hailo runtime not available"):
        hailo_source.HailoDetectionSource(_config())


def test_error_message_mentions_pi_and_hailo() -> None:
    if hailo_source._IMPORT_ERROR is None:
        pytest.skip("Hailo runtime is available; PC-side error path not testable here")

    with pytest.raises(RuntimeError) as exc_info:
        hailo_source.HailoDetectionSource(_config())

    text = str(exc_info.value)
    assert "Raspberry Pi" in text
    assert "hailo" in text.lower()
