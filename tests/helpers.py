"""站台層測試共用的假物件與資料建構器。

這裡的東西刻意不碰相機、Hailo 與 OCR 模型 —— 被測的都是純邏輯,在 PC 上
就要能跑完。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from speed_trap.clock import wall_clock_ns
from speed_trap.config import StationConfig
from speed_trap.event import EventSink, PassageEvent
from speed_trap.plate_recognizer import OcrAttempt, PlateReading
from speed_trap.tracker import Detection

NS_PER_SEC = 1_000_000_000


def make_config(**overrides: Any) -> StationConfig:
    """測試用 StationConfig。

    刻意不讀 config/station_a.yaml —— 那份是現場調參數用的,值會一直變,
    測試不該跟著它一起壞。
    """
    base: dict[str, Any] = {
        "station_id": "test_station",
        "camera_source": "/dev/video0",
        "frame_width": 1920,
        "frame_height": 1080,
        "frame_fps": 30,
        "hef_path": Path("/nonexistent.hef"),
        "hailofilter_so_path": Path("/nonexistent.so"),
        "vehicle_classes": ("car", "truck", "bus", "motorcycle"),
        "trigger_line_y": 0.7,
        "mqtt_broker": None,
        "mqtt_topic": "test/passage",
        "log_level": "INFO",
        "ocr_backend": "noop",
        # 測試預設不落檔;要測 CSV 的用例自己覆寫。
        "passage_csv_path": "",
        "passage_crop_dir": "",
    }
    base.update(overrides)
    return StationConfig(**base)


def make_detection(
    track_id: int,
    bbox: tuple[float, float, float, float],
    frame_ns: int,
    *,
    label: str = "car",
    confidence: float = 0.88,
    frame_jpeg: bytes | None = b"jpeg",
    sharpness: float = 400.0,
    sharpness_failed: bool = False,
    crop_size: tuple[int, int] | None = (600, 480),
    capture_wall_ns: int | None = None,
) -> Detection:
    return Detection(
        track_id=track_id,
        label=label,
        bbox=bbox,
        confidence=confidence,
        frame_ns=frame_ns,
        capture_wall_ns=(
            wall_clock_ns() if capture_wall_ns is None else capture_wall_ns
        ),
        frame_jpeg=frame_jpeg,
        sharpness=sharpness,
        sharpness_failed=sharpness_failed,
        crop_size=crop_size,
    )


def approaching_car(
    track_id: int,
    start_ns: int,
    *,
    frames: int = 30,
    label: str = "car",
) -> list[Detection]:
    """一台朝相機開來的車:bbox 逐漸變大並下移,最後幾幀衝出畫面下緣。

    這是本專案最典型的軌跡 —— 面積最大的幾幀正好是車頭已經出框的那幾幀,
    也就是邊界排除要處理的情況。
    """
    out: list[Detection] = []
    for index in range(frames):
        progress = index / (frames - 1)
        half = 0.06 + 0.34 * progress
        center_y = 0.35 + 0.55 * progress
        bbox = (0.5 - half, center_y - half, 0.5 + half, center_y + half)
        out.append(
            make_detection(
                track_id,
                bbox,
                start_ns + index * NS_PER_SEC // 30,
                label=label,
                frame_jpeg=f"t{track_id}f{index:02d}".encode(),
            )
        )
    return out


class ListSink(EventSink):
    """把送出的事件收在 list 裡供斷言。"""

    def __init__(self) -> None:
        self.events: list[PassageEvent] = []

    def emit(self, event: PassageEvent) -> None:
        self.events.append(event)


class FakeSource:
    """依序吐出預先排好的 Detection,取代 HailoDetectionSource。"""

    def __init__(self, detections: list[Detection]) -> None:
        self._detections = detections
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        self.stopped = True

    def iter_detections(self, *, poll_interval_s: float = 0.1) -> Any:
        yield from self._detections


class EchoOCR:
    """讀出的「車牌」就是那一幀的代號,用來斷言系統挑了第幾幀。"""

    model_name = "echo"

    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None:
        text = jpeg_bytes.decode()
        return PlateReading(
            text=text, raw_text=text, confidence=0.9, is_taiwan_format=True
        )


class FixedOCR:
    """所有輸入都讀成同一個車牌,用來測重複抑制。

    只實作 read_attempt,順便涵蓋 run_ocr 走詳細介面的那條路。
    """

    model_name = "fixed"

    def __init__(
        self,
        text: str | None,
        *,
        vehicle_wh: tuple[int, int] = (880, 690),
        plate_wh: tuple[int, int] = (196, 58),
        is_taiwan_format: bool = True,
    ) -> None:
        self._text = text
        self._vehicle_wh = vehicle_wh
        self._plate_wh = plate_wh
        self._is_taiwan_format = is_taiwan_format

    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None:
        return self.read_attempt(jpeg_bytes).reading

    def read_attempt(self, jpeg_bytes: bytes) -> OcrAttempt:
        if self._text is None:
            return OcrAttempt(
                reading=None,
                vehicle_wh=self._vehicle_wh,
                failure="no_plate_detected",
            )
        return OcrAttempt(
            reading=PlateReading(
                text=self._text,
                raw_text=self._text.lower(),
                confidence=0.87,
                is_taiwan_format=self._is_taiwan_format,
            ),
            vehicle_wh=self._vehicle_wh,
            plate_wh=self._plate_wh,
        )
