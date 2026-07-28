from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

_VALID_OCR_BACKENDS = frozenset({"fast-plate-ocr", "paddleocr", "noop"})


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
    # OCR configuration (Phase A — swap backends without code changes).
    # ocr_model_name: fast-plate-ocr hub name OR absolute path to a .onnx file.
    # ocr_model_config: only set when ocr_model_name is a path — points at
    #                   the plate_config.yaml that the training step produced.
    # ocr_plate_detector_path: W3 v2 — W2-trained plate_detector .pt/.onnx,
    #                          run on Pi CPU via ultralytics. When set, the
    #                          recognizer crops the plate region out of the
    #                          vehicle bbox BEFORE feeding fast-plate-ocr.
    #                          Without it, OCR gets the whole vehicle and
    #                          tries (badly) to find the plate itself.
    # save_debug_crops: dump JPGs of (vehicle, plate) to ~/speedtrap_debug on the
    #                   SD card so we can inspect what each stage sees. Off in prod.
    #
    # hailo_input_size: 編譯進 HEF 的輸入邊長(正方形),由模型決定,不是相機解析度。
    #                   偵測線把畫面縮到這個尺寸餵給 hailonet;裁切車輛區域則是回到
    #                   frame_width / frame_height 的原始畫面上做,車牌才留得住足夠
    #                   像素。換 HEF 時才需要改這個值。
    hailo_input_size: int = 640
    ocr_backend: str = "fast-plate-ocr"   # "fast-plate-ocr" | "paddleocr" | "noop"
    ocr_model_name: str = "global-plates-mobile-vit-v2-model"
    ocr_model_config: str | None = None
    ocr_plate_detector_path: str | None = None
    ocr_preprocess: bool = False          # apply CLAHE + sharpen before OCR
    save_debug_crops: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.trigger_line_y <= 1.0:
            raise ValueError(
                f"trigger_line_y must be in [0, 1], got {self.trigger_line_y}"
            )
        if self.frame_width <= 0 or self.frame_height <= 0:
            raise ValueError("frame_width and frame_height must be positive")
        if self.frame_fps <= 0:
            raise ValueError("frame_fps must be positive")
        if self.hailo_input_size <= 0:
            raise ValueError("hailo_input_size must be positive")
        if self.ocr_backend not in _VALID_OCR_BACKENDS:
            raise ValueError(
                f"ocr_backend must be one of {sorted(_VALID_OCR_BACKENDS)}, "
                f"got {self.ocr_backend!r}"
            )


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
        hailo_input_size=int(data.get("hailo_input_size", 640)),
        ocr_backend=str(data.get("ocr_backend", "fast-plate-ocr")),
        ocr_model_name=str(
            data.get("ocr_model_name", "global-plates-mobile-vit-v2-model")
        ),
        ocr_model_config=(
            str(data["ocr_model_config"])
            if data.get("ocr_model_config")
            else None
        ),
        ocr_plate_detector_path=(
            str(data["ocr_plate_detector_path"])
            if data.get("ocr_plate_detector_path")
            else None
        ),
        ocr_preprocess=bool(data.get("ocr_preprocess", False)),
        save_debug_crops=bool(data.get("save_debug_crops", False)),
    )
