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

    # === 最佳幀挑選 —— 現場調參數用,不要為了調這些去改程式 ===
    #
    # edge_margin: 正規化 bbox 只要有任一邊落在畫面邊緣這個距離之內,該幀就
    #   直接排除、不列入最佳幀候選(車頭已經出框,車牌會被裁掉一半)。
    #   0.02 = 畫面寬高的 2%。調大 = 更嚴格,可用幀變少。
    # score_*_weight: 面積 / 置中程度 / 清晰度三項的權重。面積權重過高時,
    #   系統會偏好「車子最大」也就是快出框的那一幀。
    # sharpness_reference: 清晰度正規化的參考值。清晰度用 Laplacian 變異數,
    #   數值無上界,除以這個參考值後截在 1.0,才能跟另外兩項同量級相加。
    #   拍出來普遍偏糊就調小,普遍很銳利就調大。
    # track_idle_timeout_s: 一個 track 多久沒再出現就視為車子已離開畫面。
    #   結算時機與每台車幀數統計都用這個值。
    # min_track_frames: 少於這麼多幀的 track 視為雜訊,不產生事件。
    # jpeg_quality: 車輛裁切圖的 JPEG 品質。
    edge_margin: float = 0.02
    score_area_weight: float = 0.6
    score_center_weight: float = 0.3
    score_sharpness_weight: float = 0.1
    sharpness_reference: float = 500.0
    track_idle_timeout_s: float = 2.0
    min_track_frames: int = 2
    jpeg_quality: int = 92

    # === 重複計數防治 ===
    # tracker_keep_frames: hailotracker 的 keep-tracked-frames / keep-new-frames。
    #   太小的話車子被短暫遮住就會斷開、重新編號,同一台車算成兩台。
    # duplicate_window_s: 這段時間內讀到相同車牌字串視為同一台車的重複事件,
    #   只記錄不送出。設 0 關閉。
    tracker_keep_frames: int = 10
    duplicate_window_s: float = 10.0

    # === 診斷輸出 ===
    # passage_csv_path: 每台車一行的詳細 CSV。設空字串關閉。
    # passage_crop_dir: 每台車最佳幀的裁切圖(成功與失敗都存),檔名對應
    #   CSV 的 crop_file 欄位。設空字串關閉。
    passage_csv_path: str = "~/speedtrap_events.csv"
    passage_crop_dir: str = "~/speedtrap_crops"

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
        if not 0.0 <= self.edge_margin < 0.5:
            raise ValueError(
                f"edge_margin must be in [0, 0.5), got {self.edge_margin}"
            )
        for name in (
            "score_area_weight",
            "score_center_weight",
            "score_sharpness_weight",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be >= 0, got {getattr(self, name)}")
        if self.sharpness_reference <= 0.0:
            raise ValueError("sharpness_reference must be positive")
        if self.track_idle_timeout_s <= 0.0:
            raise ValueError("track_idle_timeout_s must be positive")
        if self.min_track_frames < 1:
            raise ValueError("min_track_frames must be >= 1")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError(
                f"jpeg_quality must be in [1, 100], got {self.jpeg_quality}"
            )
        if self.tracker_keep_frames < 1:
            raise ValueError("tracker_keep_frames must be >= 1")
        if self.duplicate_window_s < 0.0:
            raise ValueError("duplicate_window_s must be >= 0")


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
        edge_margin=float(data.get("edge_margin", 0.02)),
        score_area_weight=float(data.get("score_area_weight", 0.6)),
        score_center_weight=float(data.get("score_center_weight", 0.3)),
        score_sharpness_weight=float(data.get("score_sharpness_weight", 0.1)),
        sharpness_reference=float(data.get("sharpness_reference", 500.0)),
        track_idle_timeout_s=float(data.get("track_idle_timeout_s", 2.0)),
        min_track_frames=int(data.get("min_track_frames", 2)),
        jpeg_quality=int(data.get("jpeg_quality", 92)),
        tracker_keep_frames=int(data.get("tracker_keep_frames", 10)),
        duplicate_window_s=float(data.get("duplicate_window_s", 10.0)),
        passage_csv_path=str(
            data.get("passage_csv_path", "~/speedtrap_events.csv")
        ),
        passage_crop_dir=str(data.get("passage_crop_dir", "~/speedtrap_crops")),
    )
