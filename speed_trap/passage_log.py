"""每台車一行的診斷紀錄:CSV + 最佳幀裁切圖。

現場實驗成本很高(要在白天等車流),所以一次實驗要能回答完所有問題。
這個模組把每台車的完整決策過程寫下來:被處理到幾幀、挑中第幾幀、為什麼
挑它、那一幀裁出來多大、車牌框多寬、OCR 讀到什麼、有沒有送出、沒送出的
原因是什麼。

CSV 的每一行都對應一個裁切圖檔,檔名記在 crop_file 欄位,不論成功或失敗
都會存 —— 讀不出車牌的那幾台才是要看的。

每行寫完就 flush,現場中途斷電也不會整份不見。
"""

from __future__ import annotations

import csv
import logging
import os
import re
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9_.-]")


@dataclass
class PassageRecord:
    """一台車的一行紀錄。欄位順序就是 CSV 的欄位順序。"""

    seq: int = 0
    # timestamp_iso  = 這一行被寫下來的時間(結算時刻,比擷取晚約 3 秒)
    # capture_wall_iso = 選中那一幀的影像擷取時間 —— 要算通行時間、跨站
    #   速度時看這一欄,不要看 timestamp_iso。
    timestamp_iso: str = ""
    capture_wall_iso: str = ""
    station_id: str = ""
    track_id: int = -1
    label: str = ""
    confidence: float = 0.0
    # --- 這台車總共被處理到幾幀,其中幾幀碰到邊界 ---
    total_frames: int = 0
    edge_frames: int = 0
    # --- 最後選中的是哪一幀,為什麼 ---
    chosen_frame_index: int = -1
    chosen_score: float = 0.0
    chosen_sharpness: float = 0.0
    chosen_from_edge_fallback: int = 0
    bbox_x1: float = 0.0
    bbox_y1: float = 0.0
    bbox_x2: float = 0.0
    bbox_y2: float = 0.0
    touches_edge: int = 0
    edge_distance: float = 0.0
    # 單調時鐘,只在同一次執行內有意義(量間隔用)。跨執行、跨機器請用
    # capture_wall_iso。
    frame_ns: int = 0
    # --- 尺寸:這是判斷車牌讀不讀得出來的關鍵數字 ---
    vehicle_crop_w: int = 0
    vehicle_crop_h: int = 0
    plate_box_w: int = 0
    plate_box_h: int = 0
    # --- OCR ---
    plate_text: str = ""
    plate_raw_text: str = ""
    plate_confidence: float = 0.0
    plate_is_taiwan_format: int = 0
    ocr_failure: str = ""
    # --- 結果 ---
    emitted: int = 0
    needs_review: int = 0
    review_reason: str = ""
    duplicate_of_track: int = -1
    image_sha256: str = ""
    crop_file: str = ""

    @classmethod
    def field_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    def as_row(self) -> dict[str, Any]:
        return asdict(self)


class PassageLogger:
    """把 PassageRecord 寫成 CSV,並存下對應的裁切圖。

    csv_path 或 crop_dir 給空字串就關閉該項。兩者都關閉時 enabled 為 False,
    呼叫端仍可安全呼叫 log()。
    """

    def __init__(self, csv_path: str, crop_dir: str) -> None:
        self._csv_path = Path(os.path.expanduser(csv_path)) if csv_path else None
        self._crop_dir = Path(os.path.expanduser(crop_dir)) if crop_dir else None
        self._seq = 0
        self._handle: Any = None
        self._writer: Any = None

        if self._csv_path is not None:
            self._csv_path.parent.mkdir(parents=True, exist_ok=True)
            # 續寫:重跑站台不會蓋掉上一場實驗的資料。
            is_new = not self._csv_path.exists() or self._csv_path.stat().st_size == 0
            self._seq = 0 if is_new else _last_seq(self._csv_path)
            self._handle = self._csv_path.open("a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._handle, fieldnames=PassageRecord.field_names()
            )
            if is_new:
                self._writer.writeheader()
                self._handle.flush()
            _logger.info(
                "passage CSV: %s(從第 %d 筆續寫)", self._csv_path, self._seq + 1
            )

        if self._crop_dir is not None:
            self._crop_dir.mkdir(parents=True, exist_ok=True)
            _logger.info("passage crops: %s", self._crop_dir)

    @property
    def enabled(self) -> bool:
        return self._writer is not None or self._crop_dir is not None

    def log(self, record: PassageRecord, crop_jpeg: bytes | None) -> None:
        if not self.enabled:
            return
        self._seq += 1
        record.seq = self._seq
        if not record.timestamp_iso:
            record.timestamp_iso = datetime.now().astimezone().isoformat(
                timespec="milliseconds"
            )

        if self._crop_dir is not None and crop_jpeg:
            record.crop_file = self._save_crop(record, crop_jpeg)

        if self._writer is not None:
            try:
                self._writer.writerow(record.as_row())
                self._handle.flush()
            except OSError as exc:
                _logger.warning("passage CSV write failed: %s", exc)

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None
            self._writer = None

    # --- internals ------------------------------------------------------

    def _save_crop(self, record: PassageRecord, crop_jpeg: bytes) -> str:
        # 檔名帶 seq(對應 CSV 那一行)、track id 與讀到的車牌,讀不出來的
        # 標 NOPLATE —— 之後要挑失敗案例來看,直接用檔名就找得到。
        plate = _UNSAFE_FILENAME.sub("", record.plate_text) or "NOPLATE"
        name = f"{record.seq:05d}_track{record.track_id}_{plate}.jpg"
        path = self._crop_dir / name if self._crop_dir else None
        if path is None:
            return ""
        try:
            path.write_bytes(crop_jpeg)
        except OSError as exc:
            _logger.warning("failed to save crop %s: %s", path, exc)
            return ""
        return name


def _last_seq(csv_path: Path) -> int:
    """讀出既有 CSV 的最後一個 seq,續寫時才不會從 1 重來。"""
    try:
        with csv_path.open("r", newline="", encoding="utf-8") as handle:
            last = 0
            for row in csv.DictReader(handle):
                try:
                    last = max(last, int(row.get("seq") or 0))
                except (TypeError, ValueError):
                    continue
            return last
    except (OSError, csv.Error) as exc:
        _logger.warning("could not read existing CSV %s: %s", csv_path, exc)
        return 0
