"""Speed-trap station main loop.

結算時機:車子朝相機開過來,過了觸發線之後還會繼續變大、變清楚,所以
「過線的那一刻」通常不是最好的一幀。過線只用來標記這台車該記錄,實際挑
幀與 OCR 延到 track 結束(車子離開畫面、track_idle_timeout_s 沒再出現)
才做,這時整條 track 的所有幀都已經看過了。

代價是事件會晚 track_idle_timeout_s + 一個 prune 週期(約 3 秒)才送出。
對於事後查詢的通行紀錄沒有影響。
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from speed_trap.clock import log_clock_status, wall_ns_to_iso
from speed_trap.config import StationConfig, load_config
from speed_trap.event import EventSink, PassageEvent, make_sink
from speed_trap.hailo_source import HailoDetectionSource
from speed_trap.passage_log import PassageLogger, PassageRecord
from speed_trap.plate_recognizer import (
    NoopPlateRecognizer,
    PlateReading,
    make_recognizer,
    run_ocr,
)
from speed_trap.tracker import (
    ScoringParams,
    VehicleTrack,
    VehicleTracker,
    touches_edge,
)
from speed_trap.trigger_line import TriggerLineDetector

_logger = logging.getLogger(__name__)
_EMPTY_SHA256 = "0" * 64
_PRUNE_INTERVAL_NS = 1_000_000_000

# CSV / log 上標記需人工複核的原因代碼
REVIEW_ALL_FRAMES_EDGE = "all_frames_touch_edge"
REVIEW_NO_PLATE = "no_plate_read"
REVIEW_BAD_FORMAT = "not_taiwan_format"
REVIEW_DUPLICATE = "duplicate_plate"
REVIEW_NO_FRAME = "no_usable_frame"
REVIEW_TOO_FEW_FRAMES = "too_few_frames"


class _DuplicateGuard:
    """同一台車被 tracker 中途斷開、重新編號時會產生兩筆事件。

    用車牌字串在時間視窗內比對:視窗內出現過同樣的字串、而且是不同的
    track id,就當作同一台車的重複紀錄,寫進診斷資料但不送出。

    限制:讀不到車牌時無從比對,一律放行 —— 寧可多記一筆,也不要把兩台
    真的不同的車併成一筆。
    """

    def __init__(self, window_s: float) -> None:
        self._window_ns = int(window_s * 1_000_000_000)
        self._recent: deque[tuple[int, str, int]] = deque()

    @property
    def enabled(self) -> bool:
        return self._window_ns > 0

    def duplicate_of(self, plate: str | None, now_ns: int, track_id: int) -> int | None:
        """回傳先前那筆的 track_id 表示重複;None 表示不是重複。"""
        if not self.enabled or not plate:
            return None
        while self._recent and now_ns - self._recent[0][0] > self._window_ns:
            self._recent.popleft()
        for _ts, text, previous_id in self._recent:
            if text == plate and previous_id != track_id:
                return previous_id
        return None

    def remember(self, plate: str | None, now_ns: int, track_id: int) -> None:
        if self.enabled and plate:
            self._recent.append((now_ns, plate, track_id))


class _PlateRecognizerLike(Protocol):
    @property
    def model_name(self) -> str: ...
    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None: ...


def _hash_image(frame_jpeg: bytes | None) -> str:
    if frame_jpeg is None:
        return _EMPTY_SHA256
    return hashlib.sha256(frame_jpeg).hexdigest()


class _StopFlag:
    def __init__(self) -> None:
        self._stop = False

    def request_stop(self) -> None:
        self._stop = True

    @property
    def stopped(self) -> bool:
        return self._stop


def _settle_track(
    config: StationConfig,
    track: VehicleTrack,
    sink: EventSink,
    plate_reader: _PlateRecognizerLike,
    guard: _DuplicateGuard,
    passage_log: PassageLogger | None = None,
) -> bool:
    """挑幀 → OCR → 送出事件 → 寫診斷紀錄。回傳是否真的送出。

    不論送不送出都會寫一行 CSV 與一張裁切圖 —— 讀不出來的那幾台才是要
    回頭看的。
    """
    best = track.best_frame()
    if best is None:
        _logger.warning(
            "track %d (%s): %d 幀都沒有可用的裁切圖,不產生事件",
            track.track_id,
            track.label,
            track.frame_count,
        )
        if passage_log is not None:
            passage_log.log(
                PassageRecord(
                    station_id=config.station_id,
                    track_id=track.track_id,
                    label=track.label,
                    total_frames=track.frame_count,
                    edge_frames=track.edge_frame_count,
                    needs_review=1,
                    review_reason=REVIEW_NO_FRAME,
                ),
                None,
            )
        return False

    review_reasons: list[str] = []
    if best.from_edge_fallback:
        # 保護機制:整條 track 每一幀都碰到邊界。還是送 OCR(總比放棄好),
        # 但標記為需人工複核。
        _logger.warning(
            "track %d (%s): 全部 %d 幀都碰到畫面邊界,退而取最不靠邊的第 %d 幀"
            "(距邊界 %.3f),需人工複核",
            track.track_id,
            track.label,
            track.frame_count,
            best.frame_index,
            best.edge_distance,
        )
        review_reasons.append(REVIEW_ALL_FRAMES_EDGE)

    det = best.detection
    attempt = run_ocr(plate_reader, det.frame_jpeg)
    reading = attempt.reading
    if reading is None:
        review_reasons.append(REVIEW_NO_PLATE)
    elif not reading.is_taiwan_format:
        review_reasons.append(REVIEW_BAD_FORMAT)

    plate_text = reading.text if reading else None
    duplicate_of = guard.duplicate_of(plate_text, det.frame_ns, track.track_id)
    if duplicate_of is not None:
        review_reasons.append(REVIEW_DUPLICATE)
        _logger.warning(
            "track %d (%s) 讀到 %s,與 track %d 在 %.0f 秒內重複 —— "
            "多半是同一台車被 tracker 斷開重新編號,不送出事件",
            track.track_id,
            track.label,
            plate_text,
            duplicate_of,
            config.duplicate_window_s,
        )
    else:
        guard.remember(plate_text, det.frame_ns, track.track_id)

    event = PassageEvent(
        station_id=config.station_id,
        track_id=track.track_id,
        label=track.label,
        # 對外的時間戳用真實時間,不是單調時鐘(見 speed_trap.clock)。
        timestamp_ns=det.capture_wall_ns,
        confidence=det.confidence,
        image_sha256=_hash_image(det.frame_jpeg),
        plate_text=plate_text,
        plate_confidence=reading.confidence if reading else None,
        plate_is_taiwan_format=reading.is_taiwan_format if reading else None,
        needs_review=bool(review_reasons),
        review_reason=";".join(review_reasons) or None,
    )
    emitted = duplicate_of is None
    if emitted:
        sink.emit(event)

    if passage_log is not None:
        passage_log.log(
            _make_record(config, track, best, det, attempt, event, emitted, duplicate_of),
            det.frame_jpeg,
        )

    _logger.info(
        "passage %s: track=%d label=%s frames=%d (edge %d) chosen=#%d "
        "score=%.3f sharpness=%.0f crop=%s plate_box=%s plate=%s "
        "plate_conf=%s tw_format=%s review=%s",
        "emitted" if emitted else "suppressed",
        track.track_id,
        track.label,
        track.frame_count,
        track.edge_frame_count,
        best.frame_index,
        best.score,
        det.sharpness,
        _fmt_wh(det.crop_size or attempt.vehicle_wh),
        _fmt_wh(attempt.plate_wh),
        event.plate_text or "-",
        f"{event.plate_confidence:.2f}" if event.plate_confidence else "-",
        event.plate_is_taiwan_format,
        event.review_reason or "-",
    )
    return emitted


def _fmt_wh(wh: tuple[int, int] | None) -> str:
    return f"{wh[0]}x{wh[1]}" if wh else "?"


def _make_record(
    config: StationConfig,
    track: VehicleTrack,
    best: Any,
    det: Any,
    attempt: Any,
    event: PassageEvent,
    emitted: bool,
    duplicate_of: int | None,
) -> PassageRecord:
    crop_wh = det.crop_size or attempt.vehicle_wh or (0, 0)
    plate_wh = attempt.plate_wh or (0, 0)
    reading = attempt.reading
    return PassageRecord(
        capture_wall_iso=wall_ns_to_iso(det.capture_wall_ns),
        station_id=config.station_id,
        track_id=track.track_id,
        label=track.label,
        confidence=round(det.confidence, 4),
        total_frames=track.frame_count,
        edge_frames=track.edge_frame_count,
        chosen_frame_index=best.frame_index,
        chosen_score=round(best.score, 5),
        chosen_sharpness=round(det.sharpness, 1),
        chosen_from_edge_fallback=int(best.from_edge_fallback),
        bbox_x1=round(det.bbox[0], 5),
        bbox_y1=round(det.bbox[1], 5),
        bbox_x2=round(det.bbox[2], 5),
        bbox_y2=round(det.bbox[3], 5),
        touches_edge=int(touches_edge(det.bbox, config.edge_margin)),
        edge_distance=round(best.edge_distance, 5),
        frame_ns=det.frame_ns,
        vehicle_crop_w=crop_wh[0],
        vehicle_crop_h=crop_wh[1],
        plate_box_w=plate_wh[0],
        plate_box_h=plate_wh[1],
        plate_text=event.plate_text or "",
        plate_raw_text=reading.raw_text if reading else "",
        plate_confidence=round(reading.confidence, 4) if reading else 0.0,
        plate_is_taiwan_format=int(bool(reading and reading.is_taiwan_format)),
        ocr_failure=attempt.failure or "",
        emitted=int(emitted),
        needs_review=int(event.needs_review),
        review_reason=event.review_reason or "",
        duplicate_of_track=duplicate_of if duplicate_of is not None else -1,
        image_sha256=event.image_sha256,
    )


def _settle_finished(
    config: StationConfig,
    tracks: list[VehicleTrack],
    trigger: TriggerLineDetector,
    sink: EventSink,
    plate_reader: _PlateRecognizerLike,
    guard: _DuplicateGuard | None = None,
    passage_log: PassageLogger | None = None,
) -> int:
    guard = guard if guard is not None else _DuplicateGuard(config.duplicate_window_s)
    emitted = 0
    for track in tracks:
        triggered = trigger.was_triggered(track.track_id)
        # 不論有沒有過線都要清掉狀態:_states 原本只增不減,而且 track id
        # 會被 hailotracker 回收再利用。
        trigger.forget(track.track_id)

        if not triggered:
            # 沒過線的 track(還沒開到、或只是畫面邊緣晃過)不算通行,
            # 也不寫診斷紀錄,否則 CSV 會被路邊靜物洗掉。
            continue
        if track.frame_count < config.min_track_frames:
            _logger.info(
                "track %d (%s) 只出現 %d 幀(門檻 %d),視為雜訊不計入通行",
                track.track_id,
                track.label,
                track.frame_count,
                config.min_track_frames,
            )
            if passage_log is not None:
                best = track.best_frame()
                passage_log.log(
                    PassageRecord(
                        capture_wall_iso=(
                            wall_ns_to_iso(best.detection.capture_wall_ns)
                            if best
                            else ""
                        ),
                        station_id=config.station_id,
                        track_id=track.track_id,
                        label=track.label,
                        total_frames=track.frame_count,
                        edge_frames=track.edge_frame_count,
                        chosen_frame_index=best.frame_index if best else -1,
                        needs_review=1,
                        review_reason=REVIEW_TOO_FEW_FRAMES,
                    ),
                    best.detection.frame_jpeg if best else None,
                )
            continue
        if _settle_track(config, track, sink, plate_reader, guard, passage_log):
            emitted += 1
    return emitted


def run_station(
    config: StationConfig,
    sink: EventSink,
    source: HailoDetectionSource,
    stop_flag: _StopFlag,
    recognizer: _PlateRecognizerLike | None = None,
    passage_log: PassageLogger | None = None,
) -> int:
    tracker = VehicleTracker(
        stale_timeout_ns=int(config.track_idle_timeout_s * 1_000_000_000),
        scoring=ScoringParams.from_config(config),
    )
    trigger = TriggerLineDetector(line_y=config.trigger_line_y)
    plate_reader: _PlateRecognizerLike = recognizer or NoopPlateRecognizer()
    guard = _DuplicateGuard(config.duplicate_window_s)
    vehicle_classes = set(config.vehicle_classes)
    emitted = 0
    skipped_non_vehicle = 0
    last_prune_ns = time.monotonic_ns()

    source.start()
    try:
        for det in source.iter_detections():
            if stop_flag.stopped:
                break

            if det.label not in vehicle_classes:
                # 來源(hailo_source)已經依 vehicle_classes 過濾過,這裡是
                # 防呆:確保無論來源怎麼換,只有車輛類別會產生通行事件。
                # RTSP 畫面上看得到行人與盆栽的框,是因為 hailooverlay 畫的
                # 是過濾前的全部偵測,那條線不影響這裡的計數。
                skipped_non_vehicle += 1
                continue

            tracker.update([det])
            track = tracker.get_track(det.track_id)
            if track is None:
                continue

            crossed = trigger.check_crossing(track, det)
            if crossed and not trigger.was_triggered(det.track_id):
                # 只記錄「這台車該算一次通行」,不在此結算 —— 車子還在
                # 靠近,更好的幀多半還沒發生。
                trigger.mark_triggered(det.track_id)
                _logger.info(
                    "track %d (%s) 通過觸發線,等離開畫面後結算",
                    det.track_id,
                    det.label,
                )

            now_ns = time.monotonic_ns()
            if now_ns - last_prune_ns > _PRUNE_INTERVAL_NS:
                emitted += _settle_finished(
                    config,
                    tracker.prune_stale(now_ns),
                    trigger,
                    sink,
                    plate_reader,
                    guard,
                    passage_log,
                )
                last_prune_ns = now_ns
    finally:
        source.stop()

    # 收工:畫面上還在的車,只要已經過線就補結算,不要因為停止而漏掉。
    emitted += _settle_finished(
        config, tracker.drain(), trigger, sink, plate_reader, guard, passage_log
    )
    if skipped_non_vehicle:
        _logger.info(
            "略過 %d 筆非車輛類別的偵測(vehicle_classes=%s)",
            skipped_non_vehicle,
            sorted(vehicle_classes),
        )
    _logger.info(
        "trigger 狀態表殘留 %d 筆(正常應為 0)", trigger.tracked_count
    )
    return emitted


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run speed-trap station with Hailo detection on Pi."
    )
    parser.add_argument(
        "--config", required=True, type=Path, help="station YAML config"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    config = load_config(args.config)

    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    # 時間戳的可信度先講清楚:沒對到時的話,通行時間與跨站速度都不能用。
    log_clock_status()

    sink = make_sink(config)
    source = HailoDetectionSource(config)
    recognizer = make_recognizer(config)
    passage_log = PassageLogger(config.passage_csv_path, config.passage_crop_dir)
    stop_flag = _StopFlag()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        _logger.info("received signal %d, requesting stop", signum)
        stop_flag.request_stop()

    previous_handlers: dict[int, Any] = {
        signal.SIGINT: signal.signal(signal.SIGINT, _handle_signal),
        signal.SIGTERM: signal.signal(signal.SIGTERM, _handle_signal),
    }

    try:
        emitted = run_station(
            config,
            sink,
            source,
            stop_flag,
            recognizer=recognizer,
            passage_log=passage_log,
        )
    finally:
        sink.close()
        passage_log.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    _logger.info("station stopped — %d passage events emitted", emitted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
