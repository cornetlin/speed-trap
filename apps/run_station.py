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
from pathlib import Path
from types import FrameType
from typing import Any, Protocol

from speed_trap.config import StationConfig, load_config
from speed_trap.event import EventSink, PassageEvent, make_sink
from speed_trap.hailo_source import HailoDetectionSource
from speed_trap.plate_recognizer import (
    NoopPlateRecognizer,
    PlateReading,
    make_recognizer,
)
from speed_trap.tracker import ScoringParams, VehicleTrack, VehicleTracker
from speed_trap.trigger_line import TriggerLineDetector

_logger = logging.getLogger(__name__)
_EMPTY_SHA256 = "0" * 64
_PRUNE_INTERVAL_NS = 1_000_000_000

# CSV / log 上標記需人工複核的原因代碼
REVIEW_ALL_FRAMES_EDGE = "all_frames_touch_edge"
REVIEW_NO_PLATE = "no_plate_read"
REVIEW_BAD_FORMAT = "not_taiwan_format"


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
) -> bool:
    """挑幀 → OCR → 送出事件。回傳是否真的送出。"""
    best = track.best_frame()
    if best is None:
        _logger.warning(
            "track %d (%s): %d 幀都沒有可用的裁切圖,不產生事件",
            track.track_id,
            track.label,
            track.frame_count,
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
    reading = (
        plate_reader.read_from_jpeg(det.frame_jpeg) if det.frame_jpeg else None
    )
    if reading is None:
        review_reasons.append(REVIEW_NO_PLATE)
    elif not reading.is_taiwan_format:
        review_reasons.append(REVIEW_BAD_FORMAT)

    event = PassageEvent(
        station_id=config.station_id,
        track_id=track.track_id,
        label=track.label,
        timestamp_ns=det.frame_ns,
        confidence=det.confidence,
        image_sha256=_hash_image(det.frame_jpeg),
        plate_text=reading.text if reading else None,
        plate_confidence=reading.confidence if reading else None,
        plate_is_taiwan_format=reading.is_taiwan_format if reading else None,
        needs_review=bool(review_reasons),
        review_reason=";".join(review_reasons) or None,
    )
    sink.emit(event)

    _logger.info(
        "passage emitted: track=%d label=%s frames=%d (edge %d) "
        "chosen=#%d score=%.3f sharpness=%.0f plate=%s plate_conf=%s "
        "tw_format=%s review=%s",
        track.track_id,
        track.label,
        track.frame_count,
        track.edge_frame_count,
        best.frame_index,
        best.score,
        det.sharpness,
        event.plate_text or "-",
        f"{event.plate_confidence:.2f}" if event.plate_confidence else "-",
        event.plate_is_taiwan_format,
        event.review_reason or "-",
    )
    return True


def _settle_finished(
    config: StationConfig,
    tracks: list[VehicleTrack],
    trigger: TriggerLineDetector,
    sink: EventSink,
    plate_reader: _PlateRecognizerLike,
) -> int:
    emitted = 0
    for track in tracks:
        if not trigger.was_triggered(track.track_id):
            # 沒過線的 track(還沒開到、或只是畫面邊緣晃過)不算通行。
            continue
        if _settle_track(config, track, sink, plate_reader):
            emitted += 1
    return emitted


def run_station(
    config: StationConfig,
    sink: EventSink,
    source: HailoDetectionSource,
    stop_flag: _StopFlag,
    recognizer: _PlateRecognizerLike | None = None,
) -> int:
    tracker = VehicleTracker(
        stale_timeout_ns=int(config.track_idle_timeout_s * 1_000_000_000),
        scoring=ScoringParams.from_config(config),
    )
    trigger = TriggerLineDetector(line_y=config.trigger_line_y)
    plate_reader: _PlateRecognizerLike = recognizer or NoopPlateRecognizer()
    emitted = 0
    last_prune_ns = time.monotonic_ns()

    source.start()
    try:
        for det in source.iter_detections():
            if stop_flag.stopped:
                break

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
                )
                last_prune_ns = now_ns
    finally:
        source.stop()

    # 收工:畫面上還在的車,只要已經過線就補結算,不要因為停止而漏掉。
    emitted += _settle_finished(
        config, tracker.drain(), trigger, sink, plate_reader
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

    sink = make_sink(config)
    source = HailoDetectionSource(config)
    recognizer = make_recognizer(config)
    stop_flag = _StopFlag()

    def _handle_signal(signum: int, _frame: FrameType | None) -> None:
        _logger.info("received signal %d, requesting stop", signum)
        stop_flag.request_stop()

    previous_handlers: dict[int, Any] = {
        signal.SIGINT: signal.signal(signal.SIGINT, _handle_signal),
        signal.SIGTERM: signal.signal(signal.SIGTERM, _handle_signal),
    }

    try:
        emitted = run_station(config, sink, source, stop_flag, recognizer=recognizer)
    finally:
        sink.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)

    _logger.info("station stopped — %d passage events emitted", emitted)
    return 0


if __name__ == "__main__":
    sys.exit(main())
