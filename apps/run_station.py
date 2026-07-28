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
from speed_trap.tracker import VehicleTracker
from speed_trap.trigger_line import TriggerLineDetector                                                                    

_logger = logging.getLogger(__name__)
_EMPTY_SHA256 = "0" * 64
_PRUNE_INTERVAL_NS = 1_000_000_000


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

def run_station(
    config: StationConfig,
    sink: EventSink,
    source: HailoDetectionSource,
    stop_flag: _StopFlag,
    recognizer: _PlateRecognizerLike | None = None,
) -> int:
    tracker = VehicleTracker()
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
                best = track.best_detection
                reading = (
                    plate_reader.read_from_jpeg(best.frame_jpeg)
                    if best.frame_jpeg
                    else None
                )

                event = PassageEvent(
                    station_id=config.station_id,
                    track_id=det.track_id,
                    label=det.label,
                    timestamp_ns=det.frame_ns,
                    confidence=det.confidence,
                    image_sha256=_hash_image(best.frame_jpeg),
                    plate_text=reading.text if reading else None,
                    plate_confidence=reading.confidence if reading else None,
                    plate_is_taiwan_format=reading.is_taiwan_format if reading else None,
                )
                sink.emit(event)
                trigger.mark_triggered(det.track_id)
                emitted += 1
                _logger.info(                                                                                  "passage emitted: track=%d label=%s conf=%.2f "
                    "plate=%s plate_conf=%s tw_format=%s",
                    det.track_id,
                    det.label,
                    det.confidence,
                    event.plate_text or "-",
                    f"{event.plate_confidence:.2f}" if event.plate_confidence else "-",
                    event.plate_is_taiwan_format,
                )

            now_ns = time.monotonic_ns()
            if now_ns - last_prune_ns > _PRUNE_INTERVAL_NS:
                tracker.prune_stale(now_ns)
                last_prune_ns = now_ns
    finally:
        source.stop()

    return emitted


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run speed-trap station with Hailo detection on Pi."
    )                                                                                                            
    parser.add_argument("--config", required=True, type=Path, help="station YAML config")
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
