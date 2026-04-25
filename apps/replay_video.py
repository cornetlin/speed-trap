from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import cv2

from speed_trap.config import StationConfig, load_config
from speed_trap.event import EventSink, PassageEvent, make_sink
from speed_trap.mock_detector import MockDetector
from speed_trap.tracker import VehicleTracker
from speed_trap.trigger_line import TriggerLineDetector

_logger = logging.getLogger(__name__)

_EMPTY_SHA256 = "0" * 64


def _hash_image(frame_jpeg: bytes | None) -> str:
    if frame_jpeg is None:
        return _EMPTY_SHA256
    return hashlib.sha256(frame_jpeg).hexdigest()


def run_replay(
    video_path: Path,
    config: StationConfig,
    sink: EventSink,
    *,
    detector: MockDetector | None = None,
    show_display: bool = False,
) -> list[PassageEvent]:
    """Run the replay pipeline on a video file. Returns events emitted."""
    detector = detector or MockDetector()
    tracker = VehicleTracker()
    trigger = TriggerLineDetector(line_y=config.trigger_line_y)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"cannot open video: {video_path}")

    emitted: list[PassageEvent] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_ns = time.monotonic_ns()
            detections = detector.detect(frame, frame_index, frame_ns)
            tracker.update(detections)

            for det in detections:
                track = tracker.get_track(det.track_id)
                if track is None:
                    continue
                crossed = trigger.check_crossing(track, det)
                if crossed and not trigger.was_triggered(det.track_id):
                    event = PassageEvent(
                        station_id=config.station_id,
                        track_id=det.track_id,
                        label=det.label,
                        timestamp_ns=det.frame_ns,
                        confidence=det.confidence,
                        image_sha256=_hash_image(det.frame_jpeg),
                    )
                    sink.emit(event)
                    emitted.append(event)
                    trigger.mark_triggered(det.track_id)

            if show_display:
                cv2.imshow("speed-trap replay", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show_display:
            cv2.destroyAllWindows()

    return emitted


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a video through the speed-trap pipeline using MockDetector."
    )
    parser.add_argument("--video", required=True, type=Path, help="input video file")
    parser.add_argument("--config", required=True, type=Path, help="station YAML config")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="run headless (no OpenCV window) — required on Pi / CI",
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
    try:
        events = run_replay(
            args.video,
            config,
            sink,
            show_display=not args.no_display,
        )
    finally:
        sink.close()

    _logger.info("replay finished — %d passage events emitted", len(events))
    return 0


if __name__ == "__main__":
    sys.exit(main())
