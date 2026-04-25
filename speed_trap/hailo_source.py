"""Hailo + GStreamer detection source for Raspberry Pi 5 + Hailo AI HAT+.

Pipeline:
    libcamerasrc -> videoconvert -> hailonet -> hailofilter -> hailotracker -> fakesink

Detection objects are extracted from HAILO_ROI metadata in a fakesink pad probe
callback and pushed onto a thread-safe queue that ``iter_detections`` drains.

Importing this module on a non-Pi machine (no PyGObject / Hailo runtime) is
deliberately safe: the heavy imports live inside a try/except. Only when you
*instantiate* :class:`HailoDetectionSource` will it raise a RuntimeError that
explains what's missing.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Iterator
from typing import Any

from speed_trap.config import StationConfig
from speed_trap.tracker import Detection

_logger = logging.getLogger(__name__)

_IMPORT_ERROR: str | None
Gst: Any = None
GLib: Any = None
hailo: Any = None

try:
    import gi

    gi.require_version("Gst", "1.0")
    import hailo as _hailo
    from gi.repository import GLib as _GLib
    from gi.repository import Gst as _Gst

    _Gst.init(None)
    Gst = _Gst
    GLib = _GLib
    hailo = _hailo
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - hardware-only path
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


_QUEUE_MAX = 512


class HailoDetectionSource:
    """Wraps a GStreamer + Hailo pipeline and yields :class:`Detection` objects.

    Lifecycle:
        source = HailoDetectionSource(config)
        source.start()
        for det in source.iter_detections():
            ...
        source.stop()
    """

    def __init__(
        self,
        config: StationConfig,
        *,
        queue_max: int = _QUEUE_MAX,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "Hailo runtime not available on this platform "
                f"({_IMPORT_ERROR}). Install PyGObject (gi) + hailo-python; "
                "this source only runs on Raspberry Pi with the Hailo HAT."
            )

        self._config = config
        self._vehicle_classes = set(config.vehicle_classes)

        self._pipeline: Any = None
        self._loop: Any = None
        self._loop_thread: threading.Thread | None = None
        self._queue: queue.Queue[Detection] = queue.Queue(maxsize=queue_max)
        self._running = False
        self._dropped = 0

    # --- public API -----------------------------------------------------

    def start(self) -> None:
        if self._running:
            return

        pipeline_str = self._build_pipeline_str()
        _logger.info("starting GStreamer pipeline: %s", pipeline_str)

        self._pipeline = Gst.parse_launch(pipeline_str)
        sink = self._pipeline.get_by_name("speedtrap_sink")
        if sink is None:
            raise RuntimeError("speedtrap_sink element missing from pipeline")

        sink_pad = sink.get_static_pad("sink")
        sink_pad.add_probe(Gst.PadProbeType.BUFFER, self._on_buffer, None)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._pipeline.set_state(Gst.State.PLAYING)
        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._loop.run, name="hailo-glib-loop", daemon=True
        )
        self._loop_thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop is not None:
            self._loop.quit()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)

        if self._dropped:
            _logger.warning(
                "dropped %d detections due to full queue during run", self._dropped
            )

    def iter_detections(self, *, poll_interval_s: float = 0.1) -> Iterator[Detection]:
        """Yield detections until :meth:`stop` is called.

        Polls the internal queue with a small timeout so the consumer loop
        can interleave signal checks (Ctrl+C, SIGTERM) between yields.
        """
        while self._running:
            try:
                yield self._queue.get(timeout=poll_interval_s)
            except queue.Empty:
                continue

    # --- internals ------------------------------------------------------

    def _build_pipeline_str(self) -> str:
        cfg = self._config
        return (
            "libcamerasrc ! "
            f"video/x-raw,width={cfg.frame_width},height={cfg.frame_height},"
            f"framerate={cfg.frame_fps}/1 ! "
            "videoconvert ! "
            f"hailonet hef-path={cfg.hef_path} batch-size=1 ! "
            f"hailofilter so-path={cfg.hailofilter_so_path} qos=false ! "
            "hailotracker name=tracker keep-tracked-frames=10 keep-new-frames=10 ! "
            "fakesink name=speedtrap_sink sync=false async=false"
        )

    def _on_buffer(self, _pad: Any, info: Any, _user_data: Any) -> Any:
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        frame_ns = time.monotonic_ns()

        for det in detections:
            label = det.get_label()
            if label not in self._vehicle_classes:
                continue

            bbox = det.get_bbox()
            x1 = float(bbox.xmin())
            y1 = float(bbox.ymin())
            x2 = x1 + float(bbox.width())
            y2 = y1 + float(bbox.height())

            track_objs = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            track_id = int(track_objs[0].get_id()) if track_objs else -1

            try:
                self._queue.put_nowait(
                    Detection(
                        track_id=track_id,
                        label=label,
                        bbox=(x1, y1, x2, y2),
                        confidence=float(det.get_confidence()),
                        frame_ns=frame_ns,
                        frame_jpeg=None,
                    )
                )
            except queue.Full:
                self._dropped += 1

        return Gst.PadProbeReturn.OK

    def _on_bus_message(self, _bus: Any, message: Any) -> None:
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            _logger.info("gstreamer EOS received, stopping source")
            self._running = False
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            _logger.error("gstreamer error: %s (%s)", err, debug)
            self._running = False
