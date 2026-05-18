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
_cv2: Any = None
_np: Any = None

try:
    import gi

    gi.require_version("Gst", "1.0")
    import cv2 as _cv2_real
    import hailo as _hailo
    import numpy as _np_real
    from gi.repository import GLib as _GLib
    from gi.repository import Gst as _Gst

    _Gst.init(None)
    Gst = _Gst
    GLib = _GLib
    hailo = _hailo
    _cv2 = _cv2_real
    _np = _np_real
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - hardware-only path
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


_QUEUE_MAX = 512
_HAILO_INPUT_SIZE = 640  # pipeline scales every frame to 640x640 RGB before hailonet
_JPEG_QUALITY = 85


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
        # Don't pin width/height/framerate after libcamerasrc — Pi 5's libcamera
        # + pisp ISP picks a sensor-native mode (e.g. IMX708 emits 2304x1296),
        # then videoconvert + videoscale step it down to the HEF's input shape.
        #
        # We use plain `videoscale` (squish) rather than `add-borders=true`
        # (letterbox). Letterbox is "theoretically" the right thing — it
        # preserves aspect ratio for the detector — but on the camera-of-screen
        # YouTube test it dropped recall from 62% / 36% to ~10% because
        # fitting 1280x1080 -> 640x540 inside a 640x640 canvas leaves only
        # ~84% of pixel rows for actual content; small/far cars hit the
        # detection threshold floor. Squish keeps more pixels per object at
        # the cost of unnatural aspect, which YOLO tolerates well.
        # When we move to real lamppost deployment (cars larger in frame),
        # revisit add-borders=true + pixel-aspect-ratio=1/1.
        # YOLOv6n / YOLOv8s / YOLOX-s on Hailo-8(L) all expect 640x640 RGB.
        hailo_input = "video/x-raw,format=RGB,width=640,height=640"
        return (
            "libcamerasrc ! "
            "videoconvert ! "
            "videoscale ! "
            f"{hailo_input} ! "
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
        if not detections:
            return Gst.PadProbeReturn.OK

        frame_ns = time.monotonic_ns()
        # Extract the 640x640 RGB frame once per buffer (not per detection).
        # If extraction fails we still emit Detection records but with frame_jpeg=None,
        # so the consumer can fall back to no-OCR mode.
        frame_rgb = self._extract_frame_rgb(buffer)

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

            frame_jpeg = None
            if frame_rgb is not None:
                frame_jpeg = self._crop_and_encode_jpeg(frame_rgb, (x1, y1, x2, y2))

            try:
                self._queue.put_nowait(
                    Detection(
                        track_id=track_id,
                        label=label,
                        bbox=(x1, y1, x2, y2),
                        confidence=float(det.get_confidence()),
                        frame_ns=frame_ns,
                        frame_jpeg=frame_jpeg,
                    )
                )
            except queue.Full:
                self._dropped += 1

        return Gst.PadProbeReturn.OK

    def _extract_frame_rgb(self, buffer: Any) -> Any:
        """Map the GStreamer buffer and return the RGB pixel data as a numpy
        array of shape (640, 640, 3). Returns None on failure (logged once)."""
        try:
            success, mapinfo = buffer.map(Gst.MapFlags.READ)
        except Exception:
            return None
        if not success:
            return None
        try:
            expected_size = _HAILO_INPUT_SIZE * _HAILO_INPUT_SIZE * 3
            data = bytes(mapinfo.data)
            if len(data) < expected_size:
                # Pipeline doesn't actually carry the pixel buffer through to
                # fakesink (some configurations strip the payload). In that
                # case we can't crop — caller will see frame_jpeg=None and
                # skip OCR for this passage.
                return None
            arr = _np.frombuffer(data[:expected_size], dtype=_np.uint8)
            return arr.reshape(_HAILO_INPUT_SIZE, _HAILO_INPUT_SIZE, 3)
        finally:
            buffer.unmap(mapinfo)

    def _crop_and_encode_jpeg(
        self,
        frame_rgb: Any,
        bbox: tuple[float, float, float, float],
    ) -> bytes | None:
        """Crop the bbox region from a 640x640 RGB frame and encode as JPEG.
        Coordinates are normalised 0..1; returns None for degenerate boxes
        or encoding failure."""
        x1, y1, x2, y2 = bbox
        h, w = frame_rgb.shape[:2]
        # Clamp to frame, pad a small margin around the vehicle so the plate
        # at the bumper isn't sliced.
        margin = 0.02
        px1 = max(0, int((x1 - margin) * w))
        py1 = max(0, int((y1 - margin) * h))
        px2 = min(w, int((x2 + margin) * w))
        py2 = min(h, int((y2 + margin) * h))
        if px2 - px1 < 16 or py2 - py1 < 16:
            return None

        crop_rgb = frame_rgb[py1:py2, px1:px2]
        crop_bgr = _cv2.cvtColor(crop_rgb, _cv2.COLOR_RGB2BGR)
        ok, jpeg = _cv2.imencode(
            ".jpg", crop_bgr, [int(_cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY]
        )
        if not ok:
            return None
        return bytes(jpeg)

    def _on_bus_message(self, _bus: Any, message: Any) -> None:
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            _logger.info("gstreamer EOS received, stopping source")
            self._running = False
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            _logger.error("gstreamer error: %s (%s)", err, debug)
            self._running = False
