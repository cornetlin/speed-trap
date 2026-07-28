"""Hailo + GStreamer detection source for Raspberry Pi 5 + Hailo AI HAT+.

Pipeline — one camera, split by a tee right after the source:

    libcamerasrc ─┬─ videoscale → hailonet → hailofilter → hailotracker ─┬─ fakesink   (偵測線)
                  │                                                     └─ hailooverlay → RTSP (直播線)
                  └─ appsink                                                            (原始解析度取圖線)

車輛偵測必須跑在 HEF 規定的輸入尺寸(預設 640x640),但在那張縮圖上裁切車輛區域,
車牌只剩下幾十個像素寬,OCR 讀不出來。因此偵測與取圖分開:偵測線照舊縮圖,
取圖線保留 config 指定的原始解析度並存入環形緩衝區,以 buffer PTS 為 key;
偵測結果帶著同一個 PTS 回頭取出原始畫面,再依 bbox 裁切。

座標對應:hailofilter 給的 bbox 是 normalised 0..1。偵測線的 videoscale 明確
設定 ``add-borders=false`` 且輸出 caps 固定 ``pixel-aspect-ratio=1/1``,因此縮放
是純粹的等比例壓縮(不加黑邊),normalised 座標可以直接線性映射回原始畫面。
若改成加黑邊的 letterbox,下面的裁切必須先扣掉黑邊,否則會裁錯位置。

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
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from speed_trap.clock import wall_clock_ns
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

    # Importing GstApp registers the AppSink GType so ``pull-sample`` is
    # available on the element we fetch out of the parsed pipeline. Not fatal
    # if the typelib is missing — the plugin itself is loaded by parse_launch.
    try:
        gi.require_version("GstApp", "1.0")
        from gi.repository import GstApp as _GstApp  # noqa: F401
    except Exception:  # pragma: no cover - hardware-only path
        pass

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

# 相機輸出格式必須寫死在 caps 裡。留空讓 GStreamer 自行協商的話,結果會隨
# libcamera 版本與相機模式改變,同一份程式會時好時壞。
#
# 選 RGB 而不是 NV12 的理由(兩者都試算過,見下):取圖線需要的就是全解析度
# RGB,而 Pi 5 的 ISP 可以直接輸出 RGB,這個轉換不花 CPU。若來源改成 NV12,
# 取圖線就得自己用 videoconvert 把 1080p 轉成 RGB —— 等於把 ISP 免費做的事
# 搬到 CPU 上做。NV12 省的是相機到記憶體那一段頻寬,但省不到我們真正需要
# RGB 的那一段,而 tee 本身不複製 buffer(只加 refcount),所以「送到三條
# 分支」並不會讓頻寬變成三倍。
#
# 要 A/B 實測時改這一個字串即可,其餘不用動。
_CAMERA_FORMAT = "RGB"

# How many full-resolution frames to keep around. The detection branch runs
# behind the raw branch by however long hailonet + hailofilter + hailotracker
# take, so the ring only has to cover that lag. 30 frames at 30 fps is a full
# second of slack; at 1920x1080 RGB that is ~180 MB resident.
_RAW_RING_SIZE = 30

# Crop margin around the vehicle bbox, in normalised units, so a plate sitting
# right on the bumper edge isn't sliced off.
_CROP_MARGIN = 0.02

# Reject crops smaller than this (pixels) on either axis — nothing useful can
# be OCR'd out of them.
_MIN_CROP_PX = 16

# 算清晰度前先把裁切圖縮到這個寬度。Laplacian 變異數會隨解析度變動,同一
# 台車在大張裁切圖上算出來的值天生就比小張的高 —— 固定寬度之後,不同大小
# 的裁切圖才可以互相比較。也順便讓這步的成本跟裁切大小無關(約 1 ms)。
_SHARPNESS_WIDTH = 320

# How often the pipeline health line is logged.
_STATS_INTERVAL_S = 10.0

# 一個 track_id 多久沒再出現,就當作那台車已經離開畫面、可以結算它被處理到
# 幾幀。設 2 秒:比 hailotracker 的 keep-tracked-frames 寬鬆,免得車子只是被
# 短暫遮住就被算成兩台。
_TRACK_IDLE_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class _CropResult:
    jpeg: bytes
    width: int
    height: int
    sharpness: float


def _crop_sharpness(crop_bgr: Any) -> float:
    """Laplacian variance of a crop — higher means sharper.

    Motion blur is the failure mode this is meant to catch: a car crossing the
    frame at speed can be perfectly framed and still unreadable. Computed on a
    fixed-width grayscale copy so values are comparable between a 300 px crop
    of a distant car and a 900 px crop of a near one.
    """
    try:
        gray = _cv2.cvtColor(crop_bgr, _cv2.COLOR_BGR2GRAY)
        height, width = gray.shape[:2]
        if width > _SHARPNESS_WIDTH:
            scale = _SHARPNESS_WIDTH / width
            gray = _cv2.resize(
                gray,
                (_SHARPNESS_WIDTH, max(1, int(height * scale))),
                interpolation=_cv2.INTER_AREA,
            )
        return float(_cv2.Laplacian(gray, _cv2.CV_64F).var())
    except Exception as exc:  # noqa: BLE001 - scoring must never kill the probe
        _logger.debug("sharpness calculation failed: %s", exc)
        return 0.0


class _PipelineStats:
    """Rolling counters for the ``pipeline ...`` health line.

    Written from the GStreamer streaming thread (detection probe) and the
    appsink thread, read from the reporter thread — everything goes through
    one lock. No GStreamer types in here, so it is unit-testable off the Pi.

    The number that matters most is frames-per-vehicle: a car is in frame for
    roughly 1.5 s, so 30 fps means ~45 frames to pick a best shot from and
    8 fps means ~12. Everything downstream (best-frame selection, edge
    exclusion) is only as good as that number.
    """

    def __init__(self, *, track_idle_timeout_s: float = _TRACK_IDLE_TIMEOUT_S) -> None:
        self._lock = threading.Lock()
        self._idle_ns = int(track_idle_timeout_s * 1_000_000_000)
        # Tracks currently in frame: id -> frames seen so far / last seen.
        self._track_frames: dict[int, int] = {}
        self._track_last_ns: dict[int, int] = {}
        self._lifetime_hits = 0
        self._lifetime_misses = 0
        self._window_start_ns = time.monotonic_ns()
        self._detect_frames = 0
        self._raw_frames = 0
        self._hits = 0
        self._misses = 0
        # Frame counts of vehicles that left the frame during this window.
        self._done_tracks: list[int] = []

    def note_detect_frame(self) -> None:
        with self._lock:
            self._detect_frames += 1

    def note_raw_frame(self) -> None:
        with self._lock:
            self._raw_frames += 1

    def note_lookup(self, *, hit: bool) -> None:
        with self._lock:
            if hit:
                self._hits += 1
                self._lifetime_hits += 1
            else:
                self._misses += 1
                self._lifetime_misses += 1

    def note_track_frame(self, track_id: int, now_ns: int) -> None:
        # track_id < 0 means hailotracker gave us no unique id; lumping those
        # together would invent one vehicle with a huge frame count.
        if track_id < 0:
            return
        with self._lock:
            self._track_frames[track_id] = self._track_frames.get(track_id, 0) + 1
            self._track_last_ns[track_id] = now_ns

    def collect(self, now_ns: int) -> dict[str, Any]:
        """Retire idle tracks, then snapshot and reset the window."""
        with self._lock:
            for track_id in [
                t
                for t, last in self._track_last_ns.items()
                if now_ns - last > self._idle_ns
            ]:
                self._done_tracks.append(self._track_frames.pop(track_id, 0))
                del self._track_last_ns[track_id]

            elapsed_s = max(1e-9, (now_ns - self._window_start_ns) / 1_000_000_000)
            snapshot = {
                "elapsed_s": elapsed_s,
                "detect_fps": self._detect_frames / elapsed_s,
                "raw_fps": self._raw_frames / elapsed_s,
                "hits": self._hits,
                "misses": self._misses,
                "done": list(self._done_tracks),
                "in_frame": len(self._track_frames),
            }

            self._window_start_ns = now_ns
            self._detect_frames = 0
            self._raw_frames = 0
            self._hits = 0
            self._misses = 0
            self._done_tracks.clear()
        return snapshot

    def lifetime_lookups(self) -> tuple[int, int]:
        with self._lock:
            return self._lifetime_hits, self._lifetime_misses


def format_stats(snapshot: dict[str, Any]) -> str:
    """Render one health line. Split out so it can be tested directly."""
    done: list[int] = snapshot["done"]
    if done:
        vehicles = (
            f"{len(done)} vehicle{'s' if len(done) != 1 else ''} left frame, "
            f"avg {sum(done) / len(done):.1f} frames each "
            f"(min {min(done)}, max {max(done)})"
        )
    else:
        vehicles = "no vehicle completed"

    lookups = snapshot["hits"] + snapshot["misses"]
    hit_pct = 100.0 * snapshot["hits"] / lookups if lookups else 0.0
    return (
        f"pipeline {snapshot['elapsed_s']:.1f}s: "
        f"detect {snapshot['detect_fps']:.1f} fps, "
        f"raw capture {snapshot['raw_fps']:.1f} fps | "
        f"ring {snapshot['hits']} hit / {snapshot['misses']} fallback "
        f"({hit_pct:.1f}%) | "
        f"{vehicles}; {snapshot['in_frame']} still in frame"
    )


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
        raw_ring_size: int = _RAW_RING_SIZE,
        stats_interval_s: float = _STATS_INTERVAL_S,
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

        # --- full-resolution frame ring, keyed by buffer PTS ---------------
        self._ring_size = max(1, int(raw_ring_size))
        self._raw_frames: "OrderedDict[int, Any]" = OrderedDict()
        self._raw_lock = threading.Lock()
        self._raw_size: tuple[int, int] | None = None      # negotiated (w, h)
        self._detect_size: tuple[int, int] | None = None    # negotiated (w, h)
        # A detection's PTS should match its raw frame exactly; allow half a
        # frame period of slop in case an element re-stamps timestamps.
        self._pts_tolerance_ns = int(0.5 * 1_000_000_000 / max(1, config.frame_fps))
        self._raw_misses = 0  # 只用來節流 warning,統計數字在 _stats 裡

        # --- health reporting ----------------------------------------------
        self._stats = _PipelineStats(
            track_idle_timeout_s=config.track_idle_timeout_s
        )
        self._stats_interval_s = float(stats_interval_s)
        self._stats_stop = threading.Event()
        self._stats_thread: threading.Thread | None = None

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

        raw_sink = self._pipeline.get_by_name("speedtrap_raw")
        if raw_sink is None:
            raise RuntimeError("speedtrap_raw appsink missing from pipeline")
        raw_sink.connect("new-sample", self._on_raw_sample)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self._pipeline.set_state(Gst.State.PLAYING)
        self._loop = GLib.MainLoop()
        self._loop_thread = threading.Thread(
            target=self._loop.run, name="hailo-glib-loop", daemon=True
        )
        self._loop_thread.start()

        self._stats_stop.clear()
        self._stats_thread = threading.Thread(
            target=self._stats_loop, name="hailo-stats", daemon=True
        )
        self._stats_thread.start()
        self._running = True

    def stop(self) -> None:
        if not self._running:
            return
        self._running = False

        self._stats_stop.set()
        if self._stats_thread is not None:
            self._stats_thread.join(timeout=2.0)
            self._log_stats()

        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop is not None:
            self._loop.quit()
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=2.0)

        with self._raw_lock:
            self._raw_frames.clear()

        if self._dropped:
            _logger.warning(
                "dropped %d detections due to full queue during run", self._dropped
            )
        hits, misses = self._stats.lifetime_lookups()
        total = hits + misses
        if total:
            _logger.info(
                "full-resolution crops: %d/%d frames (%.1f%%), %d fell back to the "
                "downscaled detection frame",
                hits,
                total,
                100.0 * hits / total,
                misses,
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

    # --- health reporting -------------------------------------------------

    def _stats_loop(self) -> None:
        # Event.wait doubles as the sleep and the stop signal, so shutdown
        # doesn't have to wait out a full interval.
        while not self._stats_stop.wait(self._stats_interval_s):
            self._log_stats()

    def _log_stats(self) -> None:
        _logger.info("%s", format_stats(self._stats.collect(time.monotonic_ns())))

    # --- pipeline -------------------------------------------------------

    def _build_pipeline_str(self) -> str:
        cfg = self._config
        # Camera caps come from the station YAML — this is the resolution the
        # crops (and therefore the plate pixels) are taken from. format is
        # pinned (see _CAMERA_FORMAT): leaving it to negotiation makes the
        # pipeline's behaviour depend on the libcamera build and camera mode.
        src_caps = (
            f"video/x-raw,format={_CAMERA_FORMAT},"
            f"width={cfg.frame_width},height={cfg.frame_height},"
            f"framerate={cfg.frame_fps}/1"
        )
        # The detection branch is resized to whatever the compiled HEF expects.
        # That size is a property of the model, not of the camera, so it comes
        # from config (hailo_input_size) rather than being assumed anywhere.
        hef_size = cfg.hailo_input_size
        hailo_caps = (
            f"video/x-raw,format=RGB,width={hef_size},height={hef_size},"
            "pixel-aspect-ratio=1/1"
        )
        return (
            "libcamerasrc ! "
            f"{src_caps} ! "
            "tee name=rawtee ! "
            # === 偵測線:縮到 HEF 輸入尺寸,只用來找車 ===
            "queue leaky=downstream max-size-buffers=5 ! "
            "videoconvert n-threads=2 ! "
            # add-borders=false + 固定 PAR:純等比壓縮不加黑邊,
            # normalised bbox 才能線性映射回原始畫面
            "videoscale add-borders=false n-threads=2 ! "
            f"{hailo_caps} ! "
            f"hailonet hef-path={cfg.hef_path} batch-size=1 ! "
            f"hailofilter so-path={cfg.hailofilter_so_path} qos=false ! "
            f"hailotracker name=tracker keep-tracked-frames={cfg.tracker_keep_frames} "
            f"keep-new-frames={cfg.tracker_keep_frames} ! "
            "tee name=dettee ! "
            # === 線路 A：純辨識線 (送給 Python 取圖與 OCR) ===
            "queue leaky=downstream max-size-buffers=5 ! "
            "fakesink name=speedtrap_sink sync=false async=false "
            # === 線路 B：純直播線 (硬體畫框 + 推流，24小時不中斷) ===
            "dettee. ! "
            "queue leaky=downstream max-size-buffers=5 ! "
            "hailooverlay ! "  # 硬體級別的畫框工具
            "videoconvert ! "
            "x264enc tune=zerolatency bitrate=6000 speed-preset=superfast ! "
            "rtspclientsink location=rtsp://127.0.0.1:8554/live protocols=tcp "
            # === 線路 C：原始解析度取圖線 (裁切車輛區域用) ===
            # 來源已經是 RGB 時 videoconvert 是 passthrough(不複製、不耗
            # CPU),留著是為了改 _CAMERA_FORMAT 時這條線仍然成立。
            "rawtee. ! "
            "queue leaky=downstream max-size-buffers=3 ! "
            "videoconvert n-threads=2 ! "
            "video/x-raw,format=RGB ! "
            "appsink name=speedtrap_raw emit-signals=true sync=false "
            "max-buffers=3 drop=true"
        )

    # --- raw (full-resolution) frame ring --------------------------------

    def _on_raw_sample(self, sink: Any) -> Any:
        """appsink callback: copy the full-resolution frame into the ring."""
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        try:
            self._store_raw_sample(sample)
        except Exception as exc:  # noqa: BLE001 - never kill the streaming thread
            _logger.warning("failed to buffer raw frame: %s", exc)
        return Gst.FlowReturn.OK

    def _store_raw_sample(self, sample: Any) -> None:
        buffer = sample.get_buffer()
        if buffer is None:
            return
        pts = self._buffer_pts(buffer)
        if pts is None:
            return

        size = self._caps_size(sample.get_caps())
        if size is None:
            return
        if size != self._raw_size:
            self._raw_size = size
            _logger.info(
                "raw capture branch negotiated %dx%d RGB (config asked for %dx%d)",
                size[0],
                size[1],
                self._config.frame_width,
                self._config.frame_height,
            )

        frame = self._map_frame_rgb(buffer, size[0], size[1])
        if frame is None:
            return

        with self._raw_lock:
            self._raw_frames[pts] = frame
            while len(self._raw_frames) > self._ring_size:
                self._raw_frames.popitem(last=False)
        self._stats.note_raw_frame()

    def _lookup_raw_frame(self, pts: int | None) -> Any:
        """Return the full-resolution frame stamped ``pts``, or None."""
        if pts is None:
            return None
        with self._raw_lock:
            frame = self._raw_frames.get(pts)
            if frame is not None:
                return frame
            if not self._raw_frames:
                return None
            key, nearest = min(
                self._raw_frames.items(), key=lambda kv: abs(kv[0] - pts)
            )
        if abs(key - pts) <= self._pts_tolerance_ns:
            return nearest
        return None

    # --- detection --------------------------------------------------------

    def _on_buffer(self, pad: Any, info: Any, _user_data: Any) -> Any:
        buffer = info.get_buffer()
        if buffer is None:
            return Gst.PadProbeReturn.OK

        # Counted before any early return so the number is the branch's real
        # frame rate, not "frames that happened to contain a car".
        self._stats.note_detect_frame()

        roi = hailo.get_roi_from_buffer(buffer)
        detections = roi.get_objects_typed(hailo.HAILO_DETECTION)
        if not detections:
            return Gst.PadProbeReturn.OK

        # 所有下游數字都建立在這道過濾之上:Detection、每台車幀數統計、
        # 通行事件。行人與盆栽拿得到 hailotracker 的 track id,但到不了
        # 這裡以下的任何一行。(RTSP 上還是看得到它們的框,因為
        # hailooverlay 畫的是過濾前的全部偵測。)
        vehicles = [d for d in detections if d.get_label() in self._vehicle_classes]
        if not vehicles:
            return Gst.PadProbeReturn.OK

        # 兩種時鐘取自同一瞬間,必須相鄰兩行,中間不要插任何工作 ——
        # 它們是同一個時刻的兩種表示法,兩者要對得起來。
        frame_ns = time.monotonic_ns()
        capture_wall_ns = wall_clock_ns()
        pts = self._buffer_pts(buffer)

        # Prefer the full-resolution frame that carries the same PTS. Falling
        # back to the downscaled detection frame keeps the station running but
        # the plate will be far too small to OCR — hence the warning.
        frame_rgb = self._lookup_raw_frame(pts)
        self._stats.note_lookup(hit=frame_rgb is not None)
        if frame_rgb is None:
            self._raw_misses += 1
            frame_rgb = self._extract_detect_frame(pad, buffer)
            if self._raw_misses == 1 or self._raw_misses % 100 == 0:
                _logger.warning(
                    "no full-resolution frame for PTS %s (miss #%d) — cropping from "
                    "the downscaled detection frame instead; plates will be too "
                    "small to read",
                    pts,
                    self._raw_misses,
                )

        for det in vehicles:
            bbox = det.get_bbox()
            x1 = float(bbox.xmin())
            y1 = float(bbox.ymin())
            x2 = x1 + float(bbox.width())
            y2 = y1 + float(bbox.height())

            track_objs = det.get_objects_typed(hailo.HAILO_UNIQUE_ID)
            track_id = int(track_objs[0].get_id()) if track_objs else -1
            self._stats.note_track_frame(track_id, frame_ns)

            crop = None
            if frame_rgb is not None:
                crop = self._crop_and_encode_jpeg(frame_rgb, (x1, y1, x2, y2))

            try:
                self._queue.put_nowait(
                    Detection(
                        track_id=track_id,
                        label=det.get_label(),
                        bbox=(x1, y1, x2, y2),
                        confidence=float(det.get_confidence()),
                        frame_ns=frame_ns,
                        capture_wall_ns=capture_wall_ns,
                        frame_jpeg=crop.jpeg if crop else None,
                        sharpness=crop.sharpness if crop else 0.0,
                        crop_size=(crop.width, crop.height) if crop else None,
                    )
                )
            except queue.Full:
                self._dropped += 1

        return Gst.PadProbeReturn.OK

    def _extract_detect_frame(self, pad: Any, buffer: Any) -> Any:
        """Fallback path: map the detection-branch buffer (the downscaled
        frame that went into hailonet). Size is read from the negotiated caps
        rather than assumed, so a different HEF input size still works."""
        if self._detect_size is None:
            size = self._caps_size(pad.get_current_caps())
            if size is None:
                return None
            self._detect_size = size
            _logger.info(
                "detection branch negotiated %dx%d RGB", size[0], size[1]
            )
        return self._map_frame_rgb(buffer, *self._detect_size)

    def _crop_and_encode_jpeg(
        self,
        frame_rgb: Any,
        bbox: tuple[float, float, float, float],
    ) -> _CropResult | None:
        """Crop the bbox region out of an RGB frame and encode it as JPEG.

        Coordinates are normalised 0..1 relative to the detection frame, which
        maps linearly onto the full-resolution frame (see the module docstring
        for why the scaler must not add borders). Returns None for degenerate
        boxes or encoding failure."""
        x1, y1, x2, y2 = bbox
        h, w = frame_rgb.shape[:2]
        px1 = max(0, int((x1 - _CROP_MARGIN) * w))
        py1 = max(0, int((y1 - _CROP_MARGIN) * h))
        px2 = min(w, int((x2 + _CROP_MARGIN) * w))
        py2 = min(h, int((y2 + _CROP_MARGIN) * h))
        if px2 - px1 < _MIN_CROP_PX or py2 - py1 < _MIN_CROP_PX:
            return None

        crop_rgb = frame_rgb[py1:py2, px1:px2]
        _logger.debug(
            "vehicle crop %dx%d px taken from a %dx%d frame",
            px2 - px1,
            py2 - py1,
            w,
            h,
        )
        crop_bgr = _cv2.cvtColor(crop_rgb, _cv2.COLOR_RGB2BGR)
        sharpness = _crop_sharpness(crop_bgr)
        ok, jpeg = _cv2.imencode(
            ".jpg",
            crop_bgr,
            [int(_cv2.IMWRITE_JPEG_QUALITY), self._config.jpeg_quality],
        )
        if not ok:
            return None
        return _CropResult(
            jpeg=bytes(jpeg),
            width=px2 - px1,
            height=py2 - py1,
            sharpness=sharpness,
        )

    # --- gstreamer helpers ------------------------------------------------

    @staticmethod
    def _buffer_pts(buffer: Any) -> int | None:
        pts = buffer.pts
        if pts is None or pts == Gst.CLOCK_TIME_NONE:
            return None
        return int(pts)

    @staticmethod
    def _caps_size(caps: Any) -> tuple[int, int] | None:
        if caps is None or caps.get_size() == 0:
            return None
        structure = caps.get_structure(0)
        ok_w, width = structure.get_int("width")
        ok_h, height = structure.get_int("height")
        if not ok_w or not ok_h or width <= 0 or height <= 0:
            return None
        return int(width), int(height)

    @staticmethod
    def _map_frame_rgb(buffer: Any, width: int, height: int) -> Any:
        """Map a GStreamer buffer and return an owned (height, width, 3) uint8
        array. Rows may be padded to a stride wider than width*3, so the row
        stride is derived from the mapped size. Returns None on failure."""
        try:
            success, mapinfo = buffer.map(Gst.MapFlags.READ)
        except Exception:
            return None
        if not success:
            return None
        try:
            data = mapinfo.data
            row_bytes = width * 3
            if len(data) < row_bytes * height:
                # Some pipeline configurations strip the pixel payload before
                # it reaches the sink. Caller falls back to frame_jpeg=None.
                return None
            stride = len(data) // height
            flat = _np.frombuffer(data, dtype=_np.uint8, count=stride * height)
            return flat.reshape(height, stride)[:, :row_bytes].copy().reshape(
                height, width, 3
            )
        finally:
            buffer.unmap(mapinfo)

    def _on_bus_message(self, _bus: Any, message: Any) -> None:
        msg_type = message.type
        if msg_type == Gst.MessageType.EOS:
            _logger.info("gstreamer EOS received, stopping source")
            self._running = False
        elif msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            _logger.error("gstreamer error: %s (%s)", err, debug)
            self._running = False
