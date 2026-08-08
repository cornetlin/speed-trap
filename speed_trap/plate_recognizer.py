"""License plate OCR adapter (CPU, fast-plate-ocr).

This is the "second stage" of our cascade pipeline: given a cropped vehicle
image (as JPEG bytes), find any license plate region inside it and read the
plate text. fast-plate-ocr handles both plate localization and OCR internally
so we don't need a separate Hailo plate-detection HEF — see W3 design notes.

The heavy imports (``fast_plate_ocr``, ``cv2``, ``numpy``) live inside a
try/except so that:

* PC dev environment can import this module without fast-plate-ocr installed
  (unit tests use a stub recognizer or skip).
* If fast-plate-ocr is missing on the Pi, instantiation raises a clear error
  but ``import speed_trap.plate_recognizer`` itself never crashes.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import inspect
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from speed_trap import preprocess
from speed_trap.config import StationConfig
from speed_trap.log_throttle import ThrottledWarning
from speed_trap.plate_format import is_valid_taiwan_plate, normalize_plate

_logger = logging.getLogger(__name__)

_IMPORT_ERROR: str | None
_LicensePlateRecognizer: Any = None
_cv2: Any = None
_np: Any = None

try:
    import cv2 as _cv2_real
    import numpy as _np_real
    import fast_plate_ocr as _fpo

    # 上游把 ONNXPlateRecognizer 改名成 LicensePlateRecognizer。兩個名字都試,
    # 順序是新的優先 —— 有些版本兩個都在(舊名是別名)。
    _LPR_real = getattr(_fpo, "LicensePlateRecognizer", None) or getattr(
        _fpo, "ONNXPlateRecognizer", None
    )
    if _LPR_real is None:
        raise ImportError(
            "fast_plate_ocr 裡找不到 LicensePlateRecognizer 或 "
            "ONNXPlateRecognizer,套件結構可能又變了"
        )

    _cv2 = _cv2_real
    _np = _np_real
    _LicensePlateRecognizer = _LPR_real
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on env
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class OcrBackendUnavailable(RuntimeError):
    """config 指定了某個 OCR 後端,但那個後端起不來。

    這是致命錯誤,不是警告 —— 靜默退回假後端的話,整場現場實驗會產出一份
    全空的 CSV,而且要跑完才發現。一次實驗的成本是一整個上午。
    """


def installed_version(package: str) -> str:
    """已安裝的套件版本,查不到就回 'unknown'。"""
    try:
        return importlib.metadata.version(package)
    except Exception:  # noqa: BLE001 - metadata 查詢不該影響主流程
        return "unknown"


DEFAULT_MODEL = "global-plates-mobile-vit-v2-model"

# Debug crops land on the SD card, not in /tmp — /tmp is tmpfs on Pi OS, so
# everything written there is gone after a reboot (and it eats RAM while the
# station runs). Overridable via the PlateRecognizer(debug_dir=...) argument.
DEFAULT_DEBUG_DIR = os.path.join(os.path.expanduser("~"), "speedtrap_debug")


# 尺寸算不出來時 CSV 的 vehicle_crop / plate_box 會變成 0,而那兩個數字正是
# 判斷「車牌讀不出來是不是因為裁切太小」的依據。靜默失敗會讓診斷資料看起來
# 像是「裁切尺寸為零」,而不是「量不到」。
_dimension_failures = ThrottledWarning(_logger)


def _jpeg_dimensions(jpeg_bytes: bytes) -> tuple[int, int] | None:
    """(width, height) of a JPEG in pixels, or None if it can't be decoded."""
    if _IMPORT_ERROR is not None or not jpeg_bytes:
        return None
    try:
        img = _cv2.imdecode(_np.frombuffer(jpeg_bytes, _np.uint8), _cv2.IMREAD_COLOR)
        if img is None:
            _dimension_failures.warn(
                "無法解碼 JPEG 取得尺寸(%d bytes),CSV 的裁切尺寸欄位會是 0",
                len(jpeg_bytes),
            )
            return None
        height, width = img.shape[:2]
        return int(width), int(height)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never break OCR
        _dimension_failures.warn(
            "取得裁切圖尺寸失敗,CSV 的裁切尺寸欄位會是 0:%s: %s",
            type(exc).__name__,
            exc,
        )
        return None


def _fmt_wh(wh: tuple[int, int] | None) -> str:
    return f"{wh[0]}x{wh[1]}" if wh else "?"


def _is_path_like(s: str) -> bool:
    """判斷 config 的 ocr_model_name 是「檔案路徑」還是「hub 模型名稱」。

    路徑含分隔符號或以 .onnx 結尾;hub 名稱(例如
    "global-plates-mobile-vit-v2-model")兩者皆無。
    """
    if not s:
        return False
    if "/" in s or "\\" in s:
        return True
    return s.lower().endswith(".onnx")


# fast-plate-ocr 每次改版都會換建構子的參數名稱,而且類別本身也改過名
# (ONNXPlateRecognizer -> LicensePlateRecognizer)。與其硬寫一串候選呼叫再
# 用 try/except 一個個試 —— 那樣既認不得沒見過的版本,失敗訊息也只剩最後
# 一個例外 —— 改成先用 inspect.signature 讀出「實際安裝的這一版」接受哪些
# 參數,再據此組裝呼叫。新版只要沿用其中一個名稱就會自動相容。
#
# 每一組是「我們要傳的東西」對應到各版本用過的參數名,依偏好排序。
_HUB_MODEL_PARAMS = ("hub_ocr_model", "model_name", "model")
_ONNX_PATH_PARAMS = ("onnx_model_path", "model_path", "onnx_path")
_CONFIG_PATH_PARAMS = (
    "plate_config_path",
    "config_file",
    "model_config_path",
    "plate_config",
)
_DEVICE_PARAMS = ("device",)

# Pi 沒有 CUDA。留著預設的 'auto' 會在每次建立 recognizer 時印兩行 GPU 偵測
# 失敗的警告,明確指定就不會去探。
_DEVICE = "cpu"


@dataclass(frozen=True)
class BackendInfo:
    """實際載入的 OCR 後端資訊。供 log 與 scripts/check_ocr.py 顯示。"""

    class_name: str
    version: str
    mode: str                 # "自訂 ONNX" 或 "hub 模型"
    kwargs: dict[str, str]

    def describe(self) -> str:
        arguments = ", ".join(f"{k}={v!r}" for k, v in self.kwargs.items())
        return f"{self.class_name}({arguments})"


def _accepted_params(target: Any) -> dict[str, inspect.Parameter]:
    """建構子接受的具名參數(排除 *args / **kwargs / self)。"""
    parameters = inspect.signature(target).parameters
    return {
        name: param
        for name, param in parameters.items()
        if name != "self"
        and param.kind
        not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    }


def _first_accepted(
    candidates: tuple[str, ...], accepted: dict[str, inspect.Parameter]
) -> str | None:
    for name in candidates:
        if name in accepted:
            return name
    return None


def _is_required(param: inspect.Parameter) -> bool:
    return param.default is inspect.Parameter.empty


def _require_file(path: str, label: str) -> None:
    """自己先檢查檔案在不在,才給得出「是哪個檔案不見了」。

    fast-plate-ocr 內部是 ``if onnx_model_path and plate_config_path`` 兩個
    一起檢查,不存在時只丟一句 "Missing model/config file!" —— 看不出是模型
    還是設定檔,現場排查很浪費時間。
    """
    target = Path(path).expanduser()
    if not target.exists():
        raise OcrBackendUnavailable(
            f"{label} 找不到:{target}\n"
            f"(fast-plate-ocr 內部只會回報 'Missing model/config file!',"
            f"分不出是哪一個檔案,所以這裡先自己檢查)"
        )
    if not target.is_file():
        raise OcrBackendUnavailable(f"{label} 不是檔案:{target}")


def _plan_load(
    target: Any, model_or_path: str, config_path: str | None
) -> BackendInfo:
    """讀出實際簽章與設定值,決定要傳哪些參數。

    自訂 ONNX 與 hub 模型是互斥的兩條路,絕不同時填:套件內部是
    ``if onnx_model_path and plate_config_path: ... elif hub_ocr_model: ...``,
    而且 hub_ocr_model 只接受預訓練清單裡的名稱 —— 把檔案路徑塞進去不會被
    當成路徑,只會變成一個不存在的模型名稱。
    """
    accepted = _accepted_params(target)
    available = ", ".join(accepted) or "(無具名參數)"
    version = installed_version("fast-plate-ocr")

    if _is_path_like(model_or_path):
        onnx_param = _first_accepted(_ONNX_PATH_PARAMS, accepted)
        if onnx_param is None:
            raise OcrBackendUnavailable(
                f"config 的 ocr_model_name 是檔案路徑({model_or_path}),但 "
                f"fast-plate-ocr {version} 的 {target.__name__} 建構子沒有任何"
                f"認得的 ONNX 路徑參數。實際接受的參數:{available}。"
                f"預期其中之一:{_ONNX_PATH_PARAMS}"
            )
        config_param = _first_accepted(_CONFIG_PATH_PARAMS, accepted)
        if config_param is None:
            raise OcrBackendUnavailable(
                f"自訂 ONNX 需要一併傳入 plate_config,但 fast-plate-ocr "
                f"{version} 的 {target.__name__} 建構子沒有對應參數。"
                f"實際接受的參數:{available}。預期其中之一:{_CONFIG_PATH_PARAMS}"
            )
        if not config_path:
            raise OcrBackendUnavailable(
                f"config 的 ocr_model_name 指向自訓 ONNX({model_or_path}),"
                f"但 ocr_model_config 是空的。自訓模型必須一併指定訓練輸出的 "
                f"plate_config.yaml"
            )

        # 兩個檔案都要存在,而且要能指出是哪一個不見了。
        _require_file(model_or_path, "ONNX 模型(ocr_model_name)")
        _require_file(config_path, "模型設定檔(ocr_model_config)")

        kwargs = {onnx_param: model_or_path, config_param: config_path}
        mode = "自訂 ONNX"

        # 這條路上 hub 參數一定留 None(不填即為預設)。若它竟是必填,
        # 表示這個版本不支援純自訂 ONNX,寧可講清楚也不要硬塞路徑進去。
        hub_param = _first_accepted(_HUB_MODEL_PARAMS, accepted)
        if hub_param is not None and _is_required(accepted[hub_param]):
            raise OcrBackendUnavailable(
                f"fast-plate-ocr {version} 的 {target.__name__} 把 {hub_param} "
                f"列為必填,無法只用自訂 ONNX 初始化。{hub_param} 只接受預訓練"
                f"模型名稱,填入檔案路徑不會被當成路徑。請改用支援 "
                f"{onnx_param} 的版本(pip install 'fast-plate-ocr<2')"
            )
    else:
        hub_param = _first_accepted(_HUB_MODEL_PARAMS, accepted)
        if hub_param is None:
            raise OcrBackendUnavailable(
                f"fast-plate-ocr {version} 的 {target.__name__} 建構子沒有任何"
                f"認得的 hub 模型參數。實際接受的參數:{available}。"
                f"預期其中之一:{_HUB_MODEL_PARAMS}"
            )
        kwargs = {hub_param: model_or_path}
        mode = "hub 模型"
        if config_path:
            _logger.warning(
                "ocr_model_name=%r 是 hub 模型名稱,ocr_model_config=%r 會被忽略"
                "(plate_config 只用於自訂 ONNX)",
                model_or_path,
                config_path,
            )

    # Pi 上一律 CPU,免得每次初始化都印 GPU 偵測失敗的警告。
    device_param = _first_accepted(_DEVICE_PARAMS, accepted)
    if device_param is not None:
        kwargs[device_param] = _DEVICE

    missing = [
        name
        for name, param in accepted.items()
        if _is_required(param) and name not in kwargs
    ]
    if missing:
        raise OcrBackendUnavailable(
            f"fast-plate-ocr {version} 的 {target.__name__} 還有我們填不了的"
            f"必填參數:{missing}。實際接受的參數:{available}"
        )

    return BackendInfo(
        class_name=target.__name__, version=version, mode=mode, kwargs=kwargs
    )


def _load_lpr(
    model_or_path: str, config_path: str | None
) -> tuple[Any, BackendInfo]:
    """依實際安裝版本的建構子簽章,實例化 fast-plate-ocr 的辨識器。

    兩種互斥的模式,由 ocr_model_name 的**內容**決定,不是由參數名稱決定:
    1. 自訂 ONNX:值是 .onnx 檔案路徑,搭配 ocr_model_config 的
       plate_config.yaml。兩個檔案都必須存在。
    2. Hub 模型:值是 "global-plates-mobile-vit-v2-model" 這類預訓練名稱,
       由 fast-plate-ocr 自行下載。

    參數名稱一律以 inspect.signature 讀到的為準,不猜。
    """
    target = _LicensePlateRecognizer
    info = _plan_load(target, model_or_path, config_path)

    try:
        instance = target(**info.kwargs)
    except Exception as exc:
        raise OcrBackendUnavailable(
            f"fast-plate-ocr {info.version} 的 {info.describe()} 呼叫失敗:"
            f"{type(exc).__name__}: {exc}"
        ) from exc

    _logger.info(
        "fast-plate-ocr %s:%s(%s)以 %s 初始化成功",
        info.version,
        info.class_name,
        info.mode,
        info.describe(),
    )
    return instance, info


@dataclass(frozen=True)
class PlateReading:
    text: str           # canonicalised, uppercase, no spaces/hyphens
    raw_text: str       # exactly what the OCR returned
    confidence: float
    is_taiwan_format: bool


@dataclass(frozen=True)
class OcrAttempt:
    """一次 OCR 的完整結果,含失敗時的尺寸資訊。

    read_from_jpeg 失敗時只回傳 None,診斷不到「是裁切圖太小、還是找不到
    車牌、還是 OCR 讀空」。CSV 需要這些數字,所以另外開這條路。
    """

    reading: PlateReading | None
    vehicle_wh: tuple[int, int] | None = None
    plate_wh: tuple[int, int] | None = None
    failure: str | None = None      # empty_input / no_plate_detected / ocr_empty


def run_ocr(recognizer: Any, jpeg_bytes: bytes | None) -> OcrAttempt:
    """對任何 recognizer 取得 OcrAttempt。

    只實作了 read_from_jpeg 的後端(例如 PaddleOCRRecognizer)也能用,
    只是拿不到尺寸欄位。
    """
    if not jpeg_bytes:
        return OcrAttempt(reading=None, failure="empty_input")
    detailed = getattr(recognizer, "read_attempt", None)
    if callable(detailed):
        return detailed(jpeg_bytes)
    reading = recognizer.read_from_jpeg(jpeg_bytes)
    return OcrAttempt(
        reading=reading, failure=None if reading else "ocr_empty"
    )


class PlateRecognizer:
    """Wraps fast-plate-ocr + optional CPU plate detector (W3 v2 cascade).

    Single instance, thread-safe enough for our use (only called from the
    consumer loop on trigger, not per-frame).

    When ``plate_detector_path`` is set, ``read_from_jpeg`` runs the W2
    plate detector on the input first to crop the plate region out of
    the vehicle bbox, then sends only that crop to fast-plate-ocr. This
    is the 3-stage cascade — Hailo vehicle → CPU plate → CPU OCR — which
    PC tests showed reaches 4/5 exact on standard Taiwan plates vs 0/25
    when we let fast-plate-ocr try to find the plate inside a whole-
    vehicle crop.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        model_config: str | None = None,
        plate_detector_path: str | None = None,
        preprocess_strategy: str = preprocess.DEFAULT,
        save_debug_crops: bool = False,
        debug_dir: str | None = None,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "PlateRecognizer dependencies missing: "
                f"{_IMPORT_ERROR}. Install with: pip install fast-plate-ocr opencv-python-headless"
            )
        # 認不得的策略名稱在建構時就炸,不要跑到第一台車才發現。
        self._preprocess = preprocess.resolve(preprocess_strategy)
        # _load_lpr decides between hub-name and custom-ONNX-path automatically.
        # For hub models this triggers a ~10MB download to ~/.cache/ on first use.
        self._lpr, self._backend_info = _load_lpr(model_name, model_config)
        self._model_name = model_name
        self._model_config = model_config
        self._save_debug_crops = save_debug_crops
        self._debug_dir = debug_dir or DEFAULT_DEBUG_DIR
        self._debug_seq = 0

        # Stage 2 of the cascade — lazy-loaded so missing ultralytics on
        # PC dev doesn't break import of this module.
        self._plate_detector: Any = None
        if plate_detector_path:
            from speed_trap.plate_detector_cpu import PlateDetectorCPU

            self._plate_detector = PlateDetectorCPU(plate_detector_path)

        if self._save_debug_crops:
            os.makedirs(self._debug_dir, exist_ok=True)

        _logger.info(
            "PlateRecognizer initialised with model=%s config=%s plate_detector=%s "
            "preprocess=%s save_debug_crops=%s debug_dir=%s",
            model_name,
            model_config,
            plate_detector_path or "(none — direct OCR)",
            self._preprocess,
            save_debug_crops,
            self._debug_dir,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def preprocess_strategy(self) -> str:
        """實際套用的前處理策略名稱(已對過別名)。"""
        return self._preprocess

    @property
    def backend_info(self) -> BackendInfo:
        """實際載入的類別、版本與參數組合。scripts/check_ocr.py 會印出來。"""
        return self._backend_info

    @property
    def has_plate_detector(self) -> bool:
        return self._plate_detector is not None

    def read_plate_only(self, jpeg_bytes: bytes) -> OcrAttempt:
        """跳過車牌偵測那一段,把輸入直接餵給 OCR。

        給 scripts/check_ocr.py 用:既有的除錯圖有兩種,
        ``*_vehicle.jpg`` 是整輛車、``*_plate.jpg`` 已經是切好的車牌。
        後者再送一次車牌偵測往往找不到東西,那是輸入型態的問題,不是 OCR
        讀不出來。
        """
        if not jpeg_bytes:
            return OcrAttempt(reading=None, failure="empty_input")
        payload = preprocess.apply_to_jpeg(self._preprocess, jpeg_bytes)
        size = _jpeg_dimensions(jpeg_bytes)
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="speedtrap_plate_")
        try:
            os.write(fd, payload)
            os.close(fd)
            reading = self._read_path(tmp_path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
        return OcrAttempt(
            reading=reading,
            vehicle_wh=size,
            plate_wh=size,
            failure=None if reading else "ocr_empty",
        )

    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None:
        return self.read_attempt(jpeg_bytes).reading

    def read_attempt(self, jpeg_bytes: bytes) -> OcrAttempt:
        """Run the configured cascade (optional plate detect → OCR).

        ``reading`` is None if any stage fails: empty input, plate detector
        finds nothing, OCR returns empty — ``failure`` says which, and the
        crop sizes are reported either way so a failure can be diagnosed.

        fast-plate-ocr models expect grayscale input at a model-specific
        resolution (e.g. 70x140 for global-plates-mobile-vit-v2-model).
        Easiest path: write the JPEG to a temp file and let ``run(path)``
        do its own preprocessing — works across all fast-plate-ocr model
        variants without us hard-coding their input shape.
        """
        if not jpeg_bytes:
            return OcrAttempt(reading=None, failure="empty_input")

        # Debug: dump what we got from the caller (e.g. Hailo vehicle crop)
        self._maybe_save_debug("vehicle", jpeg_bytes)

        vehicle_wh = _jpeg_dimensions(jpeg_bytes)

        # Stage 2 — plate detection on the vehicle crop
        plate_wh: tuple[int, int] | None = None
        if self._plate_detector is not None:
            plate_jpeg = self._plate_detector.detect_best_plate_crop(jpeg_bytes)
            if plate_jpeg is None:
                # Logged at INFO, not DEBUG: these are the passages that end up
                # without a plate string, and the crop size is the first thing
                # to look at when diagnosing them.
                _logger.info(
                    "no plate found: vehicle_crop=%s px, plate_box=- px",
                    _fmt_wh(vehicle_wh),
                )
                return OcrAttempt(
                    reading=None,
                    vehicle_wh=vehicle_wh,
                    failure="no_plate_detected",
                )
            self._maybe_save_debug("plate", plate_jpeg)
            plate_wh = _jpeg_dimensions(plate_jpeg)
            payload = plate_jpeg
        else:
            payload = jpeg_bytes

        # 前處理接在車牌偵測之後、OCR 之前 —— 離線掃描量的就是這個位置
        # (對已經切好的車牌圖做前處理再送 OCR),換了位置掃描的結論就不適用。
        if self._preprocess != preprocess.BASELINE:
            payload = preprocess.apply_to_jpeg(self._preprocess, payload)
            self._maybe_save_debug("preproc", payload)

        # The two numbers that say whether the crop path is healthy: how big
        # the incoming vehicle crop actually is, and how many pixels wide the
        # plate box inside it turned out to be. A plate under ~100 px wide is
        # where fast-plate-ocr starts guessing.
        _logger.info(
            "OCR input: vehicle_crop=%s px, plate_box=%s px, plate_width=%s px",
            _fmt_wh(vehicle_wh),
            _fmt_wh(plate_wh),
            plate_wh[0] if plate_wh else "?",
        )

        # Stage 3 — OCR. tempfile path on Pi is /tmp which is tmpfs (RAM)
        # on most distros, so the round-trip is basically a memcpy + ONNX
        # inference.
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="speedtrap_plate_")
        try:
            os.write(fd, payload)
            os.close(fd)
            reading = self._read_path(tmp_path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

        return OcrAttempt(
            reading=reading,
            vehicle_wh=vehicle_wh,
            plate_wh=plate_wh,
            failure=None if reading else "ocr_empty",
        )

    def _maybe_save_debug(self, kind: str, jpeg_bytes: bytes) -> None:
        if not self._save_debug_crops or not jpeg_bytes:
            return
        self._debug_seq += 1
        path = os.path.join(
            self._debug_dir, f"{self._debug_seq:05d}_{kind}.jpg"
        )
        try:
            with open(path, "wb") as f:
                f.write(jpeg_bytes)
        except OSError as exc:
            _logger.warning("save_debug_crops: failed to write %s: %s", path, exc)

    def _read_path(self, image_path: str) -> PlateReading | None:
        try:
            results = self._lpr.run(image_path)
        except Exception as exc:
            _logger.warning("fast-plate-ocr failed: %s", exc)
            return None

        return self._results_to_reading(results)

    # 這裡原本還有一個 _read_array:接 ndarray、小於 40x15 就放棄、小於 96x24
    # 就用 CUBIC 放大。它從來沒被呼叫過(read_from_jpeg 走的是 _read_path)。
    # 那段 CUBIC 放大現在真的接上線了 —— 它就是 speed_trap.preprocess 的
    # cubic_* 系列,而且離線掃描證明配上 unsharp 之後有效(23/46 → 26/46)。
    # 最小尺寸的門檻沒有跟著接回來:讀不出來的小圖也要留一行 CSV 才查得到
    # 原因,而 plate_box_w 欄位本來就讓事後篩選做得到同一件事。

    @staticmethod
    def _results_to_reading(results: Any) -> PlateReading | None:
        if not results:
            return None

        # Normalise result shape across fast-plate-ocr versions:
        #   - newer (>= 0.5): list[PlatePrediction(plate=str, char_probs=list|None,
        #                                          region=..., region_prob=...)]
        #   - older: list[tuple[str, float]]
        #   - oldest: list[str]
        first = results[0] if isinstance(results, list) else results

        if hasattr(first, "plate"):
            raw_text = str(first.plate)
            # Try a few ways to extract confidence; fall back to 1.0 if the
            # model doesn't surface character or region probabilities.
            char_probs = getattr(first, "char_probs", None)
            region_prob = getattr(first, "region_prob", None)
            if char_probs:
                try:
                    confidence = float(sum(char_probs) / len(char_probs))
                except (TypeError, ZeroDivisionError):
                    confidence = 1.0
            elif region_prob is not None:
                try:
                    confidence = float(region_prob)
                except (TypeError, ValueError):
                    confidence = 1.0
            else:
                confidence = 1.0
        elif isinstance(first, tuple) and len(first) >= 2:
            raw_text = str(first[0])
            confidence = float(first[1])
        elif isinstance(first, str):
            raw_text = first
            confidence = 1.0
        else:
            _logger.warning("unexpected fast-plate-ocr result shape: %r", results)
            return None

        if not raw_text:
            return None

        canonical = normalize_plate(raw_text)
        return PlateReading(
            text=canonical,
            raw_text=raw_text,
            confidence=confidence,
            is_taiwan_format=is_valid_taiwan_plate(canonical),
        )


class NoopPlateRecognizer:
    """只在 config 明確寫 ``ocr_backend: noop`` 時使用的假後端。

    用途:PC 開發環境不想為了跑單元測試裝 OCR 相依,或現場只想驗偵測與
    追蹤、不需要車牌。

    這個類別**不再**被當成初始化失敗的退路 —— 那會讓整場實驗安靜地產出
    一份沒有車牌的 CSV。失敗現在直接丟 :class:`OcrBackendUnavailable`。
    """

    @property
    def model_name(self) -> str:
        return "noop"

    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None:
        return None

    def read_attempt(self, jpeg_bytes: bytes) -> OcrAttempt:
        return OcrAttempt(reading=None, failure="noop")


def make_recognizer(config: StationConfig) -> Any:
    """Build the OCR backend chosen by the station YAML.

    初始化失敗一律丟 :class:`OcrBackendUnavailable` 中止程式,**不會**退回
    假後端。原本的作法是記一則 warning 就繼續跑,結果是整場現場實驗產出一份
    完全沒有車牌的 CSV,而且要等實驗跑完才發現 —— 一次實驗的成本是一整個
    上午。要在沒有 OCR 的情況下跑(例如只驗偵測與追蹤),請在 YAML 明確寫
    ``ocr_backend: noop``。
    """
    backend = config.ocr_backend.lower()

    if backend == "noop":
        _logger.info("OCR backend: noop(config 明確關閉,不做車牌辨識)")
        return NoopPlateRecognizer()

    if backend == "fast-plate-ocr":
        version = installed_version("fast-plate-ocr")
        try:
            return PlateRecognizer(
                model_name=config.ocr_model_name,
                model_config=config.ocr_model_config,
                plate_detector_path=config.ocr_plate_detector_path,
                preprocess_strategy=config.ocr_preprocess,
                save_debug_crops=config.save_debug_crops,
            )
        except Exception as exc:
            raise OcrBackendUnavailable(
                f"OCR 後端 'fast-plate-ocr' 初始化失敗,站台中止。\n"
                f"  已安裝版本: fast-plate-ocr {version}\n"
                f"  模型       : {config.ocr_model_name}\n"
                f"  模型設定檔 : {config.ocr_model_config}\n"
                f"  錯誤       : {type(exc).__name__}: {exc}\n"
                f"若要在沒有車牌辨識的情況下跑,請在 YAML 明確設定 "
                f"ocr_backend: noop"
            ) from exc

    if backend == "paddleocr":
        # Lazy import — keeps PC dev / fast-plate-ocr-only deployments from
        # paying the cost of importing paddle at module load time.
        version = installed_version("paddleocr")
        try:
            from speed_trap.paddle_recognizer import PaddleOCRRecognizer

            return PaddleOCRRecognizer(preprocess_strategy=config.ocr_preprocess)
        except Exception as exc:
            raise OcrBackendUnavailable(
                f"OCR 後端 'paddleocr' 初始化失敗,站台中止。\n"
                f"  已安裝版本: paddleocr {version}\n"
                f"  錯誤       : {type(exc).__name__}: {exc}\n"
                f"安裝方式 pip install -e \".[paddle]\";若要在沒有車牌辨識的"
                f"情況下跑,請在 YAML 明確設定 ocr_backend: noop"
            ) from exc

    # config 驗證應該已經擋掉,但萬一有人繞過驗證,同樣不默默降級。
    raise OcrBackendUnavailable(
        f"不認得的 ocr_backend {backend!r}。可用值:fast-plate-ocr、"
        f"paddleocr、noop"
    )
