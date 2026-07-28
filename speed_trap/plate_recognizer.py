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
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any

from speed_trap.config import StationConfig
from speed_trap.plate_format import is_valid_taiwan_plate, normalize_plate

_logger = logging.getLogger(__name__)

_IMPORT_ERROR: str | None
_LicensePlateRecognizer: Any = None
_cv2: Any = None
_np: Any = None

try:
    import cv2 as _cv2_real
    import numpy as _np_real
    from fast_plate_ocr import LicensePlateRecognizer as _LPR_real

    _cv2 = _cv2_real
    _np = _np_real
    _LicensePlateRecognizer = _LPR_real
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on env
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


DEFAULT_MODEL = "global-plates-mobile-vit-v2-model"

# Debug crops land on the SD card, not in /tmp — /tmp is tmpfs on Pi OS, so
# everything written there is gone after a reboot (and it eats RAM while the
# station runs). Overridable via the PlateRecognizer(debug_dir=...) argument.
DEFAULT_DEBUG_DIR = os.path.join(os.path.expanduser("~"), "speedtrap_debug")


def _jpeg_dimensions(jpeg_bytes: bytes) -> tuple[int, int] | None:
    """(width, height) of a JPEG in pixels, or None if it can't be decoded."""
    if _IMPORT_ERROR is not None or not jpeg_bytes:
        return None
    try:
        img = _cv2.imdecode(_np.frombuffer(jpeg_bytes, _np.uint8), _cv2.IMREAD_COLOR)
        if img is None:
            return None
        height, width = img.shape[:2]
        return int(width), int(height)
    except Exception:  # noqa: BLE001 - diagnostics must never break OCR
        return None


def _fmt_wh(wh: tuple[int, int] | None) -> str:
    return f"{wh[0]}x{wh[1]}" if wh else "?"


def _is_path_like(s: str) -> bool:
    """Heuristic: distinguish a fast-plate-ocr hub model name from a filesystem
    path. Paths contain separators or end in .onnx; hub names do not."""
    if not s:
        return False
    if "/" in s or "\\" in s:
        return True
    return s.lower().endswith(".onnx")


def _load_lpr(
    model_or_path: str, config_path: str | None
) -> Any:
    """Instantiate fast-plate-ocr's LicensePlateRecognizer.

    Two modes:
    1. Hub model: model_or_path is a name like "global-plates-mobile-vit-v2-model".
       Pass straight through; fast-plate-ocr downloads from its hub.
    2. Custom ONNX: model_or_path is a filesystem path to a .onnx file. Try
       the various parameter naming conventions fast-plate-ocr has used across
       versions (different keyword args in 0.4 vs 0.5 vs 0.7).
    """
    if not _is_path_like(model_or_path):
        return _LicensePlateRecognizer(model_or_path)

    onnx_path = model_or_path
    cfg_path = config_path

    # Versions of fast-plate-ocr have used different kwarg names for custom
    # ONNX loading. Try the most likely combinations.
    attempts: list[dict[str, str | None]] = [
        # Newer (0.5+): explicit onnx+config keywords
        {"onnx_model_path": onnx_path, "plate_config_path": cfg_path},
        {"model_path": onnx_path, "model_config_path": cfg_path},
        # Older shape: positional path + kw for config
        {"hub_ocr_model": onnx_path, "plate_config_path": cfg_path},
        # Last resort: pass path positionally (some versions accept it)
        {"model_name": onnx_path},
    ]

    last_err: Exception | None = None
    for kwargs in attempts:
        # strip Nones (config_path may be missing)
        clean = {k: v for k, v in kwargs.items() if v is not None}
        try:
            return _LicensePlateRecognizer(**clean)
        except (TypeError, ValueError) as exc:
            last_err = exc
            continue
        except Exception as exc:  # noqa: BLE001 - genuinely don't know what's raised
            last_err = exc
            continue

    raise RuntimeError(
        f"Could not load custom ONNX {onnx_path!r} via any known "
        f"fast-plate-ocr API. Last error: {last_err}"
    )


def _preprocess_for_ocr(jpeg_bytes: bytes) -> bytes:
    """Apply CLAHE (adaptive histogram equalization) + sharpening before OCR.

    Phase A experiment: many "OCR misreads" come from low-contrast or slightly
    blurry plate crops. CLAHE recovers contrast in shadowed plates; a small
    unsharp-mask helps the OCR see character edges more clearly.

    Returns a new JPEG with the same shape. If decoding fails (corrupt input),
    we return the original bytes unchanged so the caller can still attempt OCR.
    """
    if _IMPORT_ERROR is not None or not jpeg_bytes:
        return jpeg_bytes
    try:
        nparr = _np.frombuffer(jpeg_bytes, _np.uint8)
        img_bgr = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if img_bgr is None:
            return jpeg_bytes

        # CLAHE on the luminance channel only (preserves colour distribution)
        lab = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2LAB)
        l_channel, a, b = _cv2.split(lab)
        clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l_channel)
        lab_eq = _cv2.merge((l_clahe, a, b))
        img_eq = _cv2.cvtColor(lab_eq, _cv2.COLOR_LAB2BGR)

        # Light unsharp mask
        blur = _cv2.GaussianBlur(img_eq, (0, 0), sigmaX=1.0)
        img_sharp = _cv2.addWeighted(img_eq, 1.4, blur, -0.4, 0)

        ok, jpeg = _cv2.imencode(
            ".jpg", img_sharp, [int(_cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if ok:
            return bytes(jpeg)
        return jpeg_bytes
    except Exception as exc:  # pragma: no cover - opencv glitches
        _logger.warning("preprocessing failed: %s", exc)
        return jpeg_bytes


@dataclass(frozen=True)
class PlateReading:
    text: str           # canonicalised, uppercase, no spaces/hyphens
    raw_text: str       # exactly what the OCR returned
    confidence: float
    is_taiwan_format: bool


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
        preprocess: bool = False,
        save_debug_crops: bool = False,
        debug_dir: str | None = None,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "PlateRecognizer dependencies missing: "
                f"{_IMPORT_ERROR}. Install with: pip install fast-plate-ocr opencv-python-headless"
            )
        # _load_lpr decides between hub-name and custom-ONNX-path automatically.
        # For hub models this triggers a ~10MB download to ~/.cache/ on first use.
        self._lpr = _load_lpr(model_name, model_config)
        self._model_name = model_name
        self._model_config = model_config
        self._preprocess = preprocess
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
            preprocess,
            save_debug_crops,
            self._debug_dir,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def has_plate_detector(self) -> bool:
        return self._plate_detector is not None

    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None:
        """Run the configured cascade (optional plate detect → OCR).

        Returns None if any stage fails: empty input, plate detector
        finds nothing, OCR returns empty.

        fast-plate-ocr models expect grayscale input at a model-specific
        resolution (e.g. 70x140 for global-plates-mobile-vit-v2-model).
        Easiest path: write the JPEG to a temp file and let ``run(path)``
        do its own preprocessing — works across all fast-plate-ocr model
        variants without us hard-coding their input shape.
        """
        if not jpeg_bytes:
            return None

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
                return None
            self._maybe_save_debug("plate", plate_jpeg)
            plate_wh = _jpeg_dimensions(plate_jpeg)
            payload = plate_jpeg
        else:
            payload = jpeg_bytes

        # Optional contrast / sharpen preprocessing
        if self._preprocess:
            payload = _preprocess_for_ocr(payload)
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
            return self._read_path(tmp_path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

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

    def _read_array(self, image: Any) -> PlateReading | None:
        """Run OCR on a pre-decoded ndarray. Caller is responsible for the
        shape/format matching the loaded model. Use ``read_from_jpeg`` if
        you don't want to think about that."""
        try:
            # 取得圖片的 高 (h) 與 寬 (w)
            h, w = image.shape[:2]

            # 🌟 優化 1：面積守門員 (直接丟棄太小的背景雜訊)
            if w < 40 or h < 15:
                return None

            # --- 轉為灰階 ---
            if len(image.shape) == 3:
                image = _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)

            # 🌟 優化 2：畫質無損放大 (解決 OCR 鋸齒誤判，遵循手冊建議)
            if w < 96 or h < 24:
                new_w = max(w * 2, 96)
                new_h = max(h * 2, 24)
                image = _cv2.resize(image, (new_w, new_h), interpolation=_cv2.INTER_CUBIC)

            results = self._lpr.run(image)
        except Exception as exc:
            _logger.warning("fast-plate-ocr failed: %s", exc)
            return None
        return self._results_to_reading(results)

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
    """Drop-in replacement when fast-plate-ocr is unavailable.

    Used on PC dev environment where we don't want to install the OCR
    dependency just to run unit tests, and as a fallback if the runtime
    install on the Pi failed.
    """

    @property
    def model_name(self) -> str:
        return "noop"

    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None:
        return None


def make_recognizer(config: StationConfig) -> Any:
    """Build the OCR backend chosen by the station YAML.

    Falls back to NoopPlateRecognizer with a warning if the requested
    backend's dependencies aren't installed — keeps the station running
    even if a Pi is missing the optional OCR libraries, so vehicle
    detection still works and operators can see "no OCR" in the events.
    """
    backend = config.ocr_backend.lower()

    if backend == "noop":
        _logger.info("OCR backend: noop (disabled by config)")
        return NoopPlateRecognizer()

    if backend == "fast-plate-ocr":
        try:
            return PlateRecognizer(
                model_name=config.ocr_model_name,
                model_config=config.ocr_model_config,
                plate_detector_path=config.ocr_plate_detector_path,
                preprocess=config.ocr_preprocess,
                save_debug_crops=config.save_debug_crops,
            )
        except RuntimeError as exc:
            _logger.warning(
                "fast-plate-ocr unavailable, falling back to noop: %s", exc
            )
            return NoopPlateRecognizer()

    if backend == "paddleocr":
        # Lazy import — keeps PC dev / fast-plate-ocr-only deployments from
        # paying the cost of importing paddle at module load time.
        try:
            from speed_trap.paddle_recognizer import PaddleOCRRecognizer

            return PaddleOCRRecognizer(preprocess=config.ocr_preprocess)
        except RuntimeError as exc:
            _logger.warning(
                "paddleocr unavailable, falling back to noop: %s", exc
            )
            return NoopPlateRecognizer()

    # Should be unreachable thanks to config validation, but be defensive
    _logger.warning("unknown ocr_backend %r, using noop", backend)
    return NoopPlateRecognizer()
