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
    """Wraps fast-plate-ocr. Single instance, thread-safe enough for our use
    (only called from the consumer loop on trigger, not per-frame)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        preprocess: bool = False,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "PlateRecognizer dependencies missing: "
                f"{_IMPORT_ERROR}. Install with: pip install fast-plate-ocr opencv-python-headless"
            )
        # First instantiation triggers a ~10MB model download to ~/.cache/
        self._lpr = _LicensePlateRecognizer(model_name)
        self._model_name = model_name
        self._preprocess = preprocess
        _logger.info(
            "PlateRecognizer initialised with model=%s preprocess=%s",
            model_name, preprocess,
        )

    @property
    def model_name(self) -> str:
        return self._model_name

    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None:
        """Run OCR on JPEG bytes. Returns None if no plate could be read.

        fast-plate-ocr models expect grayscale input at a model-specific
        resolution (e.g. 70x140 for global-plates-mobile-vit-v2-model).
        Passing a numpy RGB array fails with
            Got invalid dimensions: index 3 Got: 3 Expected: 1
        Easiest path: write the JPEG to a temp file and let ``run(path)``
        do its own preprocessing — works across all fast-plate-ocr model
        variants without us hard-coding their input shape.
        """
        if not jpeg_bytes:
            return None

        payload = _preprocess_for_ocr(jpeg_bytes) if self._preprocess else jpeg_bytes

        # tempfile path on Pi is /tmp which is tmpfs (RAM) on most distros,
        # so the round-trip is basically a memcpy + ONNX inference.
        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="speedtrap_plate_")
        try:
            os.write(fd, payload)
            os.close(fd)
            return self._read_path(tmp_path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

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
                preprocess=config.ocr_preprocess,
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
