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

import logging
from dataclasses import dataclass
from typing import Any

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


@dataclass(frozen=True)
class PlateReading:
    text: str           # canonicalised, uppercase, no spaces/hyphens
    raw_text: str       # exactly what the OCR returned
    confidence: float
    is_taiwan_format: bool


class PlateRecognizer:
    """Wraps fast-plate-ocr. Single instance, thread-safe enough for our use
    (only called from the consumer loop on trigger, not per-frame)."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "PlateRecognizer dependencies missing: "
                f"{_IMPORT_ERROR}. Install with: pip install fast-plate-ocr opencv-python-headless"
            )
        # First instantiation triggers a ~10MB model download to ~/.cache/
        self._lpr = _LicensePlateRecognizer(model_name)
        self._model_name = model_name
        _logger.info("PlateRecognizer initialised with model=%s", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None:
        """Decode JPEG bytes -> RGB ndarray -> OCR. Returns None if no plate
        could be read (no detection or empty string)."""
        if not jpeg_bytes:
            return None

        nparr = _np.frombuffer(jpeg_bytes, _np.uint8)
        img_bgr = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if img_bgr is None:
            _logger.warning("failed to decode JPEG (%d bytes)", len(jpeg_bytes))
            return None

        img_rgb = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2RGB)
        return self._read_array(img_rgb)

    def _read_array(self, img_rgb: Any) -> PlateReading | None:
        # fast-plate-ocr API varies slightly across versions; handle both shapes
        try:
            results = self._lpr.run(img_rgb)
        except Exception as exc:
            _logger.warning("fast-plate-ocr failed: %s", exc)
            return None

        if not results:
            return None

        # Normalise result shape: list[tuple[str, float]] OR list[str] OR str
        first = results[0]
        if isinstance(first, tuple) and len(first) >= 2:
            raw_text = str(first[0])
            confidence = float(first[1])
        elif isinstance(first, str):
            raw_text = first
            confidence = 1.0  # fast-plate-ocr versions without confidence
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
