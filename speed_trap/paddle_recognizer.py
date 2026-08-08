"""PaddleOCR alternative backend for plate OCR.

PaddleOCR is a general-purpose OCR (not plate-specific) but historically has
strong support for Chinese / Asian character sets, so it often beats
fast-plate-ocr on Taiwan plates without any fine-tuning. The trade-off:

* Heavier install (~500 MB models + dependencies)
* Slower inference (~50-100ms on Pi 5 CPU vs ~10-30ms for fast-plate-ocr)
* Returns multiple text regions per image — we filter by Taiwan plate regex
  and pick the best match.

Same ``read_from_jpeg`` interface as PlateRecognizer so the consumer loop
doesn't care which backend is in use.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from typing import Any

from speed_trap import preprocess
from speed_trap.plate_format import is_valid_taiwan_plate, normalize_plate
from speed_trap.plate_recognizer import PlateReading

_logger = logging.getLogger(__name__)

_IMPORT_ERROR: str | None
_PaddleOCR: Any = None

try:
    from paddleocr import PaddleOCR as _PaddleOCR_real

    _PaddleOCR = _PaddleOCR_real
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on env
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


class PaddleOCRRecognizer:
    """Wraps PaddleOCR with our PlateRecognizer-compatible interface."""

    def __init__(
        self,
        lang: str = "en",
        *,
        preprocess_strategy: str = preprocess.DEFAULT,
        use_gpu: bool = False,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "PaddleOCRRecognizer dependencies missing: "
                f"{_IMPORT_ERROR}. Install with: "
                "pip install paddleocr paddlepaddle"
            )
        self._preprocess = preprocess.resolve(preprocess_strategy)
        # First instantiation downloads ~150-500MB models to ~/.paddleocr/
        # use_angle_cls=True helps with slightly rotated plates
        self._ocr = _PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=use_gpu)
        self._lang = lang
        _logger.info(
            "PaddleOCRRecognizer initialised with lang=%s use_gpu=%s preprocess=%s",
            lang, use_gpu, self._preprocess,
        )

    @property
    def model_name(self) -> str:
        return f"paddleocr-{self._lang}"

    def read_from_jpeg(self, jpeg_bytes: bytes) -> PlateReading | None:
        if not jpeg_bytes:
            return None

        payload = preprocess.apply_to_jpeg(self._preprocess, jpeg_bytes)

        fd, tmp_path = tempfile.mkstemp(suffix=".jpg", prefix="speedtrap_paddle_")
        try:
            os.write(fd, payload)
            os.close(fd)
            return self._read_path(tmp_path)
        finally:
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)

    def _read_path(self, image_path: str) -> PlateReading | None:
        try:
            raw = self._ocr.ocr(image_path, cls=True)
        except Exception as exc:
            _logger.warning("paddleocr failed: %s", exc)
            return None

        # PaddleOCR returns nested list:
        #   [ [ [bbox_pts, (text, confidence)], ... ] ]  # page > line > [bbox, (text, conf)]
        # We're looking for the line whose text best matches a Taiwan plate.
        candidates: list[tuple[str, float]] = []
        if raw and raw[0]:
            for line in raw[0]:
                try:
                    _bbox, (text, confidence) = line
                    candidates.append((str(text), float(confidence)))
                except (ValueError, TypeError, IndexError):
                    continue

        if not candidates:
            return None

        # Prefer a candidate that matches Taiwan plate format; else pick the
        # highest-confidence string overall.
        taiwan_matches = [
            c for c in candidates if is_valid_taiwan_plate(c[0])
        ]
        chosen = (
            max(taiwan_matches, key=lambda c: c[1])
            if taiwan_matches
            else max(candidates, key=lambda c: c[1])
        )

        raw_text, confidence = chosen
        canonical = normalize_plate(raw_text)
        if not canonical:
            return None
        return PlateReading(
            text=canonical,
            raw_text=raw_text,
            confidence=confidence,
            is_taiwan_format=is_valid_taiwan_plate(canonical),
        )
