"""CPU-side plate detection adapter (ultralytics YOLOv8n).

W3 v2 cascade — second stage. Hailo gives us a vehicle bbox crop;
this module runs the W2-trained ``plate_detector.pt`` on it via
ultralytics CPU runtime to find the license plate inside, then returns
the plate-region crop as JPEG bytes for fast-plate-ocr to OCR.

Why CPU rather than Hailo:
    Our W2 plate detector is a custom-trained YOLOv8n. When we tried to
    compile the .hef the libyolo_hailortpp_post.so dispatch expected
    Hailo Model Zoo tensor naming we couldn't replicate from
    ultralytics+DFC. ultralytics CPU on Pi 5 is roughly 30-80 ms per
    detect, which is fine for our event-triggered rate (one call per
    passage event, not per frame).

Defensive imports:
    ultralytics + torch are heavy and may not be installed in every dev
    env. Module loads fine without them — only class instantiation
    raises a clear RuntimeError naming the missing dep.
"""

from __future__ import annotations

import logging
from typing import Any

_logger = logging.getLogger(__name__)

_IMPORT_ERROR: str | None
_YOLO: Any = None
_cv2: Any = None
_np: Any = None

try:
    import cv2 as _cv2_real
    import numpy as _np_real
    from ultralytics import YOLO as _YOLO_real

    _cv2 = _cv2_real
    _np = _np_real
    _YOLO = _YOLO_real
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on env
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


_MIN_CROP_PIXELS = 8


class PlateDetectorCPU:
    """Runs the W2-trained plate detector (YOLOv8n .pt or .onnx) on CPU."""

    def __init__(
        self,
        model_path: str,
        *,
        confidence_threshold: float = 0.25,
    ) -> None:
        if _IMPORT_ERROR is not None:
            raise RuntimeError(
                "PlateDetectorCPU dependencies missing: "
                f"{_IMPORT_ERROR}. Install with: pip install ultralytics"
            )
        self._model = _YOLO(model_path)
        self._model_path = model_path
        self._threshold = float(confidence_threshold)
        _logger.info(
            "PlateDetectorCPU initialised with model=%s threshold=%.2f",
            model_path,
            self._threshold,
        )

    @property
    def model_path(self) -> str:
        return self._model_path

    def detect_best_plate_crop(self, jpeg_bytes: bytes) -> bytes | None:
        """Decode JPEG → run plate detector → encode best-bbox crop as JPEG.

        Returns None if any of: empty input, JPEG decode failure, no plate
        detected, best detection below threshold, degenerate crop region.

        We deliberately pick a single best detection (highest confidence
        above threshold) rather than emitting all candidates — the caller
        is OCR which wants one plate string per passage event.
        """
        if not jpeg_bytes:
            return None

        nparr = _np.frombuffer(jpeg_bytes, _np.uint8)
        img = _cv2.imdecode(nparr, _cv2.IMREAD_COLOR)
        if img is None:
            return None

        try:
            results = self._model(img, verbose=False)
        except Exception as exc:
            _logger.warning("plate detector inference failed: %s", exc)
            return None

        if not results or not len(results[0].boxes):
            return None

        boxes = results[0].boxes
        # Highest confidence above threshold
        best_idx = -1
        best_conf = -1.0
        for i in range(len(boxes)):
            conf = float(boxes.conf[i])
            if conf >= self._threshold and conf > best_conf:
                best_conf = conf
                best_idx = i

        if best_idx < 0:
            return None

        x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[best_idx].tolist())
        h, w = img.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        if x2 - x1 < _MIN_CROP_PIXELS or y2 - y1 < _MIN_CROP_PIXELS:
            return None

        crop = img[y1:y2, x1:x2]
        ok, jpeg = _cv2.imencode(
            ".jpg", crop, [int(_cv2.IMWRITE_JPEG_QUALITY), 95]
        )
        if not ok:
            return None
        return bytes(jpeg)
