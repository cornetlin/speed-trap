"""Smoke tests for plate_recognizer.

Heavy parts (fast-plate-ocr) are optional dependencies — on PC dev we usually
skip the real OCR path and just exercise NoopPlateRecognizer + the import
guard. Pi-side end-to-end testing happens in the field manual, not pytest.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from speed_trap import plate_recognizer as pr
from speed_trap.config import StationConfig
from speed_trap.plate_recognizer import NoopPlateRecognizer, PlateReading


def test_module_imports_on_pc() -> None:
    """plate_recognizer must import cleanly even without fast-plate-ocr."""
    assert hasattr(pr, "PlateRecognizer")
    assert hasattr(pr, "NoopPlateRecognizer")
    assert hasattr(pr, "PlateReading")


def test_real_recognizer_raises_clearly_when_deps_missing() -> None:
    """If fast-plate-ocr isn't installed, instantiation must error with a
    message that tells you how to fix it."""
    if pr._IMPORT_ERROR is None:
        pytest.skip("fast-plate-ocr is installed; can't test the missing-deps path")

    with pytest.raises(RuntimeError, match="fast-plate-ocr"):
        pr.PlateRecognizer()


def test_noop_recognizer_returns_none() -> None:
    """The Noop variant is what we use on PC dev / when OCR deps unavailable."""
    noop = NoopPlateRecognizer()
    assert noop.read_from_jpeg(b"") is None
    assert noop.read_from_jpeg(b"fake-jpeg-bytes") is None
    assert noop.model_name == "noop"


def test_plate_reading_is_frozen_dataclass() -> None:
    r = PlateReading(
        text="ABC1234",
        raw_text="ABC-1234",
        confidence=0.92,
        is_taiwan_format=True,
    )
    assert r.text == "ABC1234"
    assert r.is_taiwan_format is True
    # frozen dataclass — assignment raises FrozenInstanceError (a dataclasses
    # subclass of AttributeError)
    with pytest.raises(AttributeError):
        r.text = "XYZ1234"  # type: ignore[misc]


def test_noop_recognizer_handles_empty_bytes() -> None:
    noop = NoopPlateRecognizer()
    assert noop.read_from_jpeg(b"") is None


# ─── _results_to_reading parsing (cross-version shape handling) ─────────


class _FakePrediction:
    """Mimics fast-plate-ocr >=0.5 PlatePrediction dataclass."""

    def __init__(
        self,
        plate: str,
        char_probs: list[float] | None = None,
        region_prob: float | None = None,
    ) -> None:
        self.plate = plate
        self.char_probs = char_probs
        self.region = None
        self.region_prob = region_prob


def test_results_to_reading_newer_plate_prediction_no_probs() -> None:
    """fast-plate-ocr >=0.5 returns PlatePrediction with attributes."""
    results = [_FakePrediction(plate="ABC1234")]
    reading = pr.PlateRecognizer._results_to_reading(results)
    assert reading is not None
    assert reading.text == "ABC1234"
    assert reading.raw_text == "ABC1234"
    assert reading.confidence == 1.0  # no char_probs, fallback
    assert reading.is_taiwan_format is True


def test_results_to_reading_newer_with_char_probs() -> None:
    """Mean of per-character probs becomes overall confidence."""
    results = [_FakePrediction(plate="ABC1234", char_probs=[0.9, 0.8, 0.7])]
    reading = pr.PlateRecognizer._results_to_reading(results)
    assert reading is not None
    assert reading.text == "ABC1234"
    assert reading.confidence == pytest.approx(0.8)  # (0.9+0.8+0.7)/3


def test_results_to_reading_newer_with_region_prob_only() -> None:
    results = [_FakePrediction(plate="ABC1234", region_prob=0.92)]
    reading = pr.PlateRecognizer._results_to_reading(results)
    assert reading is not None
    assert reading.confidence == pytest.approx(0.92)


def test_results_to_reading_tuple_shape() -> None:
    """Older fast-plate-ocr returns list[tuple[str, float]]."""
    results = [("ABC1234", 0.85)]
    reading = pr.PlateRecognizer._results_to_reading(results)
    assert reading is not None
    assert reading.text == "ABC1234"
    assert reading.confidence == pytest.approx(0.85)


def test_results_to_reading_plain_string() -> None:
    """Oldest fast-plate-ocr returns list[str]."""
    results = ["ABC1234"]
    reading = pr.PlateRecognizer._results_to_reading(results)
    assert reading is not None
    assert reading.text == "ABC1234"
    assert reading.confidence == 1.0


def test_results_to_reading_empty() -> None:
    assert pr.PlateRecognizer._results_to_reading([]) is None
    assert pr.PlateRecognizer._results_to_reading(None) is None


def test_results_to_reading_non_taiwan_plate_still_returned() -> None:
    """We surface OCR output even if it doesn't match Taiwan format —
    the consumer can decide what to do with is_taiwan_format=False."""
    results = [_FakePrediction(plate="A46512")]  # OCR mis-read from the field test
    reading = pr.PlateRecognizer._results_to_reading(results)
    assert reading is not None
    assert reading.text == "A46512"
    assert reading.is_taiwan_format is False  # 1 letter + 5 digits = not Taiwan


def test_results_to_reading_empty_plate_string_returns_none() -> None:
    results = [_FakePrediction(plate="")]
    assert pr.PlateRecognizer._results_to_reading(results) is None


def test_results_to_reading_unknown_shape_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging as _log
    with caplog.at_level(_log.WARNING):
        assert pr.PlateRecognizer._results_to_reading([{"unexpected": "dict"}]) is None
    assert any("unexpected" in r.message for r in caplog.records)


# ─── make_recognizer factory (Phase A) ─────────────────────────────────


def _config(
    *,
    ocr_backend: str = "fast-plate-ocr",
    ocr_model_name: str = "global-plates-mobile-vit-v2-model",
    ocr_preprocess: bool = False,
) -> StationConfig:
    return StationConfig(
        station_id="x",
        camera_source="cam",
        frame_width=640,
        frame_height=480,
        frame_fps=30,
        hef_path=Path("m.hef"),
        hailofilter_so_path=Path("/tmp/dummy.so"),
        vehicle_classes=("car",),
        trigger_line_y=0.5,
        mqtt_broker=None,
        mqtt_topic="t",
        log_level="INFO",
        ocr_backend=ocr_backend,
        ocr_model_name=ocr_model_name,
        ocr_preprocess=ocr_preprocess,
    )


def test_make_recognizer_noop_backend() -> None:
    rec = pr.make_recognizer(_config(ocr_backend="noop"))
    assert isinstance(rec, NoopPlateRecognizer)


def test_make_recognizer_fast_plate_ocr_falls_back_when_missing() -> None:
    """If fast-plate-ocr isn't installed on the dev machine, factory
    falls back to NoopPlateRecognizer (with a warning) instead of crashing."""
    if pr._IMPORT_ERROR is None:
        pytest.skip("fast-plate-ocr is installed; can't test the missing-deps path")
    rec = pr.make_recognizer(_config(ocr_backend="fast-plate-ocr"))
    assert isinstance(rec, NoopPlateRecognizer)


def test_make_recognizer_paddleocr_falls_back_when_missing() -> None:
    """Same fallback story for paddleocr backend."""
    try:
        from speed_trap.paddle_recognizer import _IMPORT_ERROR as _paddle_err
    except ImportError:
        pytest.skip("paddle_recognizer module not importable on this machine")
    if _paddle_err is None:
        pytest.skip("paddleocr is installed; can't test the missing-deps path")
    rec = pr.make_recognizer(_config(ocr_backend="paddleocr"))
    assert isinstance(rec, NoopPlateRecognizer)


def test_make_recognizer_unknown_backend_returns_noop(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Defensive: config validation should catch unknown backends, but if a
    test somehow bypasses validation the factory still returns Noop."""
    # We can't construct StationConfig with an invalid backend (it raises),
    # so use a SimpleNamespace stand-in.
    from types import SimpleNamespace

    fake_config = SimpleNamespace(
        ocr_backend="surprise",
        ocr_model_name="x",
        ocr_preprocess=False,
    )
    rec = pr.make_recognizer(fake_config)  # type: ignore[arg-type]
    assert isinstance(rec, NoopPlateRecognizer)


def test_preprocess_helper_handles_empty_bytes() -> None:
    """Phase A preprocessing should be a no-op for empty input."""
    assert pr._preprocess_for_ocr(b"") == b""


def test_preprocess_helper_handles_invalid_jpeg() -> None:
    """Corrupt input should be returned unchanged, not crash."""
    out = pr._preprocess_for_ocr(b"not a real jpeg")
    assert out == b"not a real jpeg"
