"""Smoke tests for plate_recognizer.

Heavy parts (fast-plate-ocr) are optional dependencies — on PC dev we usually
skip the real OCR path and just exercise NoopPlateRecognizer + the import
guard. Pi-side end-to-end testing happens in the field manual, not pytest.
"""

from __future__ import annotations

import pytest

from speed_trap import plate_recognizer as pr
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
