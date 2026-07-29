"""Smoke tests for plate_detector_cpu.

The actual model inference needs ultralytics + a trained .pt file, so we
only verify the module loads cleanly and the dependency guard raises a
clear error.
"""

from __future__ import annotations

import pytest

from speed_trap import plate_detector_cpu as pdc


def test_module_imports_on_pc() -> None:
    """Module must import even without ultralytics installed."""
    assert hasattr(pdc, "PlateDetectorCPU")


def test_instantiation_raises_when_deps_missing() -> None:
    if pdc._IMPORT_ERROR is None:
        pytest.skip(
            "ultralytics is installed; can't exercise the missing-deps path"
        )
    with pytest.raises(RuntimeError, match="ultralytics"):
        pdc.PlateDetectorCPU("/some/model.pt")


def test_error_message_mentions_install_command() -> None:
    if pdc._IMPORT_ERROR is None:
        pytest.skip("ultralytics is installed; can't test the error message")
    with pytest.raises(RuntimeError) as exc_info:
        pdc.PlateDetectorCPU("/some/model.pt")
    assert "pip install ultralytics" in str(exc_info.value)
