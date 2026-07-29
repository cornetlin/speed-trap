from __future__ import annotations

import numpy as np
import pytest

from speed_trap.mock_detector import MockDetector


def _frame() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_first_frame_emits_one_detection() -> None:
    detector = MockDetector()
    dets = detector.detect(_frame(), frame_index=0, frame_ns=1)

    assert len(dets) == 1
    det = dets[0]
    assert det.track_id == 1
    assert det.label == "car"
    assert det.frame_ns == 1
    assert det.frame_jpeg is None


def test_track_id_stable_across_frames() -> None:
    detector = MockDetector(track_id=42)
    ids = {detector.detect(_frame(), i, i)[0].track_id for i in range(10)}
    assert ids == {42}


def test_y_increases_each_frame() -> None:
    detector = MockDetector()
    y_centers = []
    for i in range(20):
        det = detector.detect(_frame(), frame_index=i, frame_ns=i)[0]
        y1, y2 = det.bbox[1], det.bbox[3]
        y_centers.append((y1 + y2) / 2.0)
    assert y_centers == sorted(y_centers)
    assert y_centers[-1] > y_centers[0]


def test_emits_nothing_after_vehicle_exits() -> None:
    detector = MockDetector(start_y_center=0.95, step_per_frame=0.5, bbox_h=0.1)
    assert detector.detect(_frame(), 0, 0) != []
    assert detector.detect(_frame(), 100, 100) == []


def test_bbox_clamped_to_unit_square() -> None:
    detector = MockDetector(start_y_center=0.0, step_per_frame=0.01, bbox_h=0.4)
    det = detector.detect(_frame(), frame_index=0, frame_ns=0)[0]
    x1, y1, x2, y2 = det.bbox
    assert 0.0 <= x1 < x2 <= 1.0
    assert 0.0 <= y1 < y2 <= 1.0


def test_invalid_step_rejected() -> None:
    with pytest.raises(ValueError, match="step_per_frame"):
        MockDetector(step_per_frame=0.0)


def test_invalid_bbox_size_rejected() -> None:
    with pytest.raises(ValueError, match="bbox_w"):
        MockDetector(bbox_w=0.0)
    with pytest.raises(ValueError, match="bbox_w"):
        MockDetector(bbox_h=1.5)
