"""最佳幀挑選:邊界排除、清晰度評分、無候選時的保護機制。

背景:實測 15 分鐘 24 台車只讀出 13 個車牌,除錯圖顯示挑到的最佳幀常是
車頭已經超出畫面右緣的那一瞬間 —— 車牌被裁掉一半。原因是面積權重最重,
而車子快出框時投影面積剛好最大。
"""

from __future__ import annotations

import pytest
from tests.helpers import NS_PER_SEC, approaching_car, make_detection

from speed_trap.tracker import (
    ScoringParams,
    VehicleTracker,
    edge_distance,
    normalize_sharpness,
    score_detection,
    touches_edge,
)


def test_edge_distance_measures_nearest_border() -> None:
    assert edge_distance((0.4, 0.4, 0.6, 0.6)) == pytest.approx(0.4)
    # 只要有任何一邊靠近,回傳值就小 —— 取四邊最小
    assert edge_distance((0.7, 0.3, 0.995, 0.9)) == pytest.approx(0.005)
    # 已經超出畫面時為負
    assert edge_distance((0.7, 0.3, 1.05, 0.9)) < 0.0


def test_touches_edge_uses_margin() -> None:
    bbox = (0.7, 0.3, 0.985, 0.9)     # 距右緣 0.015
    assert touches_edge(bbox, 0.02)
    assert not touches_edge(bbox, 0.01)


def test_frame_touching_edge_is_excluded_even_when_largest() -> None:
    """這是這次要修的核心行為:面積最大但貼邊的幀不能被選中。"""
    params = ScoringParams()
    clean = make_detection(1, (0.35, 0.35, 0.70, 0.72), 0)
    bigger_but_at_edge = make_detection(1, (0.30, 0.30, 0.99, 0.85), NS_PER_SEC)

    # 先確認前提:不排除的話,貼邊那幀分數確實比較高
    assert score_detection(bigger_but_at_edge, params) > score_detection(
        clean, params
    )

    tracker = VehicleTracker(scoring=params)
    tracker.update([clean, bigger_but_at_edge])

    best = tracker.get_track(1).best_frame()
    assert best is not None
    assert not best.from_edge_fallback
    assert best.detection is clean


def test_approaching_car_picks_last_fully_visible_frame() -> None:
    tracker = VehicleTracker(scoring=ScoringParams())
    frames = approaching_car(1, 0, frames=30)
    for detection in frames:
        tracker.update([detection])

    track = tracker.get_track(1)
    best = track.best_frame()

    assert best is not None
    assert not best.from_edge_fallback
    assert track.frame_count == 30
    assert track.edge_frame_count > 0
    # 選中的幀本身必須是乾淨的
    assert not touches_edge(best.detection.bbox, 0.02)
    # 而且不是第一幀(車子太遠)也不是最後一幀(已出框)
    assert 1 < best.frame_index < 30


def test_all_frames_at_edge_falls_back_to_least_bad() -> None:
    """低幀率時整條 track 可能每一幀都碰邊界 —— 不放棄,取最不靠邊的。"""
    tracker = VehicleTracker(scoring=ScoringParams())
    boxes = [
        (0.0, 0.3, 0.50, 0.90),      # 貼左緣,距離 0.0
        (0.005, 0.3, 0.60, 0.90),    # 距離 0.005
        (0.019, 0.3, 0.70, 0.95),    # 距離 0.019 —— 最不靠邊
    ]
    for index, bbox in enumerate(boxes):
        tracker.update([make_detection(2, bbox, index * NS_PER_SEC // 30)])

    track = tracker.get_track(2)
    best = track.best_frame()

    assert best is not None
    assert best.from_edge_fallback
    assert best.frame_index == 3
    assert best.edge_distance == pytest.approx(0.019)
    assert track.edge_frame_count == 3


def test_track_with_no_detections_has_no_best_frame() -> None:
    tracker = VehicleTracker(scoring=ScoringParams())
    tracker.update([])
    assert tracker.get_track(1) is None


def test_normalize_sharpness_saturates() -> None:
    assert normalize_sharpness(0.0, 500.0) == 0.0
    assert normalize_sharpness(250.0, 500.0) == pytest.approx(0.5)
    assert normalize_sharpness(500.0, 500.0) == 1.0
    # 超過參考值不再加分,否則清晰度項會蓋過另外兩項
    assert normalize_sharpness(5000.0, 500.0) == 1.0


def test_sharpness_can_outweigh_a_slightly_larger_blurry_frame() -> None:
    params = ScoringParams()
    big_and_blurry = make_detection(3, (0.30, 0.30, 0.80, 0.80), 0, sharpness=15.0)
    small_and_sharp = make_detection(
        3, (0.34, 0.34, 0.74, 0.76), NS_PER_SEC, sharpness=900.0
    )

    tracker = VehicleTracker(scoring=params)
    tracker.update([big_and_blurry, small_and_sharp])

    best = tracker.get_track(3).best_frame()
    assert best is not None
    assert best.detection is small_and_sharp


def test_sharpness_weight_zero_ignores_sharpness() -> None:
    """權重是 config 來的 —— 關掉清晰度項時要真的不影響結果。"""
    params = ScoringParams(sharpness_weight=0.0)
    blurry = make_detection(4, (0.30, 0.30, 0.80, 0.80), 0, sharpness=0.0)
    sharp = make_detection(4, (0.30, 0.30, 0.80, 0.80), 0, sharpness=5000.0)
    assert score_detection(blurry, params) == score_detection(sharp, params)


def test_scoring_params_come_from_config() -> None:
    from tests.helpers import make_config

    config = make_config(
        score_area_weight=0.1,
        score_center_weight=0.2,
        score_sharpness_weight=0.7,
        sharpness_reference=250.0,
        edge_margin=0.05,
    )
    params = ScoringParams.from_config(config)

    assert params.area_weight == 0.1
    assert params.center_weight == 0.2
    assert params.sharpness_weight == 0.7
    assert params.sharpness_reference == 250.0
    assert params.edge_margin == 0.05
