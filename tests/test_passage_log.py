"""逐台車 CSV 與最佳幀裁切圖。

現場實驗成本很高(要在白天等車流),所以一次實驗要能回答完所有問題 ——
這份診斷紀錄壞掉的代價是整場實驗白跑。
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from tests.helpers import (
    NS_PER_SEC,
    FakeSource,
    FixedOCR,
    ListSink,
    approaching_car,
    make_config,
    make_detection,
)

from apps.run_station import _settle_finished, _StopFlag, run_station
from speed_trap.passage_log import PassageLogger, PassageRecord
from speed_trap.tracker import ScoringParams, VehicleTracker
from speed_trap.trigger_line import TriggerLineDetector


def _run(tmp_path: Path, ocr, *, detections=None, **config_overrides):
    config = make_config(
        passage_csv_path=str(tmp_path / "events.csv"),
        passage_crop_dir=str(tmp_path / "crops"),
        **config_overrides,
    )
    logger = PassageLogger(config.passage_csv_path, config.passage_crop_dir)
    sink = ListSink()
    if detections is None:
        detections = approaching_car(11, time.monotonic_ns(), frames=30)
    run_station(
        config,
        sink,
        FakeSource(detections),
        _StopFlag(),
        recognizer=ocr,
        passage_log=logger,
    )
    logger.close()
    return config, sink


def _rows(config) -> list[dict[str, str]]:
    with Path(config.passage_csv_path).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_csv_records_the_full_decision_trail(tmp_path: Path) -> None:
    config, sink = _run(tmp_path, FixedOCR("ABC1234"))
    rows = _rows(config)

    assert len(rows) == 1
    row = rows[0]
    assert row["seq"] == "1"
    assert row["track_id"] == "11"
    assert row["label"] == "car"
    assert row["station_id"] == "test_station"
    # 幀數與挑幀過程
    assert int(row["total_frames"]) == 30
    assert int(row["edge_frames"]) > 0
    assert 1 < int(row["chosen_frame_index"]) < 30
    assert float(row["chosen_score"]) > 0.0
    assert row["chosen_from_edge_fallback"] == "0"
    assert row["touches_edge"] == "0"
    # 尺寸 —— 判斷車牌讀不讀得出來的關鍵數字
    assert int(row["vehicle_crop_w"]) > 0
    assert int(row["plate_box_w"]) == 196
    # OCR 與結果
    assert row["plate_text"] == "ABC1234"
    assert row["plate_is_taiwan_format"] == "1"
    assert row["emitted"] == "1"
    assert row["needs_review"] == "0"
    assert row["crop_file"]


def test_failed_reads_are_still_recorded_with_a_crop(tmp_path: Path) -> None:
    """讀不出車牌的那幾台才是要回頭看的,不能因為失敗就不留紀錄。"""
    config, sink = _run(tmp_path, FixedOCR(None))
    rows = _rows(config)

    assert len(rows) == 1
    assert rows[0]["plate_text"] == ""
    assert rows[0]["ocr_failure"] == "no_plate_detected"
    assert rows[0]["needs_review"] == "1"
    assert "no_plate_read" in rows[0]["review_reason"]

    crops = list((tmp_path / "crops").iterdir())
    assert len(crops) == 1
    assert "NOPLATE" in crops[0].name


def test_crop_filename_maps_back_to_the_csv_row(tmp_path: Path) -> None:
    config, sink = _run(tmp_path, FixedOCR("XY5678"))
    row = _rows(config)[0]

    crop_path = tmp_path / "crops" / row["crop_file"]
    assert crop_path.exists()
    assert crop_path.name.startswith(f"{int(row['seq']):05d}_")
    assert f"track{row['track_id']}" in crop_path.name
    assert "XY5678" in crop_path.name


def test_suppressed_duplicate_is_written_but_not_emitted(tmp_path: Path) -> None:
    base = time.monotonic_ns()
    detections = approaching_car(13, base, frames=15) + approaching_car(
        14, base + 2 * NS_PER_SEC, frames=15
    )
    config, sink = _run(tmp_path, FixedOCR("ABC1234"), detections=detections)
    rows = _rows(config)

    assert len(rows) == 2, "兩條 track 都要留下紀錄"
    assert [row["emitted"] for row in rows] == ["1", "0"]
    assert "duplicate_plate" in rows[1]["review_reason"]
    assert rows[1]["duplicate_of_track"] == "13"
    assert len(sink.events) == 1


def test_sharpness_failure_is_flagged(tmp_path: Path) -> None:
    """清晰度算不出來時評分只剩面積與置中,CSV 必須看得出來。"""
    config = make_config(
        passage_csv_path=str(tmp_path / "events.csv"),
        passage_crop_dir=str(tmp_path / "crops"),
    )
    tracker = VehicleTracker(
        stale_timeout_ns=int(config.track_idle_timeout_s * NS_PER_SEC),
        scoring=ScoringParams.from_config(config),
    )
    trigger = TriggerLineDetector(line_y=config.trigger_line_y)
    logger = PassageLogger(config.passage_csv_path, config.passage_crop_dir)
    sink = ListSink()

    for detection in approaching_car(21, 0, frames=20):
        broken = make_detection(
            detection.track_id,
            detection.bbox,
            detection.frame_ns,
            frame_jpeg=detection.frame_jpeg,
            sharpness=0.0,
            sharpness_failed=True,
        )
        tracker.update([broken])
        track = tracker.get_track(21)
        if trigger.check_crossing(track, broken) and not trigger.was_triggered(21):
            trigger.mark_triggered(21)

    _settle_finished(config, tracker.drain(), trigger, sink, FixedOCR("ABC1234"), None, logger)
    logger.close()

    row = _rows(config)[0]
    assert row["chosen_sharpness_failed"] == "1"
    assert float(row["chosen_sharpness"]) == 0.0


def test_csv_appends_across_runs_without_losing_rows(tmp_path: Path) -> None:
    """重跑站台不能蓋掉上一場實驗的資料。"""
    config, _ = _run(tmp_path, FixedOCR("ABC1234"))
    assert len(_rows(config)) == 1

    logger = PassageLogger(config.passage_csv_path, config.passage_crop_dir)
    run_station(
        config,
        ListSink(),
        FakeSource(approaching_car(31, time.monotonic_ns(), frames=30)),
        _StopFlag(),
        recognizer=FixedOCR("QQ1111"),
        passage_log=logger,
    )
    logger.close()

    rows = _rows(config)
    assert [row["seq"] for row in rows] == ["1", "2"]
    assert [row["track_id"] for row in rows] == ["11", "31"]


def test_logger_can_be_disabled(tmp_path: Path) -> None:
    logger = PassageLogger("", "")
    assert not logger.enabled
    logger.log(PassageRecord(track_id=1), b"jpeg")   # 不該爆炸
    logger.close()
    assert not list(tmp_path.iterdir())


def test_record_field_names_match_csv_header(tmp_path: Path) -> None:
    config, _ = _run(tmp_path, FixedOCR("ABC1234"))
    with Path(config.passage_csv_path).open(encoding="utf-8") as handle:
        header = next(csv.reader(handle))
    assert header == PassageRecord.field_names()
