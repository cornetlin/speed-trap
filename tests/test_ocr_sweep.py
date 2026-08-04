"""ocr_sweep 的統計邏輯。

這些數字要拿來決定「哪個前處理值得接進 production」,算錯會把整場實驗
的結論帶偏,所以純邏輯的部分要有測試。影像處理本身不在這裡測(需要
opencv 與真實圖片)。
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.ocr_sweep import (
    Attempt,
    character_accuracy,
    levenshtein,
    load_truth,
    summarise,
    width_bucket,
    write_by_width,
    write_detail,
    write_summary,
)


def _attempt(
    filename: str,
    variant: str,
    predicted: str,
    truth: str,
    width: int = 45,
) -> Attempt:
    return Attempt(
        filename=filename,
        variant=variant,
        predicted=predicted,
        truth=truth,
        width=width,
        height=int(width * 0.7),
        confidence=0.9,
        failure="",
    )


# --- 編輯距離與字元正確率 ----------------------------------------------


def test_levenshtein_basics() -> None:
    assert levenshtein("ABC1234", "ABC1234") == 0
    assert levenshtein("", "ABC") == 3
    assert levenshtein("ABC", "") == 3
    assert levenshtein("", "") == 0


def test_levenshtein_counts_single_substitution() -> None:
    # 形近字元誤判就是這種情況:位數全對、只錯一個字
    assert levenshtein("AQC1234", "AMC1234") == 1
    assert levenshtein("VBC1234", "WBC1234") == 1


def test_levenshtein_handles_length_changes() -> None:
    assert levenshtein("ABC123", "ABC1234") == 1     # 少一位
    assert levenshtein("ABC12345", "ABC1234") == 1   # 多一位


def test_character_accuracy_distinguishes_one_error_from_five() -> None:
    """整串對錯太粗糙 —— 錯 1 個和錯 5 個必須分得出來。"""
    one_wrong = character_accuracy("AQC1234", "AMC1234")
    five_wrong = character_accuracy("XXXXX34", "AMC1234")

    assert one_wrong == pytest.approx(6 / 7)
    assert five_wrong == pytest.approx(2 / 7)
    assert one_wrong > five_wrong


def test_character_accuracy_edge_cases() -> None:
    assert character_accuracy("", "ABC1234") == 0.0
    assert character_accuracy("ABC1234", "ABC1234") == 1.0
    # 亂讀一長串不應該出現負分
    assert character_accuracy("ZZZZZZZZZZZZZZZ", "ABC1234") == 0.0
    assert character_accuracy("anything", "") == 0.0


# --- 寬度分組 -----------------------------------------------------------


def test_width_bucket_boundaries() -> None:
    assert width_bucket(21) == "<40px"
    assert width_bucket(39) == "<40px"
    assert width_bucket(40) == "40-60px"
    assert width_bucket(43) == "40-60px"     # 讀錯的中位數
    assert width_bucket(48) == "40-60px"     # 讀對的中位數
    assert width_bucket(60) == "40-60px"
    assert width_bucket(61) == ">60px"
    assert width_bucket(103) == ">60px"


# --- 彙整 ---------------------------------------------------------------


def test_summarise_counts_exact_and_char_accuracy() -> None:
    attempts = [
        _attempt("a.jpg", "baseline", "ABC1234", "ABC1234"),
        _attempt("b.jpg", "baseline", "AQC1234", "AMC1234"),
        _attempt("c.jpg", "baseline", "", "XYZ5678"),
    ]
    summary = summarise(attempts)[0]

    assert summary.variant == "baseline"
    assert summary.total == 3
    assert summary.correct == 1
    assert summary.empty_reads == 1
    # 編輯距離 0 + 1 + 7 = 8,答案總長 7*3 = 21
    assert summary.edit_distance_total == 8
    assert summary.char_accuracy == pytest.approx(1 - 8 / 21)


def test_summarise_tracks_fixed_and_broken_against_baseline() -> None:
    """只看總數會漏掉「救回三張又弄壞三張」這種淨值為零的情況。"""
    attempts = [
        _attempt("a.jpg", "baseline", "AQC1234", "AMC1234"),   # 錯
        _attempt("b.jpg", "baseline", "XYZ5678", "XYZ5678"),   # 對
        _attempt("c.jpg", "baseline", "AAA1111", "AAA1111"),   # 對
        _attempt("a.jpg", "cubic_3x", "AMC1234", "AMC1234"),   # 變對
        _attempt("b.jpg", "cubic_3x", "XVZ5678", "XYZ5678"),   # 變錯
        _attempt("c.jpg", "cubic_3x", "AAA1111", "AAA1111"),   # 不變
    ]
    summaries = {s.variant: s for s in summarise(attempts)}
    variant = summaries["cubic_3x"]

    assert variant.fixed == ["a.jpg"]
    assert variant.broken == ["b.jpg"]
    assert variant.correct == 2
    # 淨值為零,但實際上換掉了兩張 —— 這正是要看得見的資訊
    assert len(variant.fixed) - len(variant.broken) == 0


def test_summarise_baseline_has_no_fixed_or_broken() -> None:
    attempts = [
        _attempt("a.jpg", "baseline", "AQC1234", "AMC1234"),
        _attempt("a.jpg", "cubic_3x", "AMC1234", "AMC1234"),
    ]
    baseline = summarise(attempts)[0]
    assert baseline.variant == "baseline"
    assert baseline.fixed == []
    assert baseline.broken == []


def test_summarise_puts_baseline_first_then_best_char_accuracy() -> None:
    attempts = [
        _attempt("a.jpg", "baseline", "", "ABC1234"),
        _attempt("a.jpg", "worse", "ZZZ9999", "ABC1234"),
        _attempt("a.jpg", "better", "ABC1234", "ABC1234"),
    ]
    order = [s.variant for s in summarise(attempts)]
    assert order == ["baseline", "better", "worse"]


def test_summarise_groups_by_width() -> None:
    attempts = [
        _attempt("small.jpg", "baseline", "ABC1234", "ABC1234", width=30),
        _attempt("mid.jpg", "baseline", "XXX1234", "AMC1234", width=45),
        _attempt("big.jpg", "baseline", "XYZ5678", "XYZ5678", width=80),
    ]
    summary = summarise(attempts)[0]

    assert summary.by_width["<40px"] == (1, 1)
    assert summary.by_width["40-60px"] == (0, 1)
    assert summary.by_width[">60px"] == (1, 1)


def test_summarise_uses_original_width_not_upscaled() -> None:
    """放大後的圖不能影響分組 —— 要看的是「原本多小」。"""
    attempts = [
        _attempt("a.jpg", "baseline", "ABC1234", "ABC1234", width=35),
        _attempt("a.jpg", "cubic_4x", "ABC1234", "ABC1234", width=35),
    ]
    for summary in summarise(attempts):
        assert set(summary.by_width) == {"<40px"}


# --- 對照表讀取 ---------------------------------------------------------


def test_load_truth_skips_unverified_rows(tmp_path: Path) -> None:
    path = tmp_path / "truth.csv"
    path.write_text(
        "filename,truth\n"
        "00001_plate.jpg,ABC-1234\n"
        "00002_plate.jpg,\n"          # 還沒人工核對
        "00003_plate.jpg,   \n"       # 只有空白
        "00004_plate.jpg,xyz 5678\n",
        encoding="utf-8",
    )
    truth = load_truth(path)

    assert truth == {"00001_plate.jpg": "ABC1234", "00004_plate.jpg": "XYZ5678"}


def test_load_truth_normalises_like_the_ocr_output(tmp_path: Path) -> None:
    """答案要跟 OCR 輸出走同一套正規化,否則會拿 ABC-1234 去比 ABC1234。"""
    path = tmp_path / "truth.csv"
    path.write_text("filename,truth\na.jpg,ab-1234\n", encoding="utf-8")
    assert load_truth(path)["a.jpg"] == "AB1234"


def test_load_truth_accepts_paths_in_filename_column(tmp_path: Path) -> None:
    path = tmp_path / "truth.csv"
    path.write_text(
        "filename,truth\n/home/kevin30/speedtrap_debug/a.jpg,ABC1234\n",
        encoding="utf-8",
    )
    assert "a.jpg" in load_truth(path)


def test_load_truth_rejects_missing_header(tmp_path: Path) -> None:
    path = tmp_path / "truth.csv"
    path.write_text("a.jpg,ABC1234\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_truth(path)


# --- CSV 輸出 -----------------------------------------------------------


def test_csv_outputs_are_written(tmp_path: Path) -> None:
    attempts = [
        _attempt("a.jpg", "baseline", "AQC1234", "AMC1234", width=35),
        _attempt("a.jpg", "cubic_3x", "AMC1234", "AMC1234", width=35),
    ]
    summaries = summarise(attempts)

    detail = tmp_path / "detail.csv"
    summary = tmp_path / "summary.csv"
    bywidth = tmp_path / "bywidth.csv"
    write_detail(detail, attempts)
    write_summary(summary, summaries)
    write_by_width(bywidth, summaries)

    detail_rows = list(csv.DictReader(detail.open(encoding="utf-8")))
    assert len(detail_rows) == 2
    assert detail_rows[0]["predicted"] == "AQC1234"
    assert detail_rows[0]["truth"] == "AMC1234"
    assert detail_rows[0]["correct"] == "0"
    assert detail_rows[1]["correct"] == "1"
    assert detail_rows[0]["width_bucket"] == "<40px"

    summary_rows = {
        row["variant"]: row
        for row in csv.DictReader(summary.open(encoding="utf-8"))
    }
    assert summary_rows["cubic_3x"]["fixed_files"] == "a.jpg"
    assert summary_rows["cubic_3x"]["net_gain"] == "1"

    width_rows = list(csv.DictReader(bywidth.open(encoding="utf-8")))
    assert {row["width_bucket"] for row in width_rows} == {"<40px"}


def test_preprocessor_registry_has_the_expected_variants() -> None:
    """加新前處理只要往字典加一筆 —— 這裡固定住既有的名稱。"""
    from scripts.ocr_sweep import PREPROCESSORS

    assert "baseline" in PREPROCESSORS
    for name in (
        "cubic_2x",
        "cubic_3x",
        "cubic_4x",
        "lanczos_3x",
        "gray",
        "gray_clahe",
        "gray_clahe_cubic_3x",
        "unsharp_cubic_3x",
    ):
        assert name in PREPROCESSORS, name
    assert all(callable(fn) for fn in PREPROCESSORS.values())
