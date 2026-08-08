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
    FIX_ALREADY_VALID,
    FIX_AMBIGUOUS,
    FIX_CORRECTED,
    FIX_EMPTY,
    FIX_NO_CANDIDATE,
    FMTFIX_SUFFIX,
    Attempt,
    character_accuracy,
    derive_format_fix_attempts,
    format_constrained_fix,
    levenshtein,
    load_truth,
    summarise,
    summarise_format_fix,
    width_bucket,
    write_by_width,
    write_detail,
    write_format_fix,
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


def test_preprocessor_registry_is_the_one_production_uses() -> None:
    """掃描量的必須就是線上跑的那一份,否則結論套上去不成立。"""
    from scripts.ocr_sweep import PREPROCESSORS
    from speed_trap import preprocess

    assert PREPROCESSORS is preprocess.PREPROCESSORS
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


# --- 台灣車牌格式約束校正 ------------------------------------------------


def test_observed_pairs_cannot_change_validity() -> None:
    """現場核對出來的十對混淆字元,全部落在同一個字元類別。

    這代表單靠它們做替換,格式驗證的結果必然不變 —— 一個字都救不回來。
    這正是那 10 筆「位數全對、只錯一個字」的處境:形狀本來就對,字串已經
    通過格式驗證,格式約束沒有任何資訊可用。

    這個測試是要把這件事釘在程式碼裡,不是描述一個 bug。
    """
    from scripts.ocr_sweep import _OBSERVED_PAIRS

    for left, right in _OBSERVED_PAIRS:
        assert left.isdigit() == right.isdigit(), (left, right)


def test_cross_class_pairs_do_change_shape() -> None:
    """跨類別的那一組才是格式約束派得上用場的地方。"""
    from scripts.ocr_sweep import _CROSS_CLASS_PAIRS

    for left, right in _CROSS_CLASS_PAIRS:
        assert left.isdigit() != right.isdigit(), (left, right)


def test_valid_plate_is_left_alone() -> None:
    fix = format_constrained_fix("ABC1234")
    assert fix.action == FIX_ALREADY_VALID
    assert fix.text == "ABC1234"


def test_single_candidate_is_adopted() -> None:
    """A8C1234 只有把 8 換成 B 才合法 —— 恰好一個候選,採用。"""
    fix = format_constrained_fix("A8C1234")
    assert fix.action == FIX_CORRECTED
    assert fix.text == "ABC1234"
    assert fix.candidates == ("ABC1234",)


def test_two_candidates_leave_the_original_alone() -> None:
    """A0C1234 的 0 換成 D 或 Q 都合法 —— 選哪個都是猜,一律不動。"""
    fix = format_constrained_fix("A0C1234")
    assert fix.action == FIX_AMBIGUOUS
    assert fix.text == "A0C1234"
    assert fix.candidates == ("ADC1234", "AQC1234")


def test_no_candidate_leaves_the_original_alone() -> None:
    """長度不對時換一個字救不了 —— 替換不會改變長度。"""
    fix = format_constrained_fix("ABC12345")
    assert fix.action == FIX_NO_CANDIDATE
    assert fix.text == "ABC12345"


def test_empty_read_is_not_invented_into_a_plate() -> None:
    fix = format_constrained_fix("")
    assert fix.action == FIX_EMPTY
    assert fix.text == ""


def test_illegal_letter_o_is_corrected_back_to_zero() -> None:
    """台灣車牌沒有 O,所以 OCR 讀出 O 一定是錯的 —— 換回 0 才可能合法。"""
    fix = format_constrained_fix("ABC1O34")
    assert fix.action == FIX_CORRECTED
    assert fix.text == "ABC1034"


# --- fmtfix 變體的彙整 ---------------------------------------------------


def test_derive_adds_one_fmtfix_row_per_attempt_without_rerunning_ocr() -> None:
    attempts = [
        _attempt("a.jpg", "baseline", "A8C1234", "ABC1234"),
        _attempt("b.jpg", "baseline", "XYZ5678", "XYZ5678"),
    ]
    derived = derive_format_fix_attempts(attempts)

    assert [a.variant for a in derived] == [f"baseline{FMTFIX_SUFFIX}"] * 2
    assert derived[0].predicted == "ABC1234"      # 救回
    assert derived[1].predicted == "XYZ5678"      # 本來就對,不動
    # 尺寸與信心度照抄 —— 校正只碰字串
    assert derived[0].width == attempts[0].width
    assert derived[0].confidence == attempts[0].confidence


def test_derive_is_idempotent_on_already_derived_rows() -> None:
    """避免重複呼叫時長出 baseline+fmtfix+fmtfix。"""
    attempts = [_attempt("a.jpg", "baseline", "A8C1234", "ABC1234")]
    once = attempts + derive_format_fix_attempts(attempts)
    twice = once + derive_format_fix_attempts(once)
    assert len(twice) == 3


def test_summarise_format_fix_counts_fixed_and_broken() -> None:
    """救回與弄壞要分開看 —— 淨值為零不代表沒事發生。"""
    attempts = [
        _attempt("a.jpg", "cubic_3x", "A8C1234", "ABC1234"),   # 會被救回
        _attempt("b.jpg", "cubic_3x", "XYZ5678", "XYZ5678"),   # 本來就對
        _attempt("c.jpg", "cubic_3x", "ABC12345", "ABC1234"),  # 無解
    ]
    attempts += derive_format_fix_attempts(attempts)
    summary = summarise_format_fix(attempts)[0]

    assert summary.base_variant == "cubic_3x"
    assert summary.total == 3
    assert summary.base_correct == 1
    assert summary.fixed_correct == 2
    assert summary.fixed == ["a.jpg"]
    assert summary.broken == []
    assert summary.net_gain == 1
    assert summary.corrected == 1
    assert summary.already_valid == 1
    assert summary.no_candidate == 1


def test_summarise_format_fix_records_a_correction_that_went_wrong() -> None:
    """把對的改成錯的一定要看得見 —— 這是決定要不要上線的關鍵數字。

    這不是假想的情況:真實車牌只要不符合 plate_format 的四種格式(特殊牌、
    外交牌、電動車牌),OCR 讀對了反而會被校正成另一個「合法」的字串。
    """
    attempts = [_attempt("a.jpg", "baseline", "A8C1234", "A8C1234")]
    attempts += derive_format_fix_attempts(attempts)
    summary = summarise_format_fix(attempts)[0]

    assert summary.broken == ["a.jpg"]
    assert summary.net_gain == -1


def test_format_fix_csv_is_written(tmp_path: Path) -> None:
    attempts = [_attempt("a.jpg", "baseline", "A8C1234", "ABC1234")]
    attempts += derive_format_fix_attempts(attempts)
    path = tmp_path / "fmtfix.csv"
    write_format_fix(path, summarise_format_fix(attempts))

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 1
    assert rows[0]["base_variant"] == "baseline"
    assert rows[0]["net_gain"] == "1"
    assert rows[0]["fixed_files"] == "a.jpg"


def test_detail_csv_carries_the_fmtfix_note(tmp_path: Path) -> None:
    attempts = [_attempt("a.jpg", "baseline", "A0C1234", "ADC1234")]
    attempts += derive_format_fix_attempts(attempts)
    path = tmp_path / "detail.csv"
    write_detail(path, attempts)

    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert rows[0]["note"] == ""
    # 兩解時把候選也記進去,人工核對才知道系統在猶豫什麼
    assert rows[1]["note"].startswith(FIX_AMBIGUOUS)
    assert "ADC1234" in rows[1]["note"]
