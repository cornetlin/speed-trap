#!/usr/bin/env python3
"""離線掃描:同一批車牌裁切圖,逐一套用各種前處理,比較 OCR 讀對率。

不需要相機、不需要 Hailo,純粹讀圖跑 OCR。在 Pi 上執行。

要回答的問題
------------
現場 79 台車、54 台讀出字串,人工核對只有約 33 筆正確。錯誤集中在形近
字元(V/W、6/8、7/9、F/C、0/9、P/R、W/K、A/X、V/L、M/Q),其中 10 筆是
「位數全對、只錯一個字」—— 車牌框得準,純粹是字元判讀能力不足。把一張
60x43 的圖放大六倍用肉眼看,Q 和 M 是分得出來的,代表資訊在圖裡,只是
OCR 沒取出來。所以前處理有機會救回一部分。

為什麼要看字元層級正確率
------------------------
整串對不對太粗糙:7 個字元錯 1 個和錯 5 個都算「錯」,看不出改善趨勢。
這裡用編輯距離算字元層級正確率,才看得出某個前處理是「把 5 個錯變成
1 個錯」還是「完全沒動靜」。

用法
----
    python scripts/ocr_sweep.py
    python scripts/ocr_sweep.py --crops ~/speedtrap_debug --truth ~/plate_truth.csv
    python scripts/ocr_sweep.py --limit 10          # 先跑幾張確認流程

對照表 CSV 兩欄 ``filename,truth``;truth 為空的列會跳過(還沒人工核對)。

輸出四份 CSV(檔名帶時間戳,重跑不會蓋掉舊結果):
    ocr_sweep_<時間戳>_summary.csv   每種前處理一列
    ocr_sweep_<時間戳>_detail.csv    每張圖每種前處理一列
    ocr_sweep_<時間戳>_bywidth.csv   前處理 x 車牌寬度分組
    ocr_sweep_<時間戳>_fmtfix.csv    車牌格式約束校正的救回 / 弄壞

前處理的實作在 speed_trap/preprocess.py,與 station 線上跑的是同一份 ——
掃描出來的結論才適用於線上。

格式約束校正(fmtfix)是後處理,套在 OCR 輸出的字串上,不用重跑 OCR,
所以每個前處理都會自動多一列 "<前處理>+fmtfix"。它目前只存在於掃描裡,
還沒進 production。
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from speed_trap import preprocess  # noqa: E402
from speed_trap.config import StationConfig, load_config  # noqa: E402
from speed_trap.plate_format import (  # noqa: E402
    is_valid_taiwan_plate,
    normalize_plate,
)
from speed_trap.plate_recognizer import (  # noqa: E402
    OcrBackendUnavailable,
    installed_version,
    make_recognizer,
)

_logger = logging.getLogger("ocr_sweep")

# cv2 / numpy 跟專案其他地方一樣延後失敗:模組要能 import(才測得到純邏輯),
# 真的要跑掃描時才需要它們。
_IMPORT_ERROR: str | None
_cv2: Any = None
_np: Any = None
try:
    import cv2 as _cv2_real
    import numpy as _np_real

    _cv2 = _cv2_real
    _np = _np_real
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - 取決於環境
    _IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


DEFAULT_CROPS = Path("~/speedtrap_debug")
DEFAULT_TRUTH = Path("~/plate_truth.csv")
DEFAULT_CONFIG = _REPO_ROOT / "config" / "station_a.yaml"
DEFAULT_PATTERN = "*_plate.jpg"

# 車牌寬度分組:實測讀錯的中位數 43 px、讀對的 48 px,分界抓在這附近才
# 看得出前處理是不是特別救得到小圖。
WIDTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    ("<40px", 0, 40),
    ("40-60px", 40, 61),
    (">60px", 61, 10_000),
)


# --- 前處理 ------------------------------------------------------------
#
# 實作在 speed_trap/preprocess.py —— 那也是 station 線上跑的同一份。這裡刻意
# 不自己再寫一份:兩邊各寫一份的話,掃描出來的結論套到線上就不成立。
# 要加新的前處理就往那個模組的 PREPROCESSORS 加一筆,掃描與線上同時就有。

PREPROCESSORS: dict[str, Callable[[Any], Any]] = preprocess.PREPROCESSORS
BASELINE = preprocess.BASELINE


# --- 台灣車牌格式約束校正(後處理,實驗中)-----------------------------
#
# 現場 19 筆錯誤裡有 10 筆是「位數全對、只錯一個字」,而 plate_format 已經
# 知道台灣車牌不含 I/O、合法格式只有四種。把不合法的結果拿去試單字元替換,
# 有機會把那 10 筆救回來。
#
# 風險是把原本就對的改成錯的 —— 恰好一個合法候選才敢動,兩個以上一律不動
# (例如 ABC123X 同時能改成兩種合法字串,選哪個都是猜)。這個變體目前**只**
# 存在於掃描裡,確認淨值明顯為正才考慮進 production。

# --- 混淆字元表 ---
#
# 分成兩組,理由很實際:格式約束只看得見「形狀」(哪一位是字母、哪一位是
# 數字)與「字母不得為 I/O」。同類別之間的替換不會改變形狀,所以對格式驗證
# 完全沒有作用。
#
# OBSERVED 這一組是現場人工核對出來的實際誤判,十對全部落在同一類別
# (字母↔字母 或 數字↔數字)—— 也就是說,單靠它們一個字都救不回來:替換
# 前後 is_valid_taiwan_plate 的結果必然相同。這件事有測試釘住
# (test_observed_pairs_cannot_change_validity)。
#
# 這正好解釋了為什麼那 10 筆「位數全對、只錯一個字」救不了:位數全對代表
# 形狀本來就對,字串本身已經通過格式驗證了,只是不是那台車的車牌。格式
# 約束對這種錯誤沒有任何資訊可用。
#
# CROSS_CLASS 這一組才是格式約束派得上用場的地方:字母被讀成數字或反過來,
# 形狀跑掉、格式驗證抓得到,而且候選字通常唯一。留 OBSERVED 在表裡是為了
# 它們能跟 CROSS_CLASS 組合出候選(例如同時要換位置與換類別時)。
_OBSERVED_PAIRS: tuple[tuple[str, str], ...] = (
    ("V", "W"),
    ("6", "8"),
    ("7", "9"),
    ("F", "C"),
    ("0", "9"),
    ("P", "R"),
    ("W", "K"),
    ("A", "X"),
    ("V", "L"),
    ("M", "Q"),
)

# 跨類別的字形混淆。O 與 I 不是合法的台灣車牌字母,所以它們只會出現在
# 「OCR 讀出來的錯字」這一側 —— 往回換成 0 / 1 才是有效的修正方向,
# 反方向產生的候選一定通不過格式驗證,不必特別排除。
_CROSS_CLASS_PAIRS: tuple[tuple[str, str], ...] = (
    ("0", "D"),
    ("0", "O"),
    ("0", "Q"),
    ("1", "I"),
    ("1", "L"),
    ("2", "Z"),
    ("4", "A"),
    ("5", "S"),
    ("6", "G"),
    ("7", "T"),
    ("8", "B"),
)


def _build_confusables(
    *pair_groups: tuple[tuple[str, str], ...],
) -> dict[str, tuple[str, ...]]:
    table: dict[str, set[str]] = {}
    for group in pair_groups:
        for left, right in group:
            table.setdefault(left, set()).add(right)
            table.setdefault(right, set()).add(left)
    return {char: tuple(sorted(others)) for char, others in table.items()}


CONFUSABLES: dict[str, tuple[str, ...]] = _build_confusables(
    _OBSERVED_PAIRS, _CROSS_CLASS_PAIRS
)

# fmtfix 變體的命名後綴。summary CSV 上 "unsharp_cubic_3x+fmtfix" 就是
# "unsharp_cubic_3x" 再套格式校正。
FMTFIX_SUFFIX = "+fmtfix"

# FormatFix.action 的取值
FIX_EMPTY = "empty"                  # OCR 根本沒讀到字串
FIX_ALREADY_VALID = "already_valid"  # 本來就合法,不動
FIX_CORRECTED = "corrected"          # 恰好一個合法候選,採用
FIX_AMBIGUOUS = "ambiguous"          # 多於一個合法候選,維持原樣
FIX_NO_CANDIDATE = "no_candidate"    # 換一個字換不出合法字串,維持原樣


@dataclass(frozen=True)
class FormatFix:
    text: str                     # 校正後要採用的字串(不動時等於輸入)
    action: str
    candidates: tuple[str, ...]   # 找到的合法候選,供人工檢查


def format_constrained_fix(text: str) -> FormatFix:
    """不合法的辨識結果,試著用形近字元的單字元替換換成合法車牌。

    只換一個字元:錯兩個字以上的多半不是形近誤判,而是車牌根本沒切好,
    硬猜只會製造看起來合理的錯誤答案。
    """
    if not text:
        return FormatFix(text=text, action=FIX_EMPTY, candidates=())
    if is_valid_taiwan_plate(text):
        return FormatFix(text=text, action=FIX_ALREADY_VALID, candidates=())

    candidates: set[str] = set()
    for index, char in enumerate(text):
        for replacement in CONFUSABLES.get(char, ()):
            candidate = text[:index] + replacement + text[index + 1 :]
            if is_valid_taiwan_plate(candidate):
                candidates.add(candidate)

    ordered = tuple(sorted(candidates))
    if len(ordered) == 1:
        return FormatFix(text=ordered[0], action=FIX_CORRECTED, candidates=ordered)
    if len(ordered) > 1:
        # 選哪個都是猜 —— 維持原樣並記下來,人工核對時才知道發生過什麼。
        return FormatFix(text=text, action=FIX_AMBIGUOUS, candidates=ordered)
    return FormatFix(text=text, action=FIX_NO_CANDIDATE, candidates=())


# --- 純邏輯(可單獨測試)-----------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """編輯距離。自己實作,不為了一支實驗腳本多拉一個相依。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,          # 刪除
                    current[j - 1] + 1,       # 插入
                    previous[j - 1] + (char_a != char_b),  # 替換
                )
            )
        previous = current
    return previous[-1]


def character_accuracy(predicted: str, truth: str) -> float:
    """單筆的字元層級正確率,0..1。空的答案視為無從計分,回傳 0。"""
    if not truth:
        return 0.0
    distance = levenshtein(predicted, truth)
    return max(0.0, 1.0 - distance / len(truth))


def width_bucket(width: int) -> str:
    for label, low, high in WIDTH_BUCKETS:
        if low <= width < high:
            return label
    return WIDTH_BUCKETS[-1][0]


@dataclass
class Attempt:
    """一張圖套一種前處理的結果。"""

    filename: str
    variant: str
    predicted: str
    truth: str
    width: int
    height: int
    confidence: float
    failure: str
    # 後處理留下的註記。目前只有 fmtfix 會填(FIX_* 其中之一),供 detail
    # CSV 交代「這一列的字串為什麼跟它的來源不一樣」。
    note: str = ""

    @property
    def correct(self) -> bool:
        return bool(self.truth) and self.predicted == self.truth

    @property
    def char_accuracy(self) -> float:
        return character_accuracy(self.predicted, self.truth)

    @property
    def edit_distance(self) -> int:
        return levenshtein(self.predicted, self.truth)


@dataclass
class VariantSummary:
    variant: str
    total: int = 0
    correct: int = 0
    fixed: list[str] = field(default_factory=list)      # 從錯變對
    broken: list[str] = field(default_factory=list)     # 從對變錯
    char_accuracy: float = 0.0
    edit_distance_total: int = 0
    truth_chars_total: int = 0
    empty_reads: int = 0
    by_width: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def exact_rate(self) -> float:
        return self.correct / self.total if self.total else 0.0


def summarise(
    attempts: list[Attempt], baseline: str = BASELINE
) -> list[VariantSummary]:
    """把逐張結果整理成每種前處理一列。

    「從錯變對 / 從對變錯」是跟 baseline 逐張比對得來的 —— 只看總數會漏掉
    「救回三張又弄壞三張」這種淨值為零、其實很不穩定的情況。
    """
    by_variant: dict[str, list[Attempt]] = {}
    for attempt in attempts:
        by_variant.setdefault(attempt.variant, []).append(attempt)

    baseline_correct = {
        attempt.filename: attempt.correct
        for attempt in by_variant.get(baseline, [])
    }

    summaries: list[VariantSummary] = []
    for variant, items in by_variant.items():
        summary = VariantSummary(variant=variant, total=len(items))
        bucket_totals: Counter[str] = Counter()
        bucket_correct: Counter[str] = Counter()

        for attempt in items:
            if attempt.correct:
                summary.correct += 1
            if not attempt.predicted:
                summary.empty_reads += 1
            summary.edit_distance_total += attempt.edit_distance
            summary.truth_chars_total += len(attempt.truth)

            was_correct = baseline_correct.get(attempt.filename)
            if was_correct is not None and variant != baseline:
                if attempt.correct and not was_correct:
                    summary.fixed.append(attempt.filename)
                elif was_correct and not attempt.correct:
                    summary.broken.append(attempt.filename)

            label = width_bucket(attempt.width)
            bucket_totals[label] += 1
            bucket_correct[label] += int(attempt.correct)

        if summary.truth_chars_total:
            summary.char_accuracy = max(
                0.0,
                1.0 - summary.edit_distance_total / summary.truth_chars_total,
            )
        summary.by_width = {
            label: (bucket_correct[label], bucket_totals[label])
            for label, _low, _high in WIDTH_BUCKETS
            if bucket_totals[label]
        }
        summary.fixed.sort()
        summary.broken.sort()
        summaries.append(summary)

    # baseline 排最前面,其餘依字元正確率由高到低
    summaries.sort(
        key=lambda s: (s.variant != baseline, -s.char_accuracy, s.variant)
    )
    return summaries


# --- 格式約束校正的彙整 -------------------------------------------------


@dataclass
class FormatFixSummary:
    """一種前處理套上格式校正之後的變化。

    比較對象是**同一種前處理**的未校正版本,不是 baseline —— 要回答的是
    「校正這一步本身是賺是賠」。
    """

    base_variant: str
    total: int = 0
    base_correct: int = 0
    fixed_correct: int = 0
    fixed: list[str] = field(default_factory=list)      # 校正救回
    broken: list[str] = field(default_factory=list)     # 校正弄壞
    corrected: int = 0        # 動了手的筆數(不論對錯)
    ambiguous: int = 0        # 有多個合法候選,放棄不動
    no_candidate: int = 0     # 換一個字換不出合法字串
    already_valid: int = 0    # 本來就合法

    @property
    def net_gain(self) -> int:
        return len(self.fixed) - len(self.broken)


def derive_format_fix_attempts(attempts: list[Attempt]) -> list[Attempt]:
    """替每一筆結果生出對應的 fmtfix 版本。

    校正是套在字串上的後處理,不用重跑 OCR —— 所以這一整組變體是免費的,
    掃描時間不會因此變長。
    """
    derived: list[Attempt] = []
    for attempt in attempts:
        if attempt.variant.endswith(FMTFIX_SUFFIX):
            continue
        fix = format_constrained_fix(attempt.predicted)
        # 兩解以上時把候選一起記下來 —— 人工核對時才看得出系統當時在猶豫
        # 什麼,也才知道要不要為了消歧義再加一條規則。
        note = fix.action
        if fix.action == FIX_AMBIGUOUS:
            note = f"{fix.action}:{'|'.join(fix.candidates)}"
        derived.append(
            Attempt(
                filename=attempt.filename,
                variant=f"{attempt.variant}{FMTFIX_SUFFIX}",
                predicted=fix.text,
                truth=attempt.truth,
                width=attempt.width,
                height=attempt.height,
                confidence=attempt.confidence,
                failure=attempt.failure,
                note=note,
            )
        )
    return derived


def summarise_format_fix(attempts: list[Attempt]) -> list[FormatFixSummary]:
    """逐一比較 "<前處理>" 與 "<前處理>+fmtfix",算出救回與弄壞。"""
    by_key: dict[tuple[str, str], Attempt] = {}
    base_variants: set[str] = set()
    for attempt in attempts:
        by_key[(attempt.variant, attempt.filename)] = attempt
        if not attempt.variant.endswith(FMTFIX_SUFFIX):
            base_variants.add(attempt.variant)

    summaries: list[FormatFixSummary] = []
    for base in sorted(base_variants):
        summary = FormatFixSummary(base_variant=base)
        for (variant, filename), original in by_key.items():
            if variant != base:
                continue
            corrected = by_key.get((f"{base}{FMTFIX_SUFFIX}", filename))
            if corrected is None:
                continue
            summary.total += 1
            summary.base_correct += int(original.correct)
            summary.fixed_correct += int(corrected.correct)
            if corrected.correct and not original.correct:
                summary.fixed.append(filename)
            elif original.correct and not corrected.correct:
                summary.broken.append(filename)
            # note 在兩解時帶了候選字串,所以比對前綴而不是整串
            action = corrected.note.split(":", 1)[0]
            if action == FIX_CORRECTED:
                summary.corrected += 1
            elif action == FIX_AMBIGUOUS:
                summary.ambiguous += 1
            elif action == FIX_NO_CANDIDATE:
                summary.no_candidate += 1
            elif action == FIX_ALREADY_VALID:
                summary.already_valid += 1
        summary.fixed.sort()
        summary.broken.sort()
        summaries.append(summary)

    summaries.sort(key=lambda s: (-s.net_gain, s.base_variant))
    return summaries


def load_truth(path: Path) -> dict[str, str]:
    """讀 filename,truth 對照表。truth 空白的列跳過(還沒人工核對)。"""
    truth: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "filename" not in reader.fieldnames:
            raise SystemExit(
                f"{path} 需要含 filename 欄位的標題列(filename,truth)"
            )
        for row in reader:
            name = (row.get("filename") or "").strip()
            answer = (row.get("truth") or "").strip()
            if not name or not answer:
                continue
            # 跟 OCR 輸出走同一套正規化,否則會拿 "ABC-1234" 去比 "ABC1234"
            truth[Path(name).name] = normalize_plate(answer)
    return truth


# --- 執行 --------------------------------------------------------------


def _encode_jpeg(image: Any) -> bytes | None:
    ok, buffer = _cv2.imencode(".jpg", image, [int(_cv2.IMWRITE_JPEG_QUALITY), 95])
    return bytes(buffer) if ok else None


def run_sweep(
    recognizer: Any,
    images: list[Path],
    truth: dict[str, str],
    variants: dict[str, Callable[[Any], Any]],
) -> list[Attempt]:
    attempts: list[Attempt] = []
    for index, image_path in enumerate(images, start=1):
        raw = _cv2.imread(str(image_path), _cv2.IMREAD_COLOR)
        if raw is None:
            _logger.warning("無法解碼,跳過:%s", image_path)
            continue
        height, width = raw.shape[:2]
        answer = truth.get(image_path.name, "")

        for variant, transform in variants.items():
            try:
                processed = transform(raw)
            except Exception as exc:  # noqa: BLE001 - 前處理壞掉不該中斷掃描
                _logger.warning("%s 前處理 %s 失敗:%s", image_path.name, variant, exc)
                continue
            payload = _encode_jpeg(processed)
            if payload is None:
                _logger.warning("%s 前處理 %s 編碼失敗", image_path.name, variant)
                continue

            result = recognizer.read_plate_only(payload)
            reading = result.reading
            attempts.append(
                Attempt(
                    filename=image_path.name,
                    variant=variant,
                    predicted=reading.text if reading else "",
                    truth=answer,
                    # 一律記原圖尺寸 —— 分組要看的是「原本多小」,
                    # 不是放大後多大
                    width=width,
                    height=height,
                    confidence=reading.confidence if reading else 0.0,
                    failure=result.failure or "",
                )
            )

        if index % 10 == 0:
            _logger.info("已處理 %d/%d 張", index, len(images))
    return attempts


# --- 輸出 --------------------------------------------------------------


def write_detail(path: Path, attempts: list[Attempt]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "filename",
                "variant",
                "predicted",
                "truth",
                "correct",
                "edit_distance",
                "char_accuracy",
                "plate_width_px",
                "plate_height_px",
                "width_bucket",
                "confidence",
                "ocr_failure",
                "note",
            ]
        )
        for attempt in attempts:
            writer.writerow(
                [
                    attempt.filename,
                    attempt.variant,
                    attempt.predicted,
                    attempt.truth,
                    int(attempt.correct),
                    attempt.edit_distance,
                    f"{attempt.char_accuracy:.4f}",
                    attempt.width,
                    attempt.height,
                    width_bucket(attempt.width),
                    f"{attempt.confidence:.4f}",
                    attempt.failure,
                    attempt.note,
                ]
            )


def write_summary(path: Path, summaries: list[VariantSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "variant",
                "exact_correct",
                "total",
                "exact_rate",
                "char_accuracy",
                "edit_distance_total",
                "fixed_count",
                "broken_count",
                "net_gain",
                "empty_reads",
                "fixed_files",
                "broken_files",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.variant,
                    summary.correct,
                    summary.total,
                    f"{summary.exact_rate:.4f}",
                    f"{summary.char_accuracy:.4f}",
                    summary.edit_distance_total,
                    len(summary.fixed),
                    len(summary.broken),
                    len(summary.fixed) - len(summary.broken),
                    summary.empty_reads,
                    ";".join(summary.fixed),
                    ";".join(summary.broken),
                ]
            )


def write_by_width(path: Path, summaries: list[VariantSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["variant", "width_bucket", "correct", "total", "rate"])
        for summary in summaries:
            for label, _low, _high in WIDTH_BUCKETS:
                if label not in summary.by_width:
                    continue
                correct, total = summary.by_width[label]
                writer.writerow(
                    [
                        summary.variant,
                        label,
                        correct,
                        total,
                        f"{correct / total:.4f}" if total else "",
                    ]
                )


def write_format_fix(path: Path, summaries: list[FormatFixSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "base_variant",
                "total",
                "correct_before",
                "correct_after",
                "fixed_count",
                "broken_count",
                "net_gain",
                "corrected",
                "ambiguous",
                "no_candidate",
                "already_valid",
                "fixed_files",
                "broken_files",
            ]
        )
        for summary in summaries:
            writer.writerow(
                [
                    summary.base_variant,
                    summary.total,
                    summary.base_correct,
                    summary.fixed_correct,
                    len(summary.fixed),
                    len(summary.broken),
                    summary.net_gain,
                    summary.corrected,
                    summary.ambiguous,
                    summary.no_candidate,
                    summary.already_valid,
                    ";".join(summary.fixed),
                    ";".join(summary.broken),
                ]
            )


def print_format_fix_report(summaries: list[FormatFixSummary]) -> None:
    print()
    print("=== 台灣車牌格式約束校正(後處理,尚未進 production)===")
    print("淨值要明顯為正才值得接上線 —— 弄壞的是原本就讀對的車,代價比較高。")
    header = (
        f"{'前處理':<22}{'校正前':>8}{'校正後':>8}{'救回':>6}{'弄壞':>6}"
        f"{'淨值':>6}{'動手':>6}{'兩解':>6}{'無解':>6}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary.base_variant:<22}"
            f"{summary.base_correct:>5}/{summary.total:<2}"
            f"{summary.fixed_correct:>5}/{summary.total:<2}"
            f"{len(summary.fixed):>6}"
            f"{len(summary.broken):>6}"
            f"{summary.net_gain:>+6}"
            f"{summary.corrected:>6}"
            f"{summary.ambiguous:>6}"
            f"{summary.no_candidate:>6}"
        )
    print()
    for summary in summaries:
        if not summary.fixed and not summary.broken:
            continue
        print(f"  {summary.base_variant}")
        if summary.fixed:
            print(f"    救回 ({len(summary.fixed)}): {', '.join(summary.fixed)}")
        if summary.broken:
            print(f"    弄壞 ({len(summary.broken)}): {', '.join(summary.broken)}")


def print_report(summaries: list[VariantSummary]) -> None:
    print()
    print("=== 前處理比較(依字元正確率排序,baseline 置頂)===")
    header = (
        f"{'前處理':<22}{'完全正確':>10}{'正確率':>9}"
        f"{'字元正確率':>12}{'救回':>6}{'弄壞':>6}{'淨值':>6}{'讀空':>6}"
    )
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary.variant:<22}"
            f"{summary.correct:>6}/{summary.total:<3}"
            f"{summary.exact_rate * 100:>8.1f}%"
            f"{summary.char_accuracy * 100:>11.1f}%"
            f"{len(summary.fixed):>6}"
            f"{len(summary.broken):>6}"
            f"{len(summary.fixed) - len(summary.broken):>+6}"
            f"{summary.empty_reads:>6}"
        )

    print()
    print("=== 逐張變化(相對 baseline)===")
    for summary in summaries:
        if not summary.fixed and not summary.broken:
            continue
        print(f"  {summary.variant}")
        if summary.fixed:
            print(f"    從錯變對 ({len(summary.fixed)}): {', '.join(summary.fixed)}")
        if summary.broken:
            print(f"    從對變錯 ({len(summary.broken)}): {', '.join(summary.broken)}")

    print()
    print("=== 車牌寬度分組正確率 ===")
    labels = [label for label, _low, _high in WIDTH_BUCKETS]
    print(f"{'前處理':<22}" + "".join(f"{label:>14}" for label in labels))
    print("-" * (22 + 14 * len(labels)))
    for summary in summaries:
        row = f"{summary.variant:<22}"
        for label in labels:
            if label in summary.by_width:
                correct, total = summary.by_width[label]
                row += f"{f'{correct}/{total} {correct / total * 100:.0f}%':>14}"
            else:
                row += f"{'-':>14}"
        print(row)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="離線比較各種前處理對車牌 OCR 讀對率的影響。"
    )
    parser.add_argument("--crops", type=Path, default=DEFAULT_CROPS)
    parser.add_argument("--pattern", default=DEFAULT_PATTERN)
    parser.add_argument("--truth", type=Path, default=DEFAULT_TRUTH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--out", type=Path, default=Path("~/ocr_sweep"), help="結果輸出目錄"
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="只跑前 N 張(0 = 全部)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if _IMPORT_ERROR is not None:
        print(f"需要 opencv 與 numpy:{_IMPORT_ERROR}", file=sys.stderr)
        return 2

    crops_dir = args.crops.expanduser()
    truth_path = args.truth.expanduser()
    out_dir = args.out.expanduser()

    if not crops_dir.is_dir():
        print(f"找不到裁切圖目錄:{crops_dir}", file=sys.stderr)
        return 2
    if not truth_path.is_file():
        print(
            f"找不到對照表:{truth_path}\n"
            f"格式為兩欄 CSV:filename,truth(truth 留空表示還沒核對)",
            file=sys.stderr,
        )
        return 2

    truth = load_truth(truth_path)
    images = sorted(crops_dir.glob(args.pattern))
    images = [path for path in images if path.name in truth]
    if args.limit:
        images = images[: args.limit]
    if not images:
        print(
            f"{crops_dir} 裡沒有同時出現在對照表中的 {args.pattern}",
            file=sys.stderr,
        )
        return 2

    config: StationConfig = load_config(args.config)
    # 關掉 config 的前處理:那一項本身就是被比較的變體之一,開著的話每個變體
    # 都會再多吃一次 config 指定的前處理,量到的就不是單一變體的效果了。
    # (config 的預設值現在是 unsharp_cubic_3x,所以這一步幾乎一定會觸發。)
    if preprocess.resolve(config.ocr_preprocess) != BASELINE:
        from dataclasses import replace

        _logger.info(
            "config 的 ocr_preprocess=%s,掃描期間強制設為 %s 以免重複前處理",
            config.ocr_preprocess,
            BASELINE,
        )
        config = replace(config, ocr_preprocess=BASELINE)

    try:
        recognizer = make_recognizer(config)
    except OcrBackendUnavailable as exc:
        print(exc, file=sys.stderr)
        return 2
    if not hasattr(recognizer, "read_plate_only"):
        print(
            f"這支腳本需要 fast-plate-ocr 後端,目前是 {config.ocr_backend}",
            file=sys.stderr,
        )
        return 2

    print(f"fast-plate-ocr : {installed_version('fast-plate-ocr')}")
    print(f"模型           : {config.ocr_model_name}")
    print(f"裁切圖         : {crops_dir}({len(images)} 張有答案)")
    print(f"前處理         : {len(PREPROCESSORS)} 種")
    print(f"共 {len(images) * len(PREPROCESSORS)} 次 OCR")
    print("另外每種前處理會多一列 +fmtfix(格式約束校正,純字串後處理,不跑 OCR)")

    attempts = run_sweep(recognizer, images, truth, PREPROCESSORS)
    if not attempts:
        print("沒有任何成功的辨識嘗試", file=sys.stderr)
        return 1

    attempts += derive_format_fix_attempts(attempts)

    summaries = summarise(attempts)
    print_report(summaries)
    fix_summaries = summarise_format_fix(attempts)
    print_format_fix_report(fix_summaries)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = out_dir / f"ocr_sweep_{stamp}_detail.csv"
    summary_path = out_dir / f"ocr_sweep_{stamp}_summary.csv"
    bywidth_path = out_dir / f"ocr_sweep_{stamp}_bywidth.csv"
    fmtfix_path = out_dir / f"ocr_sweep_{stamp}_fmtfix.csv"
    write_detail(detail_path, attempts)
    write_summary(summary_path, summaries)
    write_by_width(bywidth_path, summaries)
    write_format_fix(fmtfix_path, fix_summaries)

    print()
    print("結果已寫入:")
    for path in (summary_path, detail_path, bywidth_path, fmtfix_path):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
