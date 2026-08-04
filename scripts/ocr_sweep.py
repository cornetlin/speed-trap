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

輸出三份 CSV(檔名帶時間戳,重跑不會蓋掉舊結果):
    ocr_sweep_<時間戳>_summary.csv   每種前處理一列
    ocr_sweep_<時間戳>_detail.csv    每張圖每種前處理一列
    ocr_sweep_<時間戳>_bywidth.csv   前處理 x 車牌寬度分組

順帶驗證一件事:plate_recognizer._read_array 裡寫了 CUBIC 放大與最小面積
過濾,但從來沒被呼叫過(read_from_jpeg 走的是 _read_path)。這支腳本的
cubic_* 變體等於在測那段死碼值不值得接回去。
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

from speed_trap.config import StationConfig, load_config  # noqa: E402
from speed_trap.plate_format import normalize_plate  # noqa: E402
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
# 每個函式收 BGR ndarray、回傳 ndarray。要加新的前處理就往 PREPROCESSORS
# 裡加一筆,其餘不用動 —— summary CSV 會自動多一列,跟舊結果也還能比較。


def _resize(image: Any, scale: float, interpolation: int) -> Any:
    height, width = image.shape[:2]
    return _cv2.resize(
        image,
        (int(width * scale), int(height * scale)),
        interpolation=interpolation,
    )


def _to_gray(image: Any) -> Any:
    if image.ndim == 3:
        return _cv2.cvtColor(image, _cv2.COLOR_BGR2GRAY)
    return image


def _clahe(image: Any) -> Any:
    """對比受限的自適應直方圖等化。低對比、逆光的車牌靠這個拉回筆畫。"""
    clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(_to_gray(image))


def _unsharp(image: Any) -> Any:
    blurred = _cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return _cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


PREPROCESSORS: dict[str, Callable[[Any], Any]] = {
    # 基準:目前 production 走的路(read_from_jpeg 直接把原圖丟給 OCR)
    "baseline": lambda img: img,
    # 純放大 —— 驗證 _read_array 那段死碼裡的 CUBIC 放大值不值得接回去
    "cubic_2x": lambda img: _resize(img, 2, _cv2.INTER_CUBIC),
    "cubic_3x": lambda img: _resize(img, 3, _cv2.INTER_CUBIC),
    "cubic_4x": lambda img: _resize(img, 4, _cv2.INTER_CUBIC),
    "lanczos_3x": lambda img: _resize(img, 3, _cv2.INTER_LANCZOS4),
    # 色彩 / 對比
    "gray": _to_gray,
    "gray_clahe": _clahe,
    "gray_clahe_cubic_3x": lambda img: _resize(
        _clahe(img), 3, _cv2.INTER_CUBIC
    ),
    # 先銳化再放大
    "unsharp_cubic_3x": lambda img: _resize(
        _unsharp(img), 3, _cv2.INTER_CUBIC
    ),
}


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
    attempts: list[Attempt], baseline: str = "baseline"
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
    # 關掉 config 的前處理:那一項本身就是被比較的變體之一,開著會讓每個
    # 變體都額外吃一次 CLAHE+unsharp,實驗就不乾淨了。
    if config.ocr_preprocess:
        from dataclasses import replace

        _logger.info("config 的 ocr_preprocess=True,掃描期間強制關閉以免重複前處理")
        config = replace(config, ocr_preprocess=False)

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

    attempts = run_sweep(recognizer, images, truth, PREPROCESSORS)
    if not attempts:
        print("沒有任何成功的辨識嘗試", file=sys.stderr)
        return 1

    summaries = summarise(attempts)
    print_report(summaries)

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_path = out_dir / f"ocr_sweep_{stamp}_detail.csv"
    summary_path = out_dir / f"ocr_sweep_{stamp}_summary.csv"
    bywidth_path = out_dir / f"ocr_sweep_{stamp}_bywidth.csv"
    write_detail(detail_path, attempts)
    write_summary(summary_path, summaries)
    write_by_width(bywidth_path, summaries)

    print()
    print("結果已寫入:")
    for path in (summary_path, detail_path, bywidth_path):
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
