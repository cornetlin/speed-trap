#!/usr/bin/env python3
"""拿既有的裁切圖直接驗證 OCR 讀得對不對,不用等現場有車。

「初始化成功」不等於「讀得對」—— 站台起得來、log 沒有錯誤、CSV 卻整片
空白,這種情形已經發生過。這支腳本把中間的每一步都攤開來看。

用法:
    python scripts/check_ocr.py <裁切圖路徑> [--config config/station_a.yaml]
    python scripts/check_ocr.py ~/speedtrap_debug/00001_plate.jpg --plate-only
    python scripts/check_ocr.py ~/speedtrap_crops/*.jpg          # 多張一起

除錯圖有兩種型態:
    *_vehicle.jpg / speedtrap_crops/*.jpg   整輛車 → 走完整 cascade(預設)
    *_plate.jpg                             已切好的車牌 → 加 --plate-only

已經切好的車牌再送一次車牌偵測常常找不到東西,那是輸入型態不對,不是 OCR
讀不出來 —— 所以兩種要分開驗。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# 讓沒有 pip install -e . 的環境也能直接跑。
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from speed_trap.config import StationConfig, load_config  # noqa: E402
from speed_trap.plate_recognizer import (  # noqa: E402
    OcrAttempt,
    OcrBackendUnavailable,
    installed_version,
    make_recognizer,
)

_DEFAULT_CONFIG = _REPO_ROOT / "config" / "station_a.yaml"


def _fmt_wh(wh: tuple[int, int] | None) -> str:
    return f"{wh[0]}x{wh[1]} px" if wh else "(量不到)"


def _describe_backend(recognizer: object, config: StationConfig) -> None:
    print("=== OCR 後端 ===")
    print(f"  config ocr_backend : {config.ocr_backend}")
    print(f"  recognizer 類別    : {type(recognizer).__name__}")
    print(f"  model_name         : {getattr(recognizer, 'model_name', '?')}")

    info = getattr(recognizer, "backend_info", None)
    if info is None:
        # noop 或 PaddleOCR:沒有 fast-plate-ocr 的簽章資訊
        print("  (此後端沒有 fast-plate-ocr 簽章資訊)")
    else:
        print(f"  fast-plate-ocr     : {info.version}")
        print(f"  實際類別           : {info.class_name}")
        print(f"  載入模式           : {info.mode}")
        print("  建構子參數         :")
        for key, value in info.kwargs.items():
            print(f"      {key} = {value!r}")

    has_detector = getattr(recognizer, "has_plate_detector", False)
    print(f"  車牌偵測器(cascade): {'有' if has_detector else '無'}")
    if has_detector:
        print(f"      {config.ocr_plate_detector_path}")
    print(f"  ultralytics        : {installed_version('ultralytics')}")
    print(f"  opencv             : {installed_version('opencv-python-headless')}")


def _report(path: Path, attempt: OcrAttempt, *, plate_only: bool) -> bool:
    """印出一張圖的結果,回傳「是否讀出通過台灣格式的車牌」。"""
    print(f"--- {path.name}")
    print(f"  檔案大小       : {path.stat().st_size:,} bytes")
    print(f"  輸入圖尺寸     : {_fmt_wh(attempt.vehicle_wh)}")
    if not plate_only:
        print(f"  車牌框尺寸     : {_fmt_wh(attempt.plate_wh)}")
        if attempt.plate_wh:
            width = attempt.plate_wh[0]
            verdict = "夠寬" if width >= 100 else "偏窄,OCR 開始靠猜"
            print(f"  車牌像素寬     : {width} px({verdict})")

    reading = attempt.reading
    if reading is None:
        print(f"  結果           : 讀不出來(failure={attempt.failure})")
        return False

    print(f"  OCR 原始字串   : {reading.raw_text!r}")
    print(f"  正規化後       : {reading.text!r}")
    print(f"  信心度         : {reading.confidence:.4f}")
    print(f"  台灣車牌格式   : {'通過' if reading.is_taiwan_format else '不通過'}")
    return reading.is_taiwan_format


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="拿既有裁切圖驗證 OCR,不需要相機或現場車流。"
    )
    parser.add_argument(
        "images", nargs="+", type=Path, help="裁切圖路徑(可給多張)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help=f"站台 YAML(預設 {_DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--plate-only",
        action="store_true",
        help="輸入已經是切好的車牌,跳過車牌偵測那一段直接餵 OCR",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="連 DEBUG log 一起印"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s %(message)s",
    )

    config = load_config(args.config)
    print(f"config: {args.config}")
    print()

    try:
        recognizer = make_recognizer(config)
    except OcrBackendUnavailable as exc:
        print("OCR 後端起不來:", file=sys.stderr)
        print(exc, file=sys.stderr)
        return 2

    _describe_backend(recognizer, config)
    print()

    if args.plate_only and not hasattr(recognizer, "read_plate_only"):
        print(
            "--plate-only 只在 fast-plate-ocr 後端可用,改走完整 cascade",
            file=sys.stderr,
        )
        args.plate_only = False

    print("=== 逐張辨識 ===")
    passed = 0
    total = 0
    for image_path in args.images:
        resolved = image_path.expanduser()
        if not resolved.is_file():
            print(f"--- {image_path}\n  跳過:找不到檔案")
            continue
        total += 1
        payload = resolved.read_bytes()
        if args.plate_only:
            attempt = recognizer.read_plate_only(payload)
        else:
            attempt = recognizer.read_attempt(payload)
        if _report(resolved, attempt, plate_only=args.plate_only):
            passed += 1
        print()

    if total == 0:
        print("沒有可讀的圖檔", file=sys.stderr)
        return 1

    print(f"=== 總計 {passed}/{total} 張讀出符合台灣車牌格式的字串 ===")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
