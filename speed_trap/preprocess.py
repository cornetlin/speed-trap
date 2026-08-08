"""OCR 前處理策略 —— 掃描腳本與站台共用的唯一一份實作。

前處理對車牌辨識率的影響很大,而且不能靠推理決定。同一批 46 張車牌裁切圖
的實測結果:什麼都不做讀對 23 張,先 unsharp 再用 cubic 放大三倍讀對 26 張
(字元正確率 84.7% → 89.8%),而 CLAHE 對比增強掉到 13 張 —— 它弄壞了 12 張
原本讀得對的。

因此策略的實作只放這一份:``scripts/ocr_sweep.py`` 掃描的是下面這些函式,
``speed_trap.plate_recognizer`` 線上跑的也是同樣這些函式。兩邊各寫一份的話,
掃描出來的結論套到線上就不成立 —— 先前 CLAHE 留在 production 而掃描證明它
有害,就是這樣來的。

新增策略只要往 :data:`PREPROCESSORS` 加一筆,掃描與線上同時就有。策略名稱
即 config 的 ``ocr_preprocess`` 可填的值。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

_logger = logging.getLogger(__name__)

# cv2 / numpy 跟專案其他地方一樣延後失敗:模組要能 import(config 驗證與
# 純邏輯測試才跑得動),真的要處理影像時才需要它們。
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


# 不做任何處理。名字沿用掃描報表裡的 "baseline",舊的掃描結果才對得起來;
# config 寫 "none" 也接受(見 _ALIASES)。
BASELINE = "baseline"

# 現場實測的贏家。config 沒寫 ocr_preprocess 時用它。
DEFAULT = "unsharp_cubic_3x"

# 前處理後重新編碼的 JPEG 品質。掃描腳本與線上必須一致,否則兩邊量到的
# 不是同一件事。95 是刻意偏高:這張圖只會活到送進 OCR 為止,壓縮痕跡的
# 代價遠大於省下的那幾 KB。
JPEG_QUALITY = 95


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
    """對比受限的自適應直方圖等化。

    留著只為了讓掃描能繼續量它 —— 實測它是所有變體裡最差的一組,不要拿來
    當 production 設定。
    """
    clahe = _cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(_to_gray(image))


def _unsharp(image: Any) -> Any:
    """遮罩銳化。放大之前先做,放大才不會只是把模糊的邊也一起放大。"""
    blurred = _cv2.GaussianBlur(image, (0, 0), sigmaX=1.0)
    return _cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


PREPROCESSORS: dict[str, Callable[[Any], Any]] = {
    # 原圖直送 —— 比較的基準
    BASELINE: lambda img: img,
    # 純放大。車牌普遍只有 40-60 px 寬,OCR 模型的輸入尺寸比這大得多,
    # 放大等於把插值的工作從模型內部搬到我們自己手上。
    "cubic_2x": lambda img: _resize(img, 2, _cv2.INTER_CUBIC),
    "cubic_3x": lambda img: _resize(img, 3, _cv2.INTER_CUBIC),
    "cubic_4x": lambda img: _resize(img, 4, _cv2.INTER_CUBIC),
    "lanczos_3x": lambda img: _resize(img, 3, _cv2.INTER_LANCZOS4),
    # 色彩 / 對比
    "gray": _to_gray,
    "gray_clahe": _clahe,
    "gray_clahe_cubic_3x": lambda img: _resize(_clahe(img), 3, _cv2.INTER_CUBIC),
    # 先銳化再放大 —— 實測最佳
    "unsharp_cubic_3x": lambda img: _resize(_unsharp(img), 3, _cv2.INTER_CUBIC),
}

# config 可以寫得比較自然的別名。掃描報表一律用正式名稱。
_ALIASES = {
    "none": BASELINE,
    "off": BASELINE,
    "unsharp_cubic3x": "unsharp_cubic_3x",
}


def available() -> tuple[str, ...]:
    return tuple(PREPROCESSORS)


def resolve(name: str) -> str:
    """把 config 寫的值對到正式的策略名稱,認不得就丟 ValueError。

    認不得的名稱一定要炸開,不能默默當成不做前處理 —— 現場一次實驗要等
    一個上午的車流,拼錯一個字就白跑,而且事後從 CSV 看不出來。
    """
    key = name.strip().lower()
    if key in PREPROCESSORS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(
        f"認不得的前處理策略 {name!r}。可用值:{', '.join(available())}"
        f"(不做前處理寫 none)"
    )


def apply(name: str, image: Any) -> Any:
    """對已解碼的 BGR 影像套用策略。掃描腳本走這條。"""
    return PREPROCESSORS[resolve(name)](image)


def apply_to_jpeg(name: str, jpeg_bytes: bytes) -> bytes:
    """對 JPEG bytes 套用策略,回傳新的 JPEG bytes。站台走這條。

    解碼失敗或編碼失敗時原封不動回傳 —— 前處理是加分項,不該讓一張壞掉的
    裁切圖連 OCR 都跑不成。

    baseline 直接短路,不做 decode/encode 來回:那趟只會多一次 JPEG 世代
    損失,量不到任何東西。
    """
    strategy = resolve(name)
    if strategy == BASELINE or not jpeg_bytes or _IMPORT_ERROR is not None:
        return jpeg_bytes
    try:
        image = _cv2.imdecode(_np.frombuffer(jpeg_bytes, _np.uint8), _cv2.IMREAD_COLOR)
        if image is None:
            return jpeg_bytes
        processed = PREPROCESSORS[strategy](image)
        ok, encoded = _cv2.imencode(
            ".jpg", processed, [int(_cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        return bytes(encoded) if ok else jpeg_bytes
    except Exception as exc:  # noqa: BLE001 - 前處理壞掉不該讓 OCR 跑不成
        _logger.warning("前處理 %s 失敗,改用原圖:%s: %s", strategy, type(exc).__name__, exc)
        return jpeg_bytes
