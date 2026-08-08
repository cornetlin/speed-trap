"""前處理策略登錄表。

這個模組是掃描腳本與 station 的共用實作 —— 它一旦跟掃描時量的東西不一樣,
掃描出來的結論就不適用於線上,而那種錯誤在現場看不出來(CSV 上只會看到
正確率比預期低)。所以名稱、別名與「baseline 不動原圖」這幾件事要釘住。
"""

from __future__ import annotations

import pytest

from speed_trap import preprocess

_needs_cv2 = pytest.mark.skipif(
    preprocess._IMPORT_ERROR is not None,
    reason=f"需要 opencv/numpy:{preprocess._IMPORT_ERROR}",
)


# --- 登錄表與名稱 -------------------------------------------------------


def test_registry_has_the_variants_the_sweep_reports_on() -> None:
    """掃描報表的每一列都對應這裡的一個鍵值,名稱改了舊結果就對不起來。"""
    for name in (
        "baseline",
        "cubic_2x",
        "cubic_3x",
        "cubic_4x",
        "lanczos_3x",
        "gray",
        "gray_clahe",
        "gray_clahe_cubic_3x",
        "unsharp_cubic_3x",
    ):
        assert name in preprocess.PREPROCESSORS, name
    assert all(callable(fn) for fn in preprocess.PREPROCESSORS.values())


def test_default_is_the_measured_winner() -> None:
    """46 張實測 26/46、字元正確率 89.8%,勝過不做前處理的 23/46。"""
    assert preprocess.DEFAULT == "unsharp_cubic_3x"
    assert preprocess.DEFAULT in preprocess.PREPROCESSORS


def test_resolve_accepts_canonical_names() -> None:
    assert preprocess.resolve("cubic_3x") == "cubic_3x"
    assert preprocess.resolve("baseline") == preprocess.BASELINE


def test_resolve_accepts_aliases_and_sloppy_case() -> None:
    assert preprocess.resolve("none") == preprocess.BASELINE
    assert preprocess.resolve("off") == preprocess.BASELINE
    assert preprocess.resolve(" NONE ") == preprocess.BASELINE
    # config 寫少一個底線也還是收 —— 現場拼錯字的代價是白跑一個上午
    assert preprocess.resolve("unsharp_cubic3x") == "unsharp_cubic_3x"


def test_resolve_rejects_unknown_names_with_the_available_list() -> None:
    with pytest.raises(ValueError) as excinfo:
        preprocess.resolve("clahe_but_spelled_wrong")
    message = str(excinfo.value)
    assert "clahe_but_spelled_wrong" in message
    # 錯誤訊息要能直接照抄一個可用值出來
    assert "unsharp_cubic_3x" in message


def test_available_lists_every_registered_strategy() -> None:
    assert set(preprocess.available()) == set(preprocess.PREPROCESSORS)


# --- 實際處理影像 -------------------------------------------------------


def _sample_jpeg(width: int = 60, height: int = 30) -> bytes:
    """一張有梯度與方塊的小圖,尺寸接近實際車牌裁切圖(40-60 px 寬)。"""
    import cv2
    import numpy as np

    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = np.linspace(0, 255, width, dtype=np.uint8)[None, :, None]
    cv2.rectangle(image, (10, 8), (24, 22), (255, 255, 255), -1)
    ok, buffer = cv2.imencode(".jpg", image)
    assert ok
    return bytes(buffer)


@_needs_cv2
def test_baseline_returns_the_original_bytes_untouched() -> None:
    """baseline 不做 decode/encode 來回 —— 那趟只會多一次 JPEG 世代損失。"""
    jpeg = _sample_jpeg()
    assert preprocess.apply_to_jpeg("baseline", jpeg) is jpeg
    assert preprocess.apply_to_jpeg("none", jpeg) is jpeg


@_needs_cv2
def test_cubic_3x_triples_both_dimensions() -> None:
    import cv2
    import numpy as np

    out = preprocess.apply_to_jpeg("cubic_3x", _sample_jpeg(60, 30))
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (90, 180)


@_needs_cv2
def test_unsharp_cubic_3x_also_triples_and_changes_pixels() -> None:
    """放大之外還要真的銳化過 —— 只放大的話它就跟 cubic_3x 是同一個變體。"""
    import cv2
    import numpy as np

    jpeg = _sample_jpeg(60, 30)
    sharpened = preprocess.apply_to_jpeg("unsharp_cubic_3x", jpeg)
    plain = preprocess.apply_to_jpeg("cubic_3x", jpeg)

    decoded = cv2.imdecode(np.frombuffer(sharpened, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (90, 180)
    assert sharpened != plain


@_needs_cv2
def test_gray_strategies_produce_single_channel() -> None:
    import cv2
    import numpy as np

    out = preprocess.apply_to_jpeg("gray", _sample_jpeg())
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_UNCHANGED)
    assert decoded.ndim == 2


def test_empty_input_is_returned_unchanged() -> None:
    assert preprocess.apply_to_jpeg("unsharp_cubic_3x", b"") == b""


@_needs_cv2
def test_corrupt_jpeg_is_returned_unchanged_not_raised() -> None:
    """壞掉的裁切圖不該讓 OCR 連跑都跑不成 —— 前處理是加分項。"""
    assert preprocess.apply_to_jpeg("cubic_3x", b"not a jpeg") == b"not a jpeg"


def test_apply_to_jpeg_rejects_unknown_strategy() -> None:
    with pytest.raises(ValueError):
        preprocess.apply_to_jpeg("no_such_strategy", b"whatever")
