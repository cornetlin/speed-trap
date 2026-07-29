"""fast-plate-ocr 建構子相容性,以及初始化失敗要中止而不是靜默降級。

背景:在 Pi 上跑 pip install -e . 之後 fast-plate-ocr 自動升級,
ONNXPlateRecognizer 改名為 LicensePlateRecognizer 且建構子參數換掉。舊的
載入邏輯硬寫參數名再一個個 try,認不得新版就丟出 RuntimeError,而
make_recognizer 把它吞成一則 WARNING 後繼續跑 —— 整場實驗產出一份完全
沒有車牌的 CSV,跑完才發現。
"""

from __future__ import annotations

from typing import Any

import pytest
from tests.helpers import make_config

import speed_trap.plate_recognizer as pr
from speed_trap.plate_recognizer import (
    NoopPlateRecognizer,
    OcrBackendUnavailable,
)

_ONNX = "/home/kevin30/speed-trap/models/best.onnx"
_CONFIG = "/home/kevin30/speed-trap/models/plate_config.yaml"
_HUB = "global-plates-mobile-vit-v2-model"


# --- 三種真實出現過的建構子簽章 ---------------------------------------


class OldONNXPlateRecognizer:
    """舊版:單一類別同時吃 hub 名稱與自訓 ONNX。"""

    def __init__(
        self,
        model_name: str,
        config_file: str | None = None,
        onnx_model_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.config_file = config_file
        self.onnx_model_path = onnx_model_path


class HubOnlyLicensePlateRecognizer:
    """改名後、只吃 hub 名稱的版本。"""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device


class PathStyleLicensePlateRecognizer:
    """改名後、自訓 ONNX 用專屬參數的版本。"""

    def __init__(
        self,
        onnx_model_path: str,
        plate_config_path: str,
        device: str = "auto",
    ) -> None:
        self.onnx_model_path = onnx_model_path
        self.plate_config_path = plate_config_path
        self.device = device


class RenamedParamsRecognizer:
    """假想的未來版本:參數名又換了,但仍在我們認得的候選清單內。"""

    def __init__(self, model_path: str, model_config_path: str) -> None:
        self.model_path = model_path
        self.model_config_path = model_config_path


class UnknownSignatureRecognizer:
    """完全認不得的簽章 —— 應該給出說明清楚的錯誤,不是硬試。"""

    def __init__(self, weights_blob: bytes, magic: int) -> None:
        self.weights_blob = weights_blob
        self.magic = magic


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch):
    """把某個假類別當成「已安裝的 fast-plate-ocr」。"""

    def _install(target: type) -> None:
        monkeypatch.setattr(pr, "_LicensePlateRecognizer", target)
        monkeypatch.setattr(pr, "_IMPORT_ERROR", None)
        monkeypatch.setattr(pr, "installed_version", lambda _pkg: "9.9.9-test")

    return _install


# --- 【1】三種簽章都要能初始化 ------------------------------------------


def test_old_onnx_style_custom_model(installed: Any) -> None:
    installed(OldONNXPlateRecognizer)
    loaded = pr._load_lpr(_ONNX, _CONFIG)

    assert isinstance(loaded, OldONNXPlateRecognizer)
    assert loaded.onnx_model_path == _ONNX
    assert loaded.config_file == _CONFIG


def test_old_onnx_style_hub_model(installed: Any) -> None:
    installed(OldONNXPlateRecognizer)
    loaded = pr._load_lpr(_HUB, None)

    assert loaded.model_name == _HUB
    assert loaded.onnx_model_path is None


def test_renamed_class_hub_only(installed: Any) -> None:
    installed(HubOnlyLicensePlateRecognizer)
    loaded = pr._load_lpr(_HUB, None)

    assert isinstance(loaded, HubOnlyLicensePlateRecognizer)
    assert loaded.model_name == _HUB


def test_renamed_class_path_style(installed: Any) -> None:
    installed(PathStyleLicensePlateRecognizer)
    loaded = pr._load_lpr(_ONNX, _CONFIG)

    assert isinstance(loaded, PathStyleLicensePlateRecognizer)
    assert loaded.onnx_model_path == _ONNX
    assert loaded.plate_config_path == _CONFIG


def test_future_parameter_names_still_resolve(installed: Any) -> None:
    """只要新版沿用候選清單裡的名稱就自動相容,不用改程式。"""
    installed(RenamedParamsRecognizer)
    loaded = pr._load_lpr(_ONNX, _CONFIG)

    assert loaded.model_path == _ONNX
    assert loaded.model_config_path == _CONFIG


def test_hub_model_rejected_when_signature_has_no_name_param(
    installed: Any,
) -> None:
    """簽章只吃路徑時,hub 名稱無處可放 —— 要明講,不要硬塞。"""
    installed(PathStyleLicensePlateRecognizer)
    with pytest.raises(OcrBackendUnavailable) as excinfo:
        pr._load_lpr(_HUB, None)

    assert "hub" in str(excinfo.value)


def test_unknown_signature_reports_actual_parameters(installed: Any) -> None:
    installed(UnknownSignatureRecognizer)
    with pytest.raises(OcrBackendUnavailable) as excinfo:
        pr._load_lpr(_ONNX, _CONFIG)

    message = str(excinfo.value)
    # 錯誤訊息要說出實際看到的參數,才有辦法對照上游改了什麼
    assert "weights_blob" in message
    assert "magic" in message


def test_required_config_param_without_config_is_reported(
    installed: Any,
) -> None:
    installed(PathStyleLicensePlateRecognizer)
    with pytest.raises(OcrBackendUnavailable) as excinfo:
        pr._load_lpr(_ONNX, None)

    assert "plate_config_path" in str(excinfo.value)


def test_constructor_failure_is_wrapped_with_version(installed: Any) -> None:
    class ExplodingRecognizer:
        def __init__(self, model_name: str) -> None:
            raise OSError("model file corrupt")

    installed(ExplodingRecognizer)
    with pytest.raises(OcrBackendUnavailable) as excinfo:
        pr._load_lpr(_HUB, None)

    message = str(excinfo.value)
    assert "9.9.9-test" in message
    assert "model file corrupt" in message


def test_signature_and_version_are_logged(
    installed: Any, caplog: pytest.LogCaptureFixture
) -> None:
    installed(PathStyleLicensePlateRecognizer)
    with caplog.at_level("INFO", logger="speed_trap.plate_recognizer"):
        pr._load_lpr(_ONNX, _CONFIG)

    logged = caplog.text
    assert "9.9.9-test" in logged
    assert "PathStyleLicensePlateRecognizer" in logged
    assert "onnx_model_path" in logged


def test_installed_version_handles_missing_package() -> None:
    assert pr.installed_version("definitely-not-a-real-package") == "unknown"


# --- 【2】初始化失敗要中止 ----------------------------------------------


def test_noop_backend_is_allowed_explicitly() -> None:
    recognizer = pr.make_recognizer(make_config(ocr_backend="noop"))
    assert isinstance(recognizer, NoopPlateRecognizer)


def test_failed_backend_raises_instead_of_returning_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """這是本次最重要的行為:寧可開不起來,也不要安靜地不做辨識。"""

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("fast-plate-ocr dependencies missing")

    monkeypatch.setattr(pr, "PlateRecognizer", explode)

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        pr.make_recognizer(make_config(ocr_backend="fast-plate-ocr"))

    message = str(excinfo.value)
    assert "fast-plate-ocr" in message          # 哪個後端
    assert "dependencies missing" in message    # 什麼錯誤
    assert "ocr_backend: noop" in message       # 怎麼繞過


def test_failure_message_includes_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr, "installed_version", lambda _pkg: "1.2.3-test")

    def explode(*args: object, **kwargs: object) -> None:
        raise TypeError("unexpected keyword argument 'model_name'")

    monkeypatch.setattr(pr, "PlateRecognizer", explode)

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        pr.make_recognizer(make_config(ocr_backend="fast-plate-ocr"))

    assert "1.2.3-test" in str(excinfo.value)


def test_paddle_backend_failure_also_raises() -> None:
    try:
        from speed_trap.paddle_recognizer import _IMPORT_ERROR as paddle_error
    except ImportError:
        pytest.skip("paddle_recognizer 無法 import")
    if paddle_error is None:
        pytest.skip("這台機器裝了 paddleocr,測不到失敗路徑")

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        pr.make_recognizer(make_config(ocr_backend="paddleocr"))

    assert "paddleocr" in str(excinfo.value)


def test_unknown_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    fake_config = SimpleNamespace(ocr_backend="surprise")
    with pytest.raises(OcrBackendUnavailable):
        pr.make_recognizer(fake_config)  # type: ignore[arg-type]
