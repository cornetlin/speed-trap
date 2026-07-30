"""fast-plate-ocr 建構子相容性,以及初始化失敗要中止而不是靜默降級。

背景:在 Pi 上跑 pip install -e . 之後 fast-plate-ocr 自動升級到 1.1.0,
ONNXPlateRecognizer 改名為 LicensePlateRecognizer 且建構子參數換掉。舊的
載入邏輯硬寫參數名再一個個 try,認不得新版就丟出 RuntimeError,而
make_recognizer 把它吞成一則 WARNING 後繼續跑 —— 整場實驗產出一份完全
沒有車牌的 CSV,跑完才發現。

1.1.0 的實際簽章(已在 Pi 上核對):
    LicensePlateRecognizer(
        hub_ocr_model=None,      # 只接受預訓練清單裡的名稱
        device='auto',
        providers=None, sess_options=None,
        onnx_model_path=None,
        plate_config_path=None,
        force_download=False,
    )
套件內部是 ``if onnx_model_path and plate_config_path: ... elif hub_ocr_model:``
—— 自訂 ONNX 與 hub 模型是互斥的兩條路,而且把檔案路徑塞進 hub_ocr_model
不會被當成路徑。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from tests.helpers import make_config

import speed_trap.plate_recognizer as pr
from speed_trap.plate_recognizer import (
    NoopPlateRecognizer,
    OcrBackendUnavailable,
)

_HUB = "global-plates-mobile-vit-v2-model"


# --- 各版本的建構子簽章 -------------------------------------------------


class LicensePlateRecognizer110:
    """fast-plate-ocr 1.1.0 —— Pi 上實際安裝的這一版。"""

    def __init__(
        self,
        hub_ocr_model: str | None = None,
        device: str = "auto",
        providers: Any = None,
        sess_options: Any = None,
        onnx_model_path: str | None = None,
        plate_config_path: str | None = None,
        force_download: bool = False,
    ) -> None:
        self.hub_ocr_model = hub_ocr_model
        self.device = device
        self.onnx_model_path = onnx_model_path
        self.plate_config_path = plate_config_path


class OldONNXPlateRecognizer:
    """改名前的類別,參數名不同但語意相同。"""

    def __init__(
        self,
        model_name: str | None = None,
        config_file: str | None = None,
        onnx_model_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.config_file = config_file
        self.onnx_model_path = onnx_model_path


class HubOnlyRecognizer:
    """只吃 hub 名稱、沒有自訂 ONNX 參數的版本。"""

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = model_name
        self.device = device


class RenamedParamsRecognizer:
    """假想的未來版本:參數名又換了,但仍在候選清單內。"""

    def __init__(
        self,
        model_path: str | None = None,
        model_config_path: str | None = None,
    ) -> None:
        self.model_path = model_path
        self.model_config_path = model_config_path


class RequiredHubRecognizer:
    """hub 參數被列為必填 —— 無法只用自訂 ONNX 初始化。"""

    def __init__(
        self,
        hub_ocr_model: str,
        onnx_model_path: str | None = None,
        plate_config_path: str | None = None,
    ) -> None:
        self.hub_ocr_model = hub_ocr_model


class UnknownSignatureRecognizer:
    """完全認不得的簽章 —— 要給出說明清楚的錯誤,不是硬試。"""

    def __init__(self, weights_blob: bytes, magic: int) -> None:
        self.weights_blob = weights_blob
        self.magic = magic


@pytest.fixture
def installed(monkeypatch: pytest.MonkeyPatch):
    """把某個假類別當成「已安裝的 fast-plate-ocr」。"""

    def _install(target: type) -> None:
        monkeypatch.setattr(pr, "_LicensePlateRecognizer", target)
        monkeypatch.setattr(pr, "_IMPORT_ERROR", None)
        monkeypatch.setattr(pr, "installed_version", lambda _pkg: "1.1.0")

    return _install


@pytest.fixture
def model_files(tmp_path: Path) -> tuple[str, str]:
    """真的建出 .onnx 與 plate_config.yaml —— 現在會檢查檔案存在。"""
    onnx = tmp_path / "best.onnx"
    config = tmp_path / "plate_config.yaml"
    onnx.write_bytes(b"fake onnx")
    config.write_text("max_plate_slots: 7\n", encoding="utf-8")
    return str(onnx), str(config)


def _load(model: str, config: str | None) -> Any:
    instance, _info = pr._load_lpr(model, config)
    return instance


# --- 【1】依「值的內容」決定走哪條路,兩條路互斥 ------------------------


def test_file_path_goes_to_onnx_param_only(
    installed: Any, model_files: tuple[str, str]
) -> None:
    """核心行為:ocr_model_name 是檔案路徑時只填 onnx_model_path。

    hub_ocr_model 只接受預訓練清單裡的名稱,把路徑塞進去不會被當成路徑,
    只會變成一個不存在的模型名稱。
    """
    onnx, config = model_files
    installed(LicensePlateRecognizer110)

    loaded = _load(onnx, config)

    assert loaded.onnx_model_path == onnx
    assert loaded.plate_config_path == config
    assert loaded.hub_ocr_model is None      # 絕不同時填


def test_hub_name_goes_to_hub_param_only(installed: Any) -> None:
    installed(LicensePlateRecognizer110)

    loaded = _load(_HUB, None)

    assert loaded.hub_ocr_model == _HUB
    assert loaded.onnx_model_path is None
    assert loaded.plate_config_path is None


def test_planned_kwargs_are_mutually_exclusive(
    installed: Any, model_files: tuple[str, str]
) -> None:
    onnx, config = model_files
    installed(LicensePlateRecognizer110)

    _instance, onnx_info = pr._load_lpr(onnx, config)
    _instance2, hub_info = pr._load_lpr(_HUB, None)

    assert "hub_ocr_model" not in onnx_info.kwargs
    assert onnx_info.mode == "自訂 ONNX"
    assert "onnx_model_path" not in hub_info.kwargs
    assert "plate_config_path" not in hub_info.kwargs
    assert hub_info.mode == "hub 模型"


def test_old_class_name_still_works(
    installed: Any, model_files: tuple[str, str]
) -> None:
    onnx, config = model_files
    installed(OldONNXPlateRecognizer)

    loaded = _load(onnx, config)

    assert loaded.onnx_model_path == onnx
    assert loaded.config_file == config
    assert loaded.model_name is None


def test_future_parameter_names_still_resolve(
    installed: Any, model_files: tuple[str, str]
) -> None:
    """只要新版沿用候選清單裡的名稱就自動相容,不用改程式。"""
    onnx, config = model_files
    installed(RenamedParamsRecognizer)

    loaded = _load(onnx, config)

    assert loaded.model_path == onnx
    assert loaded.model_config_path == config


def test_hub_model_rejected_when_signature_has_no_name_param(
    installed: Any,
) -> None:
    installed(RenamedParamsRecognizer)
    with pytest.raises(OcrBackendUnavailable) as excinfo:
        _load(_HUB, None)

    assert "hub" in str(excinfo.value)


def test_custom_onnx_rejected_when_hub_param_is_required(
    installed: Any, model_files: tuple[str, str]
) -> None:
    """寧可講清楚,也不要把路徑硬塞進只吃模型名稱的參數。"""
    onnx, config = model_files
    installed(RequiredHubRecognizer)

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        _load(onnx, config)

    message = str(excinfo.value)
    assert "hub_ocr_model" in message
    assert "必填" in message


def test_unknown_signature_reports_actual_parameters(
    installed: Any, model_files: tuple[str, str]
) -> None:
    onnx, config = model_files
    installed(UnknownSignatureRecognizer)

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        _load(onnx, config)

    message = str(excinfo.value)
    assert "weights_blob" in message


def test_hub_only_signature_rejects_a_file_path(
    installed: Any, model_files: tuple[str, str]
) -> None:
    onnx, config = model_files
    installed(HubOnlyRecognizer)

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        _load(onnx, config)

    assert "ONNX 路徑參數" in str(excinfo.value)


# --- 【4】檔案不存在時要指出是哪一個 -----------------------------------


def test_missing_onnx_names_the_model_file(
    installed: Any, tmp_path: Path
) -> None:
    installed(LicensePlateRecognizer110)
    config = tmp_path / "plate_config.yaml"
    config.write_text("x: 1\n", encoding="utf-8")
    missing_onnx = str(tmp_path / "nope.onnx")

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        _load(missing_onnx, str(config))

    message = str(excinfo.value)
    assert "ONNX 模型" in message
    assert "nope.onnx" in message
    # 不要只轉傳套件那句看不出是哪個檔案的訊息
    assert "Missing model/config file!" not in message.split("(")[0]


def test_missing_config_names_the_config_file(
    installed: Any, tmp_path: Path
) -> None:
    installed(LicensePlateRecognizer110)
    onnx = tmp_path / "best.onnx"
    onnx.write_bytes(b"fake")
    missing_config = str(tmp_path / "nope.yaml")

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        _load(str(onnx), missing_config)

    message = str(excinfo.value)
    assert "模型設定檔" in message
    assert "nope.yaml" in message


def test_custom_onnx_without_config_is_reported(
    installed: Any, model_files: tuple[str, str]
) -> None:
    onnx, _config = model_files
    installed(LicensePlateRecognizer110)

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        _load(onnx, None)

    assert "ocr_model_config" in str(excinfo.value)


def test_directory_instead_of_file_is_reported(
    installed: Any, tmp_path: Path
) -> None:
    installed(LicensePlateRecognizer110)
    directory = tmp_path / "models.onnx"
    directory.mkdir()
    config = tmp_path / "plate_config.yaml"
    config.write_text("x: 1\n", encoding="utf-8")

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        _load(str(directory), str(config))

    assert "不是檔案" in str(excinfo.value)


# --- 【2】device 一律 cpu ------------------------------------------------


def test_device_is_always_cpu_for_custom_onnx(
    installed: Any, model_files: tuple[str, str]
) -> None:
    """Pi 沒有 CUDA。留 'auto' 會在每次初始化印兩行 GPU 偵測失敗的警告。"""
    onnx, config = model_files
    installed(LicensePlateRecognizer110)

    loaded = _load(onnx, config)

    assert loaded.device == "cpu"


def test_device_is_always_cpu_for_hub_model(installed: Any) -> None:
    installed(LicensePlateRecognizer110)
    assert _load(_HUB, None).device == "cpu"


def test_device_omitted_when_signature_has_no_such_param(
    installed: Any, model_files: tuple[str, str]
) -> None:
    onnx, config = model_files
    installed(RenamedParamsRecognizer)

    _instance, info = pr._load_lpr(onnx, config)

    assert "device" not in info.kwargs


# --- 錯誤處理與 log ------------------------------------------------------


def test_constructor_failure_is_wrapped_with_version(installed: Any) -> None:
    class ExplodingRecognizer:
        def __init__(self, hub_ocr_model: str | None = None) -> None:
            raise OSError("model file corrupt")

    installed(ExplodingRecognizer)
    with pytest.raises(OcrBackendUnavailable) as excinfo:
        _load(_HUB, None)

    message = str(excinfo.value)
    assert "1.1.0" in message
    assert "model file corrupt" in message


def test_signature_and_version_are_logged(
    installed: Any,
    model_files: tuple[str, str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    onnx, config = model_files
    installed(LicensePlateRecognizer110)

    with caplog.at_level("INFO", logger="speed_trap.plate_recognizer"):
        _load(onnx, config)

    logged = caplog.text
    assert "1.1.0" in logged
    assert "LicensePlateRecognizer110" in logged
    assert "onnx_model_path" in logged
    assert "自訂 ONNX" in logged


def test_hub_model_with_stray_config_warns(
    installed: Any, caplog: pytest.LogCaptureFixture
) -> None:
    installed(LicensePlateRecognizer110)
    with caplog.at_level("WARNING", logger="speed_trap.plate_recognizer"):
        _load(_HUB, "/some/plate_config.yaml")

    assert "會被忽略" in caplog.text


def test_installed_version_handles_missing_package() -> None:
    assert pr.installed_version("definitely-not-a-real-package") == "unknown"


def test_backend_info_describe_is_readable(
    installed: Any, model_files: tuple[str, str]
) -> None:
    onnx, config = model_files
    installed(LicensePlateRecognizer110)

    _instance, info = pr._load_lpr(onnx, config)
    described = info.describe()

    assert described.startswith("LicensePlateRecognizer110(")
    assert "onnx_model_path=" in described
    assert "device='cpu'" in described


# --- 【2 之前】初始化失敗要中止 ----------------------------------------


def test_noop_backend_is_allowed_explicitly() -> None:
    recognizer = pr.make_recognizer(make_config(ocr_backend="noop"))
    assert isinstance(recognizer, NoopPlateRecognizer)


def test_failed_backend_raises_instead_of_returning_noop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """本次最重要的行為:寧可開不起來,也不要安靜地不做辨識。"""

    def explode(*args: object, **kwargs: object) -> None:
        raise RuntimeError("fast-plate-ocr dependencies missing")

    monkeypatch.setattr(pr, "PlateRecognizer", explode)

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        pr.make_recognizer(make_config(ocr_backend="fast-plate-ocr"))

    message = str(excinfo.value)
    assert "fast-plate-ocr" in message
    assert "dependencies missing" in message
    assert "ocr_backend: noop" in message


def test_failure_message_includes_installed_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pr, "installed_version", lambda _pkg: "1.1.0")

    def explode(*args: object, **kwargs: object) -> None:
        raise TypeError("unexpected keyword argument 'model_name'")

    monkeypatch.setattr(pr, "PlateRecognizer", explode)

    with pytest.raises(OcrBackendUnavailable) as excinfo:
        pr.make_recognizer(make_config(ocr_backend="fast-plate-ocr"))

    assert "1.1.0" in str(excinfo.value)


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


def test_unknown_backend_raises() -> None:
    from types import SimpleNamespace

    fake_config = SimpleNamespace(ocr_backend="surprise")
    with pytest.raises(OcrBackendUnavailable):
        pr.make_recognizer(fake_config)  # type: ignore[arg-type]
