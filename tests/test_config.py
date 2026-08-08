from __future__ import annotations

from pathlib import Path

import pytest

from speed_trap.config import StationConfig, load_config


def _write_yaml(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "station.yaml"
    path.write_text(body, encoding="utf-8")
    return path


def test_load_config_full(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        """
        station_id: station_a
        camera_source: /dev/video0
        frame_width: 1920
        frame_height: 1080
        frame_fps: 30
        hef_path: models/yolov8n.hef
        hailofilter_so_path: /usr/lib/hailo-post-processes/libyolo_hailortpp_postprocess.so
        vehicle_classes:
          - car
          - truck
        trigger_line_y: 0.7
        mqtt_broker: mqtt.local
        mqtt_topic: speedtrap/a
        log_level: DEBUG
        """,
    )

    cfg = load_config(yaml_path)

    assert cfg.station_id == "station_a"
    assert cfg.camera_source == "/dev/video0"
    assert cfg.frame_width == 1920
    assert cfg.frame_height == 1080
    assert cfg.frame_fps == 30
    assert cfg.hef_path == Path("models/yolov8n.hef")
    assert cfg.hailofilter_so_path == Path(
        "/usr/lib/hailo-post-processes/libyolo_hailortpp_postprocess.so"
    )
    assert cfg.vehicle_classes == ("car", "truck")
    assert cfg.trigger_line_y == pytest.approx(0.7)
    assert cfg.mqtt_broker == "mqtt.local"
    assert cfg.mqtt_topic == "speedtrap/a"
    assert cfg.log_level == "DEBUG"


def test_load_config_optional_fields_default(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        """
        station_id: station_b
        camera_source: 0
        frame_width: 640
        frame_height: 480
        frame_fps: 15
        hef_path: m.hef
        hailofilter_so_path: /tmp/dummy.so
        vehicle_classes: [car]
        trigger_line_y: 0.5
        mqtt_topic: speedtrap/b
        """,
    )

    cfg = load_config(yaml_path)

    assert cfg.mqtt_broker is None
    assert cfg.log_level == "INFO"
    assert cfg.camera_source == "0"


def test_load_config_explicit_null_broker(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        """
        station_id: x
        camera_source: cam
        frame_width: 320
        frame_height: 240
        frame_fps: 10
        hef_path: m.hef
        hailofilter_so_path: /tmp/dummy.so
        vehicle_classes: [car]
        trigger_line_y: 0.0
        mqtt_broker: null
        mqtt_topic: t
        """,
    )

    cfg = load_config(yaml_path)
    assert cfg.mqtt_broker is None
    assert cfg.trigger_line_y == 0.0


def test_load_config_rejects_non_mapping(tmp_path: Path) -> None:
    yaml_path = _write_yaml(tmp_path, "- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="mapping"):
        load_config(yaml_path)


def test_station_config_rejects_trigger_line_above_one() -> None:
    with pytest.raises(ValueError, match="trigger_line_y"):
        StationConfig(
            station_id="x",
            camera_source="cam",
            frame_width=640,
            frame_height=480,
            frame_fps=30,
            hef_path=Path("m.hef"),
            hailofilter_so_path=Path("/tmp/dummy.so"),
            vehicle_classes=("car",),
            trigger_line_y=1.5,
            mqtt_broker=None,
            mqtt_topic="t",
            log_level="INFO",
        )


def test_station_config_rejects_trigger_line_below_zero() -> None:
    with pytest.raises(ValueError, match="trigger_line_y"):
        StationConfig(
            station_id="x",
            camera_source="cam",
            frame_width=640,
            frame_height=480,
            frame_fps=30,
            hef_path=Path("m.hef"),
            hailofilter_so_path=Path("/tmp/dummy.so"),
            vehicle_classes=("car",),
            trigger_line_y=-0.1,
            mqtt_broker=None,
            mqtt_topic="t",
            log_level="INFO",
        )


def test_station_config_rejects_non_positive_frame_dims() -> None:
    with pytest.raises(ValueError, match="frame_width"):
        StationConfig(
            station_id="x",
            camera_source="cam",
            frame_width=0,
            frame_height=480,
            frame_fps=30,
            hef_path=Path("m.hef"),
            hailofilter_so_path=Path("/tmp/dummy.so"),
            vehicle_classes=("car",),
            trigger_line_y=0.5,
            mqtt_broker=None,
            mqtt_topic="t",
            log_level="INFO",
        )


def test_station_config_rejects_non_positive_fps() -> None:
    with pytest.raises(ValueError, match="frame_fps"):
        StationConfig(
            station_id="x",
            camera_source="cam",
            frame_width=640,
            frame_height=480,
            frame_fps=0,
            hef_path=Path("m.hef"),
            hailofilter_so_path=Path("/tmp/dummy.so"),
            vehicle_classes=("car",),
            trigger_line_y=0.5,
            mqtt_broker=None,
            mqtt_topic="t",
            log_level="INFO",
        )


def test_load_sample_station_a_yaml() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sample = repo_root / "config" / "station_a.yaml"
    cfg = load_config(sample)
    assert cfg.station_id == "station_a"
    assert "car" in cfg.vehicle_classes
    assert cfg.mqtt_broker is None


# ─── OCR config (Phase A) ──────────────────────────────────────────────


def test_ocr_fields_default_to_fast_plate_ocr(tmp_path: Path) -> None:
    """Existing YAMLs without ocr_* keys keep the W3 behaviour."""
    yaml_path = _write_yaml(
        tmp_path,
        """
        station_id: x
        camera_source: cam
        frame_width: 640
        frame_height: 480
        frame_fps: 30
        hef_path: m.hef
        hailofilter_so_path: /tmp/dummy.so
        vehicle_classes: [car]
        trigger_line_y: 0.5
        mqtt_topic: t
        """,
    )
    cfg = load_config(yaml_path)
    assert cfg.ocr_backend == "fast-plate-ocr"
    assert cfg.ocr_model_name == "global-plates-mobile-vit-v2-model"
    # 沒寫 ocr_preprocess 就用實測最佳的策略,不是「不做前處理」
    assert cfg.ocr_preprocess == "unsharp_cubic_3x"


def test_load_config_ocr_fields_set(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        """
        station_id: x
        camera_source: cam
        frame_width: 640
        frame_height: 480
        frame_fps: 30
        hef_path: m.hef
        hailofilter_so_path: /tmp/dummy.so
        vehicle_classes: [car]
        trigger_line_y: 0.5
        mqtt_topic: t
        ocr_backend: paddleocr
        ocr_model_name: european-plates-mobile-vit-v2-model
        ocr_preprocess: cubic_3x
        """,
    )
    cfg = load_config(yaml_path)
    assert cfg.ocr_backend == "paddleocr"
    assert cfg.ocr_model_name == "european-plates-mobile-vit-v2-model"
    assert cfg.ocr_preprocess == "cubic_3x"


# ─── ocr_preprocess:具名策略 + 舊 bool 格式 ──────────────────────────


def _minimal_yaml(tmp_path: Path, extra: str) -> Path:
    return _write_yaml(
        tmp_path,
        f"""
        station_id: x
        camera_source: cam
        frame_width: 640
        frame_height: 480
        frame_fps: 30
        hef_path: m.hef
        hailofilter_so_path: /tmp/dummy.so
        vehicle_classes: [car]
        trigger_line_y: 0.5
        mqtt_topic: t
        {extra}
        """,
    )


def test_ocr_preprocess_accepts_none_alias(tmp_path: Path) -> None:
    cfg = load_config(_minimal_yaml(tmp_path, "ocr_preprocess: none"))
    assert cfg.ocr_preprocess == "none"
    # 別名要能對回正式名稱,線上才知道實際跑的是哪一支
    from speed_trap import preprocess

    assert preprocess.resolve(cfg.ocr_preprocess) == preprocess.BASELINE


def test_ocr_preprocess_rejects_unknown_strategy(tmp_path: Path) -> None:
    """拼錯字要當場炸開 —— 現場一次實驗要等一個上午的車流。"""
    with pytest.raises(ValueError, match="認不得的前處理策略"):
        load_config(_minimal_yaml(tmp_path, "ocr_preprocess: unsharpen_3x"))


def test_legacy_bool_true_maps_to_default_not_clahe(tmp_path: Path) -> None:
    """舊的 true 是 CLAHE+unsharp,實測有害 —— 不照原意還原。"""
    cfg = load_config(_minimal_yaml(tmp_path, "ocr_preprocess: true"))
    assert cfg.ocr_preprocess == "unsharp_cubic_3x"


def test_legacy_bool_false_maps_to_baseline(tmp_path: Path) -> None:
    from speed_trap import preprocess

    cfg = load_config(_minimal_yaml(tmp_path, "ocr_preprocess: false"))
    assert cfg.ocr_preprocess == preprocess.BASELINE


def test_legacy_bool_warns_to_update_the_yaml(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        load_config(_minimal_yaml(tmp_path, "ocr_preprocess: true"))
    assert any("ocr_preprocess" in record.message for record in caplog.records)


# ─── raw_ring_size ─────────────────────────────────────────────────────


def test_raw_ring_size_defaults_to_20(tmp_path: Path) -> None:
    """2304x1296 下 20 張約 171 MiB —— 跟改解析度前的 30 張 1080p 差不多。"""
    cfg = load_config(_minimal_yaml(tmp_path, "mqtt_broker: null"))
    assert cfg.raw_ring_size == 20


def test_raw_ring_size_from_yaml(tmp_path: Path) -> None:
    cfg = load_config(_minimal_yaml(tmp_path, "raw_ring_size: 30"))
    assert cfg.raw_ring_size == 30


def test_raw_ring_size_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="raw_ring_size"):
        load_config(_minimal_yaml(tmp_path, "raw_ring_size: 0"))


# ─── 最佳幀評分的預設值 ────────────────────────────────────────────────


def test_scoring_defaults_favour_the_nearest_car(tmp_path: Path) -> None:
    """邊界排除已經擋掉出框的幀,所以面積權重可以放大 —— 面積大 = 車牌像素多。"""
    cfg = load_config(_minimal_yaml(tmp_path, "mqtt_broker: null"))
    assert cfg.score_area_weight == 0.85
    assert cfg.score_center_weight == 0.05
    assert cfg.score_sharpness_weight == 0.1
    assert cfg.sharpness_reference == 900.0


def test_invalid_ocr_backend_rejected() -> None:
    with pytest.raises(ValueError, match="ocr_backend"):
        StationConfig(
            station_id="x",
            camera_source="cam",
            frame_width=640,
            frame_height=480,
            frame_fps=30,
            hef_path=Path("m.hef"),
            hailofilter_so_path=Path("/tmp/dummy.so"),
            vehicle_classes=("car",),
            trigger_line_y=0.5,
            mqtt_broker=None,
            mqtt_topic="t",
            log_level="INFO",
            ocr_backend="not-a-real-backend",
        )


def test_ocr_backend_noop_accepted() -> None:
    cfg = StationConfig(
        station_id="x",
        camera_source="cam",
        frame_width=640,
        frame_height=480,
        frame_fps=30,
        hef_path=Path("m.hef"),
        hailofilter_so_path=Path("/tmp/dummy.so"),
        vehicle_classes=("car",),
        trigger_line_y=0.5,
        mqtt_broker=None,
        mqtt_topic="t",
        log_level="INFO",
        ocr_backend="noop",
    )
    assert cfg.ocr_backend == "noop"


# ─── W3 v2: cascade + debug crops ──────────────────────────────────────


def test_cascade_fields_default_to_none_off(tmp_path: Path) -> None:
    """Existing YAMLs without v1.6.0 keys keep the W3 v1 behaviour."""
    yaml_path = _write_yaml(
        tmp_path,
        """
        station_id: x
        camera_source: cam
        frame_width: 640
        frame_height: 480
        frame_fps: 30
        hef_path: m.hef
        hailofilter_so_path: /tmp/dummy.so
        vehicle_classes: [car]
        trigger_line_y: 0.5
        mqtt_topic: t
        """,
    )
    cfg = load_config(yaml_path)
    assert cfg.ocr_plate_detector_path is None
    assert cfg.save_debug_crops is False


def test_load_config_cascade_fields_set(tmp_path: Path) -> None:
    yaml_path = _write_yaml(
        tmp_path,
        """
        station_id: x
        camera_source: cam
        frame_width: 640
        frame_height: 480
        frame_fps: 30
        hef_path: m.hef
        hailofilter_so_path: /tmp/dummy.so
        vehicle_classes: [car]
        trigger_line_y: 0.5
        mqtt_topic: t
        ocr_plate_detector_path: /home/kevin30/speed-trap/models/plate_detector.pt
        save_debug_crops: true
        """,
    )
    cfg = load_config(yaml_path)
    assert cfg.ocr_plate_detector_path == "/home/kevin30/speed-trap/models/plate_detector.pt"
    assert cfg.save_debug_crops is True
