# speed-trap

校園區間測速 PoC — 在路口部署兩站邊緣裝置(A 站、B 站),
透過車輛偵測 + 雙站時戳比對計算路段平均速度。

## 技術 stack

- **硬體**: Raspberry Pi 5 + Hailo-8 AI HAT+
- **影像 / 偵測**: Picamera2、GStreamer、Hailo Python bindings
- **核心邏輯**: Python 3.11+(純 Python,可單元測試)
- **通訊**: MQTT(雙站共用 broker)
- **品質**: pytest + mypy strict + ruff

## 目錄結構

| 目錄 | 用途 |
|---|---|
| `speed_trap/` | 核心邏輯 package(tracker、trigger、event、config — 不依賴硬體) |
| `apps/` | 執行入口:`replay_video`(PC 重放)、`run_station`(Pi 上線) |
| `tests/` | pytest 測試 |
| `config/` | 站點 YAML 設定 |
| `scripts/` | 部署 / 啟動腳本 |
| `samples/` | 離線測試影片(gitignored) |
| `docs/` | 規劃與部署文件 |

## 快速開始

| 我想做的事 | 看哪份文件 |
|---|---|
| 在 PC 上 clone 後開始改 code、跑 pytest | [CLAUDE.md](CLAUDE.md) |
| 把這個 repo 部署到一台 Pi 上跑起來 | [docs/pi-setup-simple.md](docs/pi-setup-simple.md) |
| 看整體 6 週驗證規劃與里程碑 | [docs/6week-roadmap.md](docs/6week-roadmap.md) |

## 安裝

```bash
pip install -e .          # 必要依賴
pip install -e ".[dev]"   # 另加 pytest / mypy / ruff
```

`pip install -r requirements.txt` 裝出來的內容與 `pip install -e .` 相同,
兩份清單刻意保持一致。

**PyGObject(`gi`)與 `hailo` 不要用 pip 裝** —— 兩者走 apt,理由與指令寫在
[requirements.txt](requirements.txt) 的註解裡。缺少它們時 `speed_trap` 仍可
import,只有實際啟動 `HailoDetectionSource` 才會報錯,所以 PC 上跑 pytest
不需要 Hailo 環境。

### 可選:PaddleOCR 後端

預設的 OCR 後端是 fast-plate-ocr(約 10 MB,Pi CPU 上 10–30 ms)。想改用
PaddleOCR(中文字符集較強,但套件約 150 MB)時才需要另外安裝:

```bash
pip install -e ".[paddle]"
```

(這個 extra 含 `paddleocr` 與推論後端 `paddlepaddle`,兩個都要才跑得起來。
Pi 的 aarch64 平台上 `paddlepaddle` 官方 PyPI 可能沒有對應 wheel,屆時要另
找平台版本。)

然後在站台 YAML 切換後端:

```yaml
ocr_backend: paddleocr
```

沒安裝就把 `ocr_backend` 設成 `paddleocr` 的話,系統會記一則 warning 並退回
`noop`(只跑車輛偵測、不讀車牌),不會整個掛掉。

## License

[MIT](LICENSE) — 可任意使用、修改、再散布,保留版權聲明即可。
