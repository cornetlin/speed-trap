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

## License

[MIT](LICENSE) — 可任意使用、修改、再散布,保留版權聲明即可。
