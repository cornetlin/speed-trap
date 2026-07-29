# speed-trap

區間測速邊緣站軟體 — 跑在 Raspberry Pi 5 + Hailo AI HAT+ 上的 PoC。

## 專案用途

在校園道路部署兩站(A 站、B 站)的邊緣推論裝置,各站透過 Hailo 加速器偵測車牌與車輛,
對同一車輛在兩站之間的通過時間戳計算平均速度。目前處於 **校園驗證階段**(PoC),重點是:

- 驗證偵測與車牌比對的準確度。
- 驗證雙站之間時戳同步、MQTT 通訊的穩定度。
- 測試 Pi 5 + Hailo 在長時間運行下的熱穩定與吞吐。

尚未進入量產,程式碼以 **可讀、易改、好測試** 為優先,效能優化延後。

## 架構決策

- **核心邏輯純 Python、可單元測試**:`speed_trap/` 內的模組(比對、狀態機、速度計算、
  時戳管理)不依賴硬體,在 PC 上能直接跑 pytest。
- **Hailo / GStreamer / Picamera2 adapter 獨立**:硬體相關程式碼放在各自 adapter 模組中
  (例如 `speed_trap/adapters/hailo.py`),以 protocol/介面抽象出去,核心邏輯只依賴介面。
  PC 上用 fake/stub adapter 測試,Pi 上才切到真實 adapter。
- **apps/ 是薄執行入口**:組裝 adapter + 核心邏輯 + 設定檔,不放商業邏輯。
- **設定走 YAML**:`config/` 下一份 YAML 描述站點(ID、相機、MQTT broker、ROI 等),
  方便兩站用同一份程式不同 config。

### 目錄

- `speed_trap/` — 核心邏輯 package(純 Python)。
- `apps/` — 執行腳本(例如 `apps/station.py`、`apps/simulator.py`)。
- `tests/` — pytest 測試。
- `config/` — 站點 YAML 設定檔。
- `scripts/` — 部署、rsync、Pi 側 helper shell 腳本。
- `samples/` — 離線測試用的影片片段(`*.mp4` 被 gitignore)。

## 開發流程

1. **PC 端開發 + 單元測試**
   - 建 venv:`python -m venv .venv && source .venv/Scripts/activate`(Windows bash)。
   - 裝開發相依:`pip install -e .[dev]` 或 `pip install -r requirements-dev.txt` + `pip install -e .`。
   - 跑測試:`pytest`、型別檢查:`mypy speed_trap`、格式/lint:`ruff check . && ruff format .`。
   - `picamera2` **不要** 寫進 requirements — Pi 上用 `apt install python3-picamera2` 處理。

2. **rsync 到 Pi 做整合測試**
   - Pi 側預先:系統 apt 裝 picamera2、Hailo runtime;建 venv;`pip install -r requirements.txt && pip install -e .`。
   - 本機改完 → `scripts/sync_to_pi.sh`(rsync,排除 `.venv/`、`__pycache__/`、`samples/*.mp4`)。
   - Pi 上跑 `apps/station.py --config config/station_a.yaml`,觀察 log 與 MQTT 實際行為。
   - 整合測試的結果不寫進 unit test — unit test 只驗純邏輯。

3. **離線重放**
   - 錄好的路口影片放 `samples/*.mp4`(gitignore),用 `apps/simulator.py` 在 PC 上重放給核心邏輯,
     不需要真相機也能除錯偵測後的流程。

## 程式碼風格

- **型別標註 100%**:所有函式參數與回傳值加 type hints,mypy `strict = true`。
- **優先 `@dataclass`**:表達資料結構(偵測結果、站點狀態、設定)用 dataclass(需要不可變就
  `frozen=True`),不要散落的 dict。
- **路徑一律 `pathlib.Path`**:避免 `os.path` 字串拼接。
- **io 邊界才 try/except**:核心邏輯不做防禦性 try/except;只在外部邊界(檔案、網路、相機)處理例外。
- **logging 不用 print**:模組級 `logger = logging.getLogger(__name__)`。
- **註解只寫 why,不寫 what**:變數命名把 what 講清楚;註解留給不明顯的限制或決策。

## 未列但可能重要的事

- 時戳精度目標:毫秒級,後續若要次毫秒需處理 NTP/PTP,不在 PoC 範圍。
- 資料隱私:車牌屬個資,落地寫檔前就要決定保存策略(目前 PoC 階段先全存,量產前再收)。
