# 區間測速專案 — 6 週開發 Roadmap

**專案名稱**: Speed Trap — Edge AI 區間測速系統
**指導老師**: 林子軒 教授(中央大學土木工程學系)
**參與學生**: [專題生姓名]
**期程**: 6 週(校園驗證階段 PoC,密集開發)

---

## 專案總覽

### 研究目標

開發一套低成本、可部署於校園環境的區間測速原型系統,結合 Raspberry Pi 5 + Hailo AI HAT+ 邊緣運算平台,驗證以下技術可行性:

1. **邊緣 AI 即時車輛/車牌辨識**: Hailo NPU 上跑串接的 YOLO + LPRNet pipeline
2. **多站點時間同步事件配對**: 兩站獨立辨識,透過車牌字串配對計算路段平均車速
3. **TinyML 在交通感測的應用**: 延伸實驗室既有 SHM / 邊緣 AI 研究方法到交通領域

### 預期成果

- 兩站點完整運作的 PoC 系統
- 校園實拍車輛事件資料集
- 系統設計文件(可投稿研討會或作為碩士論文素材)
- 學生能獨立操作部署、訓練、除錯的技術能力

### 整體架構

```
站點 A (Pi-A)              站點 B (Pi-B)
─────────────              ─────────────
相機 → Hailo NPU          相機 → Hailo NPU
  ↓                          ↓
車輛偵測 (YOLOv8n)         車輛偵測 (YOLOv8n)
  ↓                          ↓
車牌偵測 + OCR             車牌偵測 + OCR
  ↓                          ↓
事件 (車牌+時戳)           事件 (車牌+時戳)
       ↘                  ↙
        中央配對伺服器
        (PC 或一台 Pi)
        ↓
     平均車速計算 + 違規記錄
```

### 6 週時程總表

| 週次 | 主題 | 主要交付物 |
|------|------|-----------|
| W1 | 程式骨架 + Pi setup + 車輛偵測上線 | speed-trap v1.0、Pi 上跑通車輛偵測 |
| W2 | 車牌偵測模型訓練 + 編譯 | 校園資料集、`plate.hef` 編譯完成 |
| W3 | 車牌 OCR + 完整單站 ALPR pipeline | 單站能輸出車牌字串事件 |
| W4 | 中央配對伺服器 + 雙站通訊 | MQTT broker、配對演算法、第二台 Pi 上線 |
| W5 | 校園實地測試 + 效能調校 | 實地資料、準確率/車速誤差分析 |
| W6 | 系統穩定化 + 文件 + 報告 | systemd、最終 demo、期末報告 |

> 💡 **進度緩衝**: 6 週密集表沒有 buffer week。若某週進度落後,優先犧牲「美化、優化」項目,核心功能必須完成。

---

## Week 1 — 程式骨架 + Pi setup + 車輛偵測上線

### 主目標

PC 端完成 `speed-trap/` v1.0 程式骨架;學生完成 Pi 環境;週末把程式部署到 Pi、跑通**車輛**偵測(尚無車牌)。

### 交付物

- ✅ `speed-trap/` repo v1.0(`git tag v1.0`)
- ✅ Pi 硬體組裝、OS、Hailo runtime 全就緒
- ✅ Pi 上能跑 `apps/run_station.py`,看到車輛偵測事件 JSON 印出
- ✅ pytest 全綠、ruff/mypy 過

### 技術重點

**架構決策**: 業務邏輯與硬體 adapter 解耦
- `speed_trap/` 純 Python 模組可在任何 PC 跑單元測試
- `speed_trap/hailo_source.py` 是唯一觸碰 Hailo / GStreamer 的地方
- 後續週次迭代不需要每次部署到 Pi 才能驗證邏輯

**Pi 端關鍵設定**:
- Raspberry Pi OS Trixie 64-bit
- `sudo apt install hailo-all` 一次裝好 driver、runtime、tappas
- venv 一定要加 `--system-site-packages` 看得到 picamera2

**第一週用的模型**:Hailo Model Zoo 的預編譯 `yolov8n.hef`(COCO 80 類),從中過濾 `car / truck / bus / motorcycle`。**這週還不訓練自己的模型**。

### 學生工作

照 `pi-setup-simple.md` 從 Step 1 跑到 Step 6:
1. 硬體組裝 + OS 燒錄 + Hailo 套件安裝
2. 收 PC 端傳來的 `speed-trap/` 資料夾
3. 建 Python venv、跑 pytest 全綠
4. 用網路下載的車流影片跑 replay
5. 接真相機跑完整 pipeline 看到 JSON 事件

### 學習資源

- [Raspberry Pi AI HAT+ 官方文件](https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html)
- [Hailo Examples for Pi 5](https://github.com/hailo-ai/hailo-rpi5-examples) — 必讀,理解 GStreamer + Hailo callback 架構
- [Hailo RPi5 Detection Tutorial](https://github.com/hailo-ai/hailo-rpi5-examples/blob/main/doc/basic-pipelines.md)

### 卡關處理

| 症狀 | 解法 |
|-----|-----|
| `hailortcli fw-control identify` 找不到裝置 | PCIe ribbon 鬆了或裝反、`dmesg \| grep hailo` 看 firmware 載入訊息 |
| `pip install picamera2` 失敗 | 不要用 pip,用 `sudo apt install python3-picamera2` |
| 主程式跑但沒事件 | 按 rpicam-hello → Hailo demo → replay → trigger_line_y 順序排查 |
| Ctrl+C 後重跑 GStreamer 卡死 | `sudo fuser -k /dev/hailo0` |

### 週末檢查點

教授可以驗證的事:
```bash
ssh ncu@speedtrap.local 'cd ~/speed-trap && bash scripts/run_on_pi.sh' | head -20
```
應該看到至少一個 JSON 事件印出,代表 W1 過關。

---

## Week 2 — 車牌偵測模型訓練 + 編譯

### 主目標

訓練一個專門偵測**車牌位置**(不辨識內容)的 YOLOv8n,並編譯成 Hailo 能跑的 `.hef` 格式。

### 交付物

- ✅ 校園實拍影像 ≥ 500 張(學生收集,W1 開始累積)
- ✅ 訓練完的 `plate_detector.pt`(PyTorch 格式)
- ✅ 編譯好的 `plate_detector.hef`(Hailo 格式,可在 Pi 上跑)
- ✅ 在 PC 上用 ultralytics 驗證準確率報告(mAP、precision、recall)

### 技術重點

**資料來源**:
- **Roboflow Taiwan License Plate Dataset**: 約 3,360 張,YOLOv8 格式,專為台灣車牌設計
- **AOLP (Application-Oriented License Plate Database)**: 約 1,874 張,分 AC/LE/RP 三子集
- **校園實拍**: 學生用 Pi 在校門口/停車場錄影,從中標註 ≥ 500 張(讓模型學習我們**自己的角度與光線**)

**訓練流程**:
1. 合併三個資料集,以校園資料為主、其他為輔
2. 80/15/5 切分 train/val/test
3. ultralytics 訓練:`yolo train model=yolov8n.pt data=plate.yaml imgsz=640 epochs=100`
4. 最佳權重 export to ONNX
5. Hailo Dataflow Compiler (DFC) 量化 + 編譯成 `.hef`

**Hailo DFC 編譯三步驟**:
```bash
hailo parser onnx best.onnx
hailo optimize best.har --calib-set-path calibration/
hailo compile best.har
```

> ⚠️ **DFC 註冊**: Hailo Dataflow Compiler 需要在 Hailo Developer Zone 註冊帳號才能下載,建議用學校 email 申請,通過會比個人 email 順。

### 學生工作

**前半週(資料蒐集與標註)**:
- 用 W1 上線的 Pi 在校園拍攝車流影片(每段 5–10 分鐘,多角度多時段)
- 用 LabelImg 或 Roboflow 標註車牌 bounding box
- 整理出至少 500 張標註好的校園影像

**後半週(訓練與編譯)**:
- 在實驗室 GPU 機器上跑 ultralytics 訓練(教授協助)
- 跑 Hailo DFC 編譯流程(教授協助)
- 在 Pi 上 sanity test 編譯後的 `.hef` 能載入

### 學習資源

- [Roboflow Taiwan Plate Dataset](https://universe.roboflow.com/jackresearch0/taiwan-license-plate-recognition-research-tlprr) — 直接下載即可
- [AOLP Dataset](https://github.com/AvLab-CV/AOLP) — 學術用,需引用
- [ultralytics YOLOv8 訓練教學](https://docs.ultralytics.com/modes/train/)
- [Hailo Model Zoo retraining guide](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/RETRAIN_ON_CUSTOM_DATASET.rst) — Hailo 官方 retrain 範例

### 卡關處理

| 症狀 | 解法 |
|-----|-----|
| GPU 機器 OOM | batch size 從 16 降到 8 或 4;imgsz 從 640 降到 416 |
| 訓練 mAP 卡在 0.6 附近 | 資料量不夠,加入更多 Roboflow 資料;檢查標註品質;延長 epochs 到 200 |
| Hailo DFC 量化後精度暴跌 | calibration set 不夠 representative,改用 200 張涵蓋各種光線/角度的影像 |
| `.hef` 編譯成功但 Pi 上載入失敗 | 檢查 DFC 版本與 Pi 上 HailoRT 版本是否相容 |

### 週末檢查點

```bash
# 在 PC 上驗證模型
yolo predict model=plate_detector.pt source=samples/test_plate.jpg

# 在 Pi 上驗證 .hef 能載入
hailortcli run plate_detector.hef --measure-latency
```

---

## Week 3 — 車牌 OCR + 完整單站 ALPR pipeline

### 主目標

把 Hailo 的 LPRNet (或 fast-plate-ocr) 串到 W2 的車牌偵測之後,**單站完整跑通**:畫面進來 → 車輛 → 車牌 → 字串 → 事件帶車牌號碼。

### 交付物

- ✅ `speed_trap/plate_recognizer.py` 模組(車牌字元辨識)
- ✅ `apps/run_station.py` 升級為三模型 cascade pipeline
- ✅ 事件 JSON 多一個 `plate_text` 欄位
- ✅ 台灣車牌格式驗證(regex)濾掉偽陽性
- ✅ 在校園實拍影片上的端到端準確率報告

### 技術重點

**OCR 路線選擇**(W3 一開始就要決定):
- **路線 A: LPRNet on Hailo** — 速度快(<10ms),但需要 fine-tune 到台灣字集
- **路線 B: Character-YOLO** — 把每個字元當一類,訓練 YOLOv8 偵測,後處理拼字串
- **路線 C: fast-plate-ocr on CPU** — 簡單快速,Pi 5 CPU 上 10–30ms,先做 baseline

**建議策略**: 先用路線 C(CPU)拿到 baseline 準確率,如果速度夠用就先 ship,後面有時間再升級到路線 A。

**Cascade pipeline 串接**:
```
影格 → YOLOv8 (車輛) → crop → YOLOv8 plate (車牌位置) → crop → OCR (字串)
```

GStreamer 上的實作可以參考 Hailo 官方 ALPR pipeline guide,中間用 `hailocropper` element 做 ROI crop。

**台灣車牌格式驗證 regex**:
```python
TAIWAN_PLATE_PATTERNS = [
    r"^[A-Z]{3}-\d{4}$",   # 現行: ABC-1234
    r"^\d{4}-[A-Z]{2}$",   # 舊式: 1234-AB
    r"^[A-Z]{2}-\d{4}$",   # 機車: AB-1234
    r"^[A-Z]{3}-\d{3}$",   # 機車: ABC-123
]
```
不符合任何 pattern → confidence 降權或丟棄。

### 學生工作

- 整合 OCR 模型到 `speed_trap/plate_recognizer.py`
- 寫端到端測試:用 W2 蒐集的影像做 ground truth,計算車牌讀取準確率
- 跑 Pi 上 30 分鐘長時間測試,記錄各模型階段的 FPS

### 學習資源

- [Hailo ALPR Reference](https://github.com/hailo-ai/Hailo-Application-Code-Examples/tree/main/runtime/python/license_plate_recognition) — 官方 ALPR 完整實作
- [LPRNet 論文](https://arxiv.org/abs/1806.10447v3) — 理解架構
- [fast-plate-ocr](https://github.com/ankandrew/fast-plate-ocr) — 快速 baseline

### 卡關處理

| 症狀 | 解法 |
|-----|-----|
| OCR 把 0/O、I/1、B/8 認錯 | 台灣車牌實際上不用 I 和 O,加進 regex 過濾 |
| 同一車輛多幀讀出不同字串 | 在 tracker 內累積多幀結果,投票取最高頻字串 |
| 整體 FPS 從 30 掉到 10 | 三模型串接時 GPU/NPU 排程問題,試 batch size、用 hailotracker 跳過已辨識的 track |
| 車牌偵測 OK 但 OCR 一直空白 | OCR 輸入的 crop 解析度不夠,放大到 ≥ 96×24 px |

### 週末檢查點

學生在 Pi 上播放 5 分鐘車流影片,**端到端準確率 ≥ 70%**(校園清晰場景)算單站初版完成。低於這個值要回頭看是車牌偵測還是 OCR 拖累。

---

## Week 4 — 中央配對伺服器 + 雙站通訊

### 主目標

第二台 Pi 上線,兩站事件透過 MQTT 送到中央伺服器,中央做車牌配對 + 平均車速計算。

### 交付物

- ✅ 中央伺服器(實驗室一台 PC)跑 Mosquitto MQTT broker
- ✅ `central/matcher.py` 配對引擎
- ✅ 第二台 Pi(Pi-B)硬體就緒、軟體部署完成
- ✅ SQLite / PostgreSQL 儲存配對成功的速度事件
- ✅ 雙站時間同步驗證(NTP < 20ms)

### 技術重點

**通訊架構**:
- 邊緣站:`paho-mqtt` publish 到 `traffic/events/{station_id}` topic
- 訊息格式:JSON `{station_id, plate, timestamp_ns, confidence, image_hash}`
- QoS=1 確保不掉訊息

**配對演算法**:
```python
# 收到 B 站事件 → 找 A 站歷史事件
def match_event(event_b):
    # 合理車速 5–100 km/h
    L = config.section_distance_m  # 路段長度
    t_min = event_b.timestamp_ns - (L / 5 * 3.6 * 1e9)
    t_max = event_b.timestamp_ns - (L / 100 * 3.6 * 1e9)

    candidates = query_a_events(t_min, t_max)
    # 模糊比對車牌(允許 1-2 字元 OCR 錯誤)
    best = min(candidates, key=lambda e: levenshtein(e.plate, event_b.plate))
    if levenshtein_distance <= 2:
        speed = L / ((event_b.timestamp_ns - best.timestamp_ns) / 1e9)
        record_passage(best, event_b, speed)
```

**時間同步**:
- 校園驗證階段用 NTP 即可(精度 5–20ms)
- 實驗室自架 chrony NTP server,兩 Pi 都指向它(比公網 NTP 穩)
- 之後若要升級到 GPS PPS 是 W6 buffer 任務

### 學生工作

- 第二台 Pi 重複 W1 的 setup 流程(這次會快很多)
- 設定 `config/station_b.yaml`(只差 station_id 和 mqtt_topic)
- 架設 Mosquitto broker 並測試通訊
- 寫配對引擎並用合成資料測試

### 學習資源

- [Eclipse Mosquitto](https://mosquitto.org/) — 最簡單的 MQTT broker
- [paho-mqtt Python](https://eclipse.dev/paho/index.php?page=clients/python/index.php)
- [chrony 設定教學](https://chrony-project.org/documentation.html) — NTP 進階

### 卡關處理

| 症狀 | 解法 |
|-----|-----|
| 兩站時間差 >100ms | 改用實驗室 LAN 內 NTP,不要走外網 |
| MQTT 偶爾掉訊息 | 確認 QoS=1,broker 端啟用 persistence |
| 配對到錯誤的車輛 | 收緊 levenshtein 距離閾值;加 confidence 加權 |
| 配對演算法跑很慢 | A 站歷史事件用 SQLite 索引 timestamp 欄位 |

### 週末檢查點

實驗室桌面用兩台 Pi 模擬:Pi-A 和 Pi-B 相距 5 公尺,玩具車從 A 走到 B,中央伺服器應該記錄到一筆配對成功的事件,速度合理(0.5–2 km/h)。

---

## Week 5 — 校園實地測試 + 效能調校

### 主目標

把兩台 Pi 拉出實驗室,在校園實地路段做控制實驗,蒐集真實資料,分析準確率與誤差來源。

### 交付物

- ✅ 校園實地部署紀錄(地點、距離、相機角度、光線條件)
- ✅ Ground truth 比對資料(用手機 GPS / 雷達槍當對照)
- ✅ 端到端準確率分析報告(車牌讀取率、配對成功率、車速誤差分布)
- ✅ 效能優化(若 FPS 不足或誤差過大)

### 技術重點

**實地測試地點建議**:
- **NCU 校內主幹道**(車速適中,有測試樣本)
- **停車場出入口**(單向車流最乾淨)
- **校警室前直路**(可以協調封閉測試)

**Ground Truth 取得方法**:
- **手機 GPS logger**(SpeedView app):±0.5 km/h 精度
- **獨立 GPS tracker**(NEO-M8N + microSD):±0.1 km/h
- **借用雷達測速槍**:第三方參考最有說服力

**Controlled test 流程**:
1. 司機開測試車以 cruise control 定速通過(20、40、60 km/h 各 10 次)
2. 系統記錄事件、Ground truth 同步記錄
3. 計算系統車速 vs 真實車速的均值誤差和標準差

**誤差來源分析框架**:
| 誤差來源 | 估計大小 | 處理方式 |
|---------|---------|---------|
| NTP 時間同步 | <20 ms | 升級 GPS PPS 才能進一步降低 |
| 車牌觸發點不一致 | ~30 ms | 觸發線位置嚴格定義 |
| OCR 誤讀導致無法配對 | 影響配對率非車速 | 提升 OCR 準確率 |
| 路段長度量測 | <0.1 m | 用 RTK GPS 或捲尺多次量 |

### 學生工作

- 帶兩台 Pi 出實驗室,架設、量距離、拍照記錄部署現場
- 開測試車跑 controlled test
- 分析資料寫成簡短報告(Jupyter notebook + matplotlib 圖表)

### 學習資源

- [pandas 時序資料分析](https://pandas.pydata.org/docs/user_guide/timeseries.html)
- [matplotlib 統計圖表](https://matplotlib.org/stable/gallery/index.html)

### 卡關處理

| 症狀 | 解法 |
|-----|-----|
| 戶外光線太強 OCR 失敗 | 相機 ND filter、調整曝光時間 |
| 相機距離太遠車牌太小 | 換長焦鏡頭(16mm)或挪近站點 |
| 配對率偏低 (<50%) | 檢查兩站 OCR 準確率、放寬 levenshtein 閾值 |
| 車速誤差 >5 km/h | 主要原因通常是觸發點抖動,檢查 trigger_line 邏輯 |

### 週末檢查點

至少 50 筆有效的「同一車輛兩站配對成功 + 速度計算結果」資料,**車速誤差中位數 < 3 km/h**(校園驗證階段可接受)。

---

## Week 6 — 系統穩定化 + 文件 + 報告

### 主目標

把系統做成「可以離開人手 24/7 運轉」的狀態,完成所有文件、demo 影片、期末報告。

### 交付物

- ✅ systemd service 自動啟動 + 異常重啟
- ✅ 一週連續運轉日誌(無 crash、記憶體無洩漏)
- ✅ README.md、技術文件、學生使用手冊
- ✅ Demo 影片(3–5 分鐘,展示完整工作流)
- ✅ 期末報告 PDF(可作為碩論章節或研討會投稿基礎)
- ✅ GitHub repo 整理乾淨,`v1.0` release

### 技術重點

**穩定性 hardening**:
- systemd unit file 設 `Restart=always`、`RestartSec=10`
- log rotation:`journalctl` 限制 size 與保留時間
- watchdog:獨立 systemd timer 每 10 分鐘檢查事件流是否正常

**文件結構**:
```
speed-trap/
├── README.md              # 簡介 + 快速 start
├── docs/
│   ├── architecture.md    # 系統架構詳述
│   ├── pi-setup-simple.md # 學生部署手冊
│   ├── training-guide.md  # 模型訓練流程
│   └── deployment.md      # 雙站部署 SOP
└── reports/
    ├── final_report.pdf   # 期末報告
    └── demo.mp4           # demo 影片
```

**期末報告建議架構**:
1. Introduction(問題、動機、貢獻)
2. Related Work(既有區間測速、邊緣 AI 應用)
3. System Design(架構、硬體、軟體 stack)
4. Implementation(三模型 pipeline、配對演算法、時間同步)
5. Evaluation(W5 的實地測試結果)
6. Discussion(限制、未來工作)
7. Conclusion

### 學生工作

- systemd 服務化,讓系統能自動啟動 + 異常自動重啟
- 跑一週連續測試,每天檢查狀態
- 寫使用手冊(下一屆學弟妹接手用)
- 錄 demo 影片
- 整理 GitHub repo,寫好 README

### 學習資源

- [systemd unit file 教學](https://www.freedesktop.org/software/systemd/man/systemd.unit.html)
- [Markdown 文件最佳實踐](https://www.markdownguide.org/)
- [GitHub README 範本](https://github.com/othneildrew/Best-README-Template)

### 卡關處理

| 症狀 | 解法 |
|-----|-----|
| systemd 啟動失敗 | `journalctl -u speed-trap -n 50` 看錯誤;通常是 venv 路徑或 user 設錯 |
| 跑了三天突然 crash | 檢查記憶體使用量是否持續成長,可能 GStreamer pipeline 有 leak |
| log 把磁碟塞爆 | systemd 加 `StandardOutput=journal` + 設 journald 大小上限 |

### 結業檢查點

- [ ] 系統能在無人值守下穩定跑滿 7 天
- [ ] GitHub repo 有完整文件、學弟妹能照著重現
- [ ] Demo 影片清楚展示「車輛通過 → 車牌辨識 → 配對 → 計算車速」全流程
- [ ] 期末報告成稿可呈交

---

## 風險管理

### 6 週密集排程的高風險點

**風險 1: W2 模型訓練撞牆**(機率高)
- *症狀*: 訓練好的車牌偵測 mAP 卡在低水位
- *對策*: 預留兩個資料集備選(Roboflow + AOLP),不夠就合併用;校園資料量在 W1 就要開始累積

**風險 2: W3 OCR 準確率不夠**(機率中)
- *症狀*: 端到端準確率 <50%
- *對策*: 路線 C (fast-plate-ocr) 是保險,先用它做 baseline 不要等完美

**風險 3: W4 第二台 Pi 硬體出問題**(機率低但影響大)
- *對策*: W1 結束時建議多訂一片備用 AI HAT+;若拖到 W4 才發現故障,是 6 週密集排程承受不起的

**風險 4: W5 戶外光線/距離問題**(機率高)
- *症狀*: 實驗室能跑、戶外掉準確率
- *對策*: W3 結束就要去戶外做小範圍測試,不要等到 W5 才面對戶外環境

### 應急縮小範圍方案

如果 W4 結束發現 6 週做不完整版,可以**縮小到單站可用**:
- 放棄 W4 雙站配對
- W5 改成「單站車輛計數 + 平均車速估算」(用畫面內車輛通過時間估)
- W6 仍照計畫做文件、報告

這樣仍是一個可發表的 PoC,只是降到「單站交通流量監測」應用,而非「區間測速」。

---

## 成功標準

### 最低標準(MVP)
- 單站能即時辨識車輛 + 車牌,輸出帶時戳事件
- 兩站 demo 能配對至少 20 筆有效事件
- 文件完整,他人能照做重現

### 期望標準
- 端到端車牌讀取率 ≥ 70%
- 雙站配對成功率 ≥ 80%
- 校園實地車速估計誤差中位數 < 3 km/h
- 系統可穩定運轉 7 天無人介入

### 優秀標準
- 端到端車牌讀取率 ≥ 85%
- 配對成功率 ≥ 90%
- 車速誤差 < 1.5 km/h
- 投稿一篇 conference paper 或 demo paper

---

## 給學生的話

這是一個**真正在做研究**的專案,不是 lab 練習。每一週都會碰到「文件沒寫、套件出 bug、模型不準」的真實狀況。卡關不丟人,**不問才會延誤進度**。

每週五跟教授會議,先列「這週做了什麼、卡在哪、下週計畫」三點即可,不用 PPT。Live demo 比 slides 更能溝通。

技術之外,把這當成一次「從 idea 到部署的完整工程歷練」 — 寫文件、debug、跟人協作、面對 deadline。這些在課堂上學不到。

---

## 給教授/評審的話

本專案的研究貢獻定位:

1. **方法層**:驗證 Hailo-8 NPU 在低成本邊緣 AI 部署的可行性,以區間測速為應用案例
2. **系統層**:多站點時間同步事件配對的完整實作,從硬體到演算法都開源
3. **資料層**:校園實拍車輛事件資料集(可去識別化後公開)
4. **教育層**:作為土木/交通工程系所學生接觸 edge AI 的入門教材

未來延伸方向(學生畢業後可接續):
- GPS PPS 升級到 sub-millisecond 同步,進入執法級精度
- 多車道、多車型的擴充 dataset
- 跟 NCU 校警合作做實際部署測試
- TinyML on-robot 模型蒸餾(承接實驗室 SHM 的 edge AI 研究脈絡)
