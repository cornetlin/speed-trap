# 區間測速 — Pi 部署與測試(單台)

**你的目標**: 把 PC 端寫好的程式碼跑在這一台 Pi 上,驗證相機 + Hailo + 車輛偵測 + 事件輸出都能動。

**預計時間**: 3–5 天。

---

## 你會拿到的東西

PC 端會給你一個 `speed-trap/` 資料夾,裡面有:

- `apps/run_station.py` — 主程式(要跑的就是這個)
- `apps/replay_video.py` — 用影片檔測試(不用相機也能跑)
- `config/station_a.yaml` — 設定檔(你會動這個)
- `tests/` — pytest 測試
- `scripts/run_on_pi.sh` — 啟動腳本

---

## Step 1 — 硬體與 OS

### 1.1 組裝

**先準備好這些零件**:
- Raspberry Pi 5 主機板
- Active Cooler(散熱風扇,黏在 Pi 的 CPU 上)
- AI HAT+(Hailo AI 加速板) + PCIe ribbon cable
- 相機模組(Pi Camera 3 或 HQ Camera)+ 相機排線
- 27W USB-C 電源供應器
- microSD 卡(≥32GB Class 10)或 NVMe SSD

**組裝順序**(務必先斷電):

1. **Active Cooler** 壓到 Pi 主板上,風扇接頭插在 Pi 的 4-pin JST 插座
2. **AI HAT+** 用 PCIe ribbon 連到 Pi,金屬接點**朝 USB 那一側**
3. **相機排線**插到 Pi 的 CAM 插槽,藍色補強膠帶**朝 USB 那一側**

圖文教學: <https://www.raspberrypi.com/documentation/accessories/ai-hat-plus.html>

不確定對不對,YouTube 搜尋「Raspberry Pi 5 AI HAT install」看一兩個影片再動手。

> ⚠️ **PCIe ribbon 最脆弱**,裝反整片 AI HAT+ 會燒掉。比對官網照片再壓下。

### 1.2 燒 OS 與裝 Hailo

**A. 下載 Raspberry Pi Imager**:
<https://www.raspberrypi.com/software/>(Mac / Windows / Linux 都有)

**B. 燒錄 OS**:

1. SD 卡(或 NVMe SSD)接到**你自己的電腦**
2. 開啟 Raspberry Pi Imager
3. **Device**: 選 `Raspberry Pi 5`
4. **Operating System**: 選 `Raspberry Pi OS (64-bit)`(預設就是 Trixie)
5. **Storage**: 選你的 SD 卡 / SSD
6. 點 **NEXT** → **Edit Settings(編輯設定)**:
   - General 頁:
     - Hostname: `speedtrap`
     - Username / Password: 自己設(**記起來**,SSH 要用)
     - Wi-Fi: 填實驗室 AP(或用有線網路可留空)
     - Time zone: `Asia/Taipei`
   - Services 頁: 勾 **Enable SSH**,用 password 驗證
7. 點 **Save** → **Yes** 開始燒(10–20 分鐘)

**C. 首次開機**:

卡插回 Pi、接相機、接電源。第一次開機要 2–3 分鐘(會自動重開一次)。

**D. 從你電腦 SSH 進 Pi**:

```bash
# Mac / Linux Terminal、或 Windows PowerShell(Win10 以上內建)
ssh <username>@speedtrap.local
```

第一次會問 `yes/no` 打 `yes`,然後輸入剛才設的密碼。

**連不上 `.local`?**(有些網路環境 mDNS 不通):
- 接螢幕鍵盤到 Pi,查 IP:`ip addr | grep inet`
- 或路由器後台看 DHCP 客戶端清單
- 用 IP 連:`ssh <username>@192.168.x.x`

**E. 更新系統並裝 Hailo**(在 Pi 上):

```bash
sudo apt update && sudo apt full-upgrade -y
sudo rpi-eeprom-update -a
sudo reboot
```

SSH 重連進去:

```bash
sudo apt install dkms hailo-all rpicam-apps git python3-picamera2 -y
sudo reboot
```

SSH 重連進去就完成了。(總共重開兩次,每次要重新 SSH)

### 1.3 驗收(三個指令)

```bash
hailortcli fw-control identify       # 看到 Board Name: Hailo-8
rpicam-still -o /tmp/test.jpg        # 產生 >50KB 的 jpg
rpicam-hello -t 10000 --post-process-file /usr/share/rpi-camera-assets/hailo_yolov8_inference.json -v 2
```

第三個指令會印出物件偵測結果(`person 0.85` 之類),這步過了代表相機 + Hailo 都 OK。

**截圖存證**,Step 1 完成。

---

## Step 2 — 把程式碼放進 Pi

教授會給你一個 `speed-trap/` 資料夾(USB 或雲端連結)。用你自己的電腦把它傳進 Pi。三種方式挑一種:

### 方式 A:用 scp(最快,推薦)

在**你自己電腦**的 terminal:

```bash
# Mac / Linux
scp -r /path/to/speed-trap <username>@<pi-hostname>.local:~/

# Windows PowerShell(Win10 以上內建 scp)
scp -r C:\path\to\speed-trap <username>@<pi-hostname>.local:~/
```

`<username>` 和 `<pi-hostname>` 用你設定 Pi 時填的那組。系統會問 Pi 的密碼,打進去就開始傳。

### 方式 B:用 USB 隨身碟

1. 把 `speed-trap/` 複製到 USB
2. USB 插到 Pi
3. 在 Pi 上:
```bash
lsblk                                                       # 看 USB 掛載在哪
cp -r /media/<username>/<USB名稱>/speed-trap ~/              # 複製到 home
```

### 方式 C:用 WinSCP / FileZilla(圖形介面)

Windows 不想碰命令列的話:
- 裝 WinSCP <https://winscp.net/> 或 FileZilla
- 連線:hostname 填 `<pi-hostname>.local`,username / password 填 Pi 的那組
- 左邊拖 `speed-trap/` 到右邊的 `/home/<username>/`

### 確認進來了

SSH 進 Pi 上:

```bash
cd ~/speed-trap
ls
# 看到 apps/ config/ tests/ scripts/ requirements.txt 就 OK
```

### 之後拿到新版怎麼辦

教授改完程式碼會再給你一版新的,重複同樣步驟覆蓋舊的就行。若怕覆蓋掉你改過的 config,先備份:

```bash
cp ~/speed-trap/config/station_a.yaml ~/station_a.yaml.bak
```

傳完新版再把備份的 config 放回去。

---

## Step 3 — 建 Python 環境

**為什麼要搞這個**: Trixie 預設不給你直接 `pip install`,要用**虛擬環境 venv** 隔離專案套件。`--system-site-packages` 這個參數是讓 venv 看得到系統套件 — 因為 `picamera2` 只能用 `apt` 裝,不能用 pip。忘記加這個參數,後面跑程式會 `ImportError: picamera2`。

```bash
cd ~/speed-trap
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**驗證**:

```bash
python3 -c "import picamera2, cv2, yaml, numpy; print('OK')"
```

印出 `OK` 就行。

**之後每次開新的 SSH terminal,記得先啟用 venv**:

```bash
cd ~/speed-trap
source .venv/bin/activate
```

沒啟用的徵兆是 prompt 前面沒有 `(.venv)`,這時跑 `pytest` 或 `python3 -m apps.xxx` 都會找不到套件。

---

## Step 4 — 跑單元測試

```bash
cd ~/speed-trap
source .venv/bin/activate
pytest -v
```

全綠就過。有紅的**不要自己改程式碼**,截圖丟給 PC 團隊:

```bash
pytest -v --tb=long 2>&1 | tee /tmp/pytest-fail.log
```

---

## Step 5 — 用影片測邏輯(不用相機)

這步驗證 tracker / 觸發線 / 事件邏輯都對,**不用開真相機**,用預錄影片就能跑。

**1. 準備一段車流影片**(任何 mp4 都行):
- 自己手機拍 30 秒學校車道
- 網路下載任何行車記錄器影片
- 不要太大(<100MB 比較快)

**2. 把影片傳進 Pi**:用跟 Step 2 一樣的方式(scp / USB / WinSCP)。範例用 scp:

在 Pi 上先建好資料夾:
```bash
mkdir -p ~/speed-trap/samples
```

然後從你自己電腦:
```bash
scp traffic.mp4 <username>@speedtrap.local:~/speed-trap/samples/
```

**3. 跑 replay**:

```bash
cd ~/speed-trap
source .venv/bin/activate
python3 -m apps.replay_video \
  --video samples/traffic.mp4 \
  --no-display \
  --config config/station_a.yaml \
  2>&1 | tee /tmp/replay_$(date +%m%d_%H%M).log
```

`2>&1 | tee ...` 是把輸出同時印出來也存進 log 檔,之後才能交作業。

**4. 預期**: stdout 印出 JSON,每台車一行:

```json
{"station_id": "A", "track_id": 3, "label": "car", "timestamp_ns": 1735689600123456789, ...}
```

**5. 沒看到 JSON?** 排查順序:
- 影片裡車輛有從上往下穿越畫面中間嗎?(觸發線預設畫面 60% 高)
- 沒有穿越 → 改 `config/station_a.yaml` 裡 `trigger_line_y: 0.4` 再試
- 換一部車流比較明顯的影片
- 還是不行 → log 丟給教授

---

## Step 6 — 接真相機跑完整流程

最後一關 — 讓 Pi 從真相機抓影像跑完整 pipeline。

**1. 啟動**:

```bash
cd ~/speed-trap
source .venv/bin/activate
bash scripts/run_on_pi.sh 2>&1 | tee /tmp/run_$(date +%m%d_%H%M).log
```

**2. 三種測試情境**(由簡單到真實,都試試看):

- **玩具車**: Pi 放桌上相機朝外,同學拿玩具車從相機前走過
- **播影片**: 用手機或平板播行車記錄器影片,對著 Pi 相機(最方便)
- **真車**: 把 Pi 拿到校園車道邊試拍(背電池可 cordless 測 15 分鐘)

應該會看到 JSON 事件一個個印出。

**3. 觀察**:

- ✅ 有 JSON 事件印出
- ✅ 每台車**只觸發一次**(同一台觸發多次 = tracker 有 bug,回報教授)
- ✅ Ctrl+C 能乾淨停下來

**4. 停不下來時**:

```bash
sudo fuser -k /dev/hailo0
```

再不行就重開 Pi。

**5. 把 log 傳回你電腦保存**(交作業用):

```bash
# 在你自己電腦:
scp <username>@speedtrap.local:/tmp/run_*.log ./
```

或用 WinSCP / USB 都行。

---

## 驗收清單(週五跟教授報告用)

| Step | 證據 |
|------|------|
| 1. 硬體 | `hailortcli` 輸出截圖 + Hailo demo 畫面 |
| 3. 環境 | `pytest` 全綠截圖 |
| 5. Replay | `replay_video` 的 JSON 輸出 log |
| 6. 真相機 | `run_on_pi.sh` 的 JSON 輸出 log + 測試情境描述 |

直接 live demo + terminal 輸出給教授看就行,不用做 PPT。

---

## 常見問題

**`hailortcli` 找不到裝置**
排線鬆了或裝反。`dmesg | grep hailo` 有沒有 "Firmware was loaded"。沒有 → 關機重插 ribbon。

**`rpicam-hello` 說沒相機**
相機排線藍膠帶方向。兩端都朝 USB 側。

**`pip install picamera2` 失敗**
不要用 pip 裝 picamera2,用 `sudo apt install python3-picamera2`,venv 建立時加 `--system-site-packages`。

**主程式跑但沒事件**
按這個順序查:
1. `rpicam-hello` 看得到畫面?
2. `rpicam-hello --post-process-file .../hailo_yolov8_inference.json -v 2` 有 bbox 輸出?
3. `python3 -m apps.replay_video --video samples/xxx.mp4` 有 JSON?
4. `trigger_line_y` 是不是設太低,車輛 bbox 根本沒穿越?

**Ctrl+C 後重跑失敗**
```bash
sudo fuser -k /dev/hailo0
```
或重開 Pi。

**pytest 有失敗**
不要自己改程式碼,`pytest -v --tb=long 2>&1 | tee /tmp/fail.log` 把 log 丟給 PC 團隊。

---

## 回報問題用這個格式

```
在做什麼: [e.g. Step 6 跑 run_on_pi.sh]
指令: [完整指令]
預期: [該看到什麼]
實際: [看到什麼,附 log 或截圖]
試過: [你已經嘗試的解法]
```

卡超過 30 分鐘就問,不要自己耗。
