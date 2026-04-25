# speed-trap

區間測速邊緣站軟體 — Raspberry Pi 5 + Hailo AI HAT+。

開發與架構說明見 [CLAUDE.md](CLAUDE.md)。

## Sample video

`samples/test_traffic.mp4` 用於 PC 上跑 `apps/replay_video.py` 驗證 pipeline。
這支影片由以下公共領域素材切前 30 秒而成,並非 repo 的一部分(`samples/*.mp4` 已 gitignore),
請各自下載重現:

- **來源**: [Why Do School Buses Still Look The Same — Internet Archive](https://archive.org/details/why-do-school-buses-still-look-the-same)
- **License**: [CC0 1.0 Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/) — 任意使用,無需署名
- **規格**: 640×360, ~30 fps, 30 秒, ~620 KB

重新下載 + 切片:

```bash
curl -sL -o /tmp/_full.mp4 \
  "https://archive.org/download/why-do-school-buses-still-look-the-same/Why%20Do%20School%20Buses%20Still%20Look%20The%20Same_.mp4"
ffmpeg -y -ss 0 -t 30 -i /tmp/_full.mp4 -c:v copy -an samples/test_traffic.mp4
rm /tmp/_full.mp4
```

`MockDetector` 不看影片內容(它產生的是固定軌跡的假車),所以任何 mp4 都能跑;
這支只是給人看一個會動的背景而已。
