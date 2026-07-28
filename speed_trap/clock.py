"""時間來源:單調時鐘與真實時鐘分開使用。

兩種時鐘各有各的正確用途,不能互相取代:

* **單調時鐘**(``time.monotonic_ns``)—— 只保證往前走,不受 NTP 校正影響。
  量「經過多久」一律用它:重複判定視窗、prune_stale、track_idle_timeout、
  幀率統計。若改用真實時鐘,NTP 往回校正時間隔會算短、甚至變成負數。
* **真實時鐘**(``CLOCK_REALTIME``)—— UTC epoch,會被 NTP 校正。凡是要
  對外的時間戳一律用它:通行事件的時間、CSV 的擷取時間。單調時鐘的起點
  是各台機器自己的開機時刻,重開機後前後兩批資料對不起來,A、B 兩站相減
  算速度也沒有意義。

CNMV 205 要求內建時鐘與國家時間標準一致,所以站台啟動時會檢查並記錄
NTP 對時狀態(見 :func:`ntp_status`)。
"""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone

_logger = logging.getLogger(__name__)

_NTP_QUERY = ("timedatectl", "show", "-p", "NTPSynchronized")
_NTP_TIMEOUT_S = 5.0


def wall_clock_ns() -> int:
    """真實時間,UTC epoch 奈秒。

    Pi(Linux)走 CLOCK_REALTIME。開發用的 Windows 沒有 clock_gettime_ns,
    退回 time_ns():語意相同(同樣是 epoch 真實時間),只是解析度較粗。
    """
    getter = getattr(time, "clock_gettime_ns", None)
    if getter is not None:
        return getter(time.CLOCK_REALTIME)
    return time.time_ns()


def wall_ns_to_iso(wall_ns: int) -> str:
    """epoch 奈秒 → 本地時區的 ISO 8601 字串(到毫秒)。0 回傳空字串。"""
    if not wall_ns:
        return ""
    seconds, remainder = divmod(int(wall_ns), 1_000_000_000)
    moment = datetime.fromtimestamp(seconds, tz=timezone.utc).astimezone()
    moment = moment.replace(microsecond=remainder // 1000)
    return moment.isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class NtpStatus:
    synchronized: bool | None   # None = 問不到(非 systemd 環境、指令不存在)
    detail: str

    def __str__(self) -> str:
        return self.detail


def ntp_status() -> NtpStatus:
    """問 systemd-timesyncd 目前有沒有對到時。"""
    try:
        proc = subprocess.run(
            _NTP_QUERY,
            capture_output=True,
            text=True,
            timeout=_NTP_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return NtpStatus(None, f"unknown ({type(exc).__name__}: {exc})")

    if proc.returncode != 0:
        stderr = proc.stderr.strip() or f"exit {proc.returncode}"
        return NtpStatus(None, f"unknown (timedatectl: {stderr})")

    raw = proc.stdout.strip()          # 形如 "NTPSynchronized=yes"
    value = raw.split("=", 1)[-1].strip().lower() if "=" in raw else raw.lower()
    if value in ("yes", "true", "1"):
        return NtpStatus(True, "synchronized")
    if value in ("no", "false", "0"):
        return NtpStatus(False, "NOT synchronized")
    return NtpStatus(None, f"unknown ({raw!r})")


def log_clock_status() -> NtpStatus:
    """啟動時記錄一次時鐘狀態。沒對到時要顯眼,時間戳的可信度靠它。"""
    status = ntp_status()
    _logger.info(
        "system clock: %s | NTP: %s",
        wall_ns_to_iso(wall_clock_ns()),
        status.detail,
    )
    if status.synchronized is False:
        _logger.warning(
            "系統時鐘未與 NTP 同步 —— 通行時間戳不可信,跨站速度計算會出錯。"
            "CNMV 205 要求內建時鐘與國家時間標準一致,請先確認對時"
            "(timedatectl set-ntp true)"
        )
    elif status.synchronized is None:
        _logger.warning(
            "無法確認 NTP 對時狀態(%s)—— 請自行確認系統時間正確",
            status.detail,
        )
    return status
