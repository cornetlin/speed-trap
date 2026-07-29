"""在高頻迴圈裡記錄反覆發生的失敗,而不把 log 洗掉。

用在每幀都會執行的程式碼:例外不能讓它往上炸(pipeline 中途死掉更糟),
但也絕對不能靜默吞掉 —— 失敗時退回的預設值會讓系統看起來正常運作,實際
上某一項功能已經失效。折衷是第一次一定記,之後每 N 次記一則,並附上累計
次數,讓「偶爾一次」和「一直在錯」看得出差別。
"""

from __future__ import annotations

import logging


class ThrottledWarning:
    """同一個錯誤第一次一定記,之後每 ``every`` 次記一則。"""

    def __init__(
        self, logger: logging.Logger, *, every: int = 100
    ) -> None:
        self._logger = logger
        self._every = max(1, every)
        self._count = 0

    @property
    def count(self) -> int:
        """至今發生過幾次(不是記了幾則 log)。"""
        return self._count

    def warn(self, message: str, *args: object) -> None:
        self._count += 1
        if self._count == 1 or self._count % self._every == 0:
            self._logger.warning(
                message + " (第 %d 次)", *args, self._count
            )
