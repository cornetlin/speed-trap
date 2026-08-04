"""Helper scripts.

有這個檔案是為了讓 tests/ 能 import scripts.ocr_sweep 裡的純邏輯
(編輯距離、統計彙整),腳本本身照樣可以 `python scripts/xxx.py` 直接跑。
pyproject 的 packages.find 已排除 scripts*,不會被打包進 wheel。
"""
