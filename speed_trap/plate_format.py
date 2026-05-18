"""Taiwan license plate format validation + normalization.

Taiwan plates do not use the letters I and O (to avoid confusion with digits
1 and 0), so the regex character classes exclude them.

Supported formats:
- Current 4-digit ABC-1234 (most common since 2014)
- Old style 1234-AB
- Motorcycle AB-1234
- Motorcycle ABC-123

We accept input with or without hyphens / spaces. Use ``normalize_plate`` to
canonicalize before equality comparisons.
"""

from __future__ import annotations

import re

_LETTER = "[A-HJ-NP-Z]"  # Taiwan plates: no I, no O

_PATTERNS = tuple(
    re.compile(pat)
    for pat in [
        rf"^{_LETTER}{{3}}\d{{4}}$",  # ABC1234
        rf"^\d{{4}}{_LETTER}{{2}}$",  # 1234AB
        rf"^{_LETTER}{{2}}\d{{4}}$",  # AB1234 (motorcycle)
        rf"^{_LETTER}{{3}}\d{{3}}$",  # ABC123  (motorcycle)
    ]
)


def normalize_plate(text: str) -> str:
    """Strip whitespace + hyphens, uppercase. Use before pattern matching."""
    return text.strip().upper().replace(" ", "").replace("-", "")


def is_valid_taiwan_plate(text: str) -> bool:
    """Return True iff ``text`` matches any Taiwan plate format."""
    if not text:
        return False
    cleaned = normalize_plate(text)
    return any(p.match(cleaned) for p in _PATTERNS)
