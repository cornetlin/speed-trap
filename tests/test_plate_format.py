from __future__ import annotations

import pytest

from speed_trap.plate_format import is_valid_taiwan_plate, normalize_plate


def test_normalize_strips_whitespace_and_hyphens() -> None:
    assert normalize_plate(" ABC-1234 ") == "ABC1234"
    assert normalize_plate("abc-1234") == "ABC1234"
    assert normalize_plate("  AB - 1234  ") == "AB1234"


def test_normalize_empty() -> None:
    assert normalize_plate("") == ""
    assert normalize_plate("   ") == ""


def test_valid_current_format_abc1234() -> None:
    assert is_valid_taiwan_plate("ABC-1234")
    assert is_valid_taiwan_plate("ABC1234")
    assert is_valid_taiwan_plate("abc-1234")
    assert is_valid_taiwan_plate(" ABC-1234 ")


def test_valid_old_format_1234ab() -> None:
    assert is_valid_taiwan_plate("1234-AB")
    assert is_valid_taiwan_plate("1234AB")


def test_valid_motorcycle_ab1234() -> None:
    assert is_valid_taiwan_plate("AB-1234")
    assert is_valid_taiwan_plate("AB1234")


def test_valid_motorcycle_abc123() -> None:
    assert is_valid_taiwan_plate("ABC-123")
    assert is_valid_taiwan_plate("ABC123")


def test_reject_letters_i_and_o() -> None:
    # Taiwan plates exclude I and O to avoid 1/0 confusion
    assert not is_valid_taiwan_plate("AIC-1234")
    assert not is_valid_taiwan_plate("AOC-1234")
    assert not is_valid_taiwan_plate("IBC-1234")


def test_reject_too_short() -> None:
    assert not is_valid_taiwan_plate("AB-123")
    assert not is_valid_taiwan_plate("A-1234")


def test_reject_too_long() -> None:
    assert not is_valid_taiwan_plate("ABCD-1234")
    assert not is_valid_taiwan_plate("ABC-12345")


def test_reject_empty_or_none_like() -> None:
    assert not is_valid_taiwan_plate("")
    assert not is_valid_taiwan_plate("   ")
    assert not is_valid_taiwan_plate("---")


def test_reject_pure_numbers() -> None:
    assert not is_valid_taiwan_plate("12345678")
    assert not is_valid_taiwan_plate("1234")


def test_reject_pure_letters() -> None:
    assert not is_valid_taiwan_plate("ABCDEFG")


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("abc-1234", True),
        ("ABC-1234", True),
        ("1234-AB", True),
        ("AB-1234", True),
        ("ABC-123", True),
        ("XYZ-9999", True),
        ("ABC1234", True),  # no hyphen
        ("IIC-1234", False),  # has I
        ("OOC-1234", False),  # has O
        ("12-3456", False),
        ("", False),
    ],
)
def test_table(raw: str, expected: bool) -> None:
    assert is_valid_taiwan_plate(raw) is expected
