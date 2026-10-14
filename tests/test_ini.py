"""Tests for the INI reader."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.parser.errors import TrackParseError
from src.parser.ini import IniFile, read_text


def test_parse_sections_and_keys() -> None:
    ini = IniFile.from_text("[A]\nX=1\nY = hello \n\n[B]\nZ=2\n")
    assert ini.sections == {"A": {"X": "1", "Y": "hello"}, "B": {"Z": "2"}}
    assert ini.get("A", "X") == "1"
    assert ini.get("A", "Y") == "hello"
    assert ini.get("B", "missing") is None


def test_typed_getters() -> None:
    ini = IniFile.from_text("[P]\nF=1.25\nI=3\nB=1\nEMPTY=\n")
    assert ini.get_float("P", "F") == 1.25
    assert ini.get_int("P", "I") == 3
    assert ini.get_bool("P", "B") is True
    assert ini.get_float("P", "EMPTY", 9.0) == 9.0


def test_invalid_number_raises() -> None:
    ini = IniFile.from_text("[P]\nF=abc\n")
    with pytest.raises(TrackParseError, match="not a number"):
        ini.get_float("P", "F")


def test_require_missing_raises() -> None:
    ini = IniFile.from_text("[P]\nF=1\n")
    with pytest.raises(TrackParseError, match="missing"):
        ini.require("P", "G", path="track.ini")


def test_read_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "a.ini"
    path.write_bytes("\ufeff[A]\nK=1\n".encode())
    assert IniFile.read(path).get("A", "K") == "1"


def test_read_windows_1252(tmp_path: Path) -> None:
    path = tmp_path / "a.ini"
    path.write_bytes("[A]\nK=caf\xe9\n".encode("cp1252"))
    assert read_text(path) == "[A]\nK=caf\xe9\n"
