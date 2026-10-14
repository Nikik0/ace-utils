"""Minimal INI reader for Assetto Corsa track files.

AC1 ships ``.ini`` files as UTF-8 (often with a BOM) or Windows-1252. Sections
are ``[NAME]`` and entries are ``KEY=VALUE``; the first ``=`` splits key from
value. Blank lines are ignored. Values are returned raw (whitespace trimmed).
"""

from __future__ import annotations

from pathlib import Path

from src.parser.errors import TrackParseError


def read_text(path: str | Path) -> str:
    """Read an AC text file, trying UTF-8 (with BOM) then Windows-1252."""
    data = Path(path).read_bytes()
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise TrackParseError(path, "cannot decode file as UTF-8 or Windows-1252")


class IniFile:
    """A parsed INI file; sections and keys keep their original casing."""

    def __init__(self, sections: dict[str, dict[str, str]]) -> None:
        self.sections = sections

    @classmethod
    def from_text(cls, text: str) -> IniFile:
        sections: dict[str, dict[str, str]] = {}
        current: dict[str, str] | None = None
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[") and line.endswith("]"):
                name = line[1:-1].strip()
                current = sections.setdefault(name, {})
                continue
            if "=" not in line:
                continue
            if current is None:
                current = sections.setdefault("", {})
            key, _, value = line.partition("=")
            current[key.strip()] = value.strip()
        return cls(sections)

    @classmethod
    def read(cls, path: str | Path) -> IniFile:
        return cls.from_text(read_text(path))

    def get(self, section: str, key: str, default: str | None = None) -> str | None:
        return self.sections.get(section, {}).get(key, default)

    def require(self, section: str, key: str, path: str | Path = "<ini>") -> str:
        value = self.get(section, key)
        if value is None:
            raise TrackParseError(path, f"missing [{section}] {key}")
        return value

    def get_float(self, section: str, key: str, default: float = 0.0) -> float:
        value = self.get(section, key)
        if value is None or value == "":
            return default
        try:
            return float(value)
        except ValueError as exc:
            raise TrackParseError("<ini>", f"[{section}] {key} is not a number: {value!r}") from exc

    def get_int(self, section: str, key: str, default: int = 0) -> int:
        value = self.get(section, key)
        if value is None or value == "":
            return default
        try:
            return int(float(value))
        except ValueError as exc:
            raise TrackParseError("<ini>", f"[{section}] {key} is not an int: {value!r}") from exc

    def get_bool(self, section: str, key: str, default: bool = False) -> bool:
        value = self.get(section, key)
        if value is None or value == "":
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}
