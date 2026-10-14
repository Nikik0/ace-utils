"""Assetto Corsa 1 track parsing errors."""

from __future__ import annotations

from pathlib import Path


class TrackParseError(Exception):
    """Raised when an AC1 track or one of its files cannot be parsed."""

    def __init__(self, path: str | Path, message: str) -> None:
        self.path = str(path)
        super().__init__(f"{message}: {self.path}")
