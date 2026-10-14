"""Tests for the KN5 metadata reader."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from src.parser.errors import TrackParseError
from src.parser.kn5 import KN5_MAGIC, Kn5Metadata

_REAL = Path("test_data/ac1/ek_tsubaki_line/ek_tsubaki_line.kn5")


def _build(
    *, textures: list[tuple[str, bytes]], materials: list[tuple[str, str]], geometry: bytes
) -> bytes:
    data = KN5_MAGIC + struct.pack("<2i", 6, 0)
    data += struct.pack("<i", len(textures))
    for name, payload in textures:
        encoded = name.encode()
        data += struct.pack("<i", 1) + struct.pack("<i", len(encoded)) + encoded
        data += struct.pack("<I", len(payload)) + payload
    data += struct.pack("<i", len(materials))
    for name, shader in materials:
        for text in (name, shader):
            encoded = text.encode()
            data += struct.pack("<i", len(encoded)) + encoded
        data += b"\x00" * 6 + struct.pack("<2i", 0, 0)
    return data + geometry


def test_parse_synthetic(tmp_path: Path) -> None:
    path = tmp_path / "m.kn5"
    path.write_bytes(
        _build(
            textures=[("a.dds", b"DATA")],
            materials=[("ROAD", "ksPerPixel")],
            geometry=b"\x01\x02\x03",
        )
    )
    meta = Kn5Metadata.read(path)
    assert meta.version == 6
    assert meta.textures[0].name == "a.dds"
    assert meta.textures[0].length == 4
    assert meta.textures[0].active
    assert meta.materials[0].name == "ROAD"
    assert meta.materials[0].shader == "ksPerPixel"
    assert meta.geometry_bytes == 3
    assert meta.has_geometry


def test_bad_magic_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.kn5"
    path.write_bytes(b"not-a-kn5-file!!")
    with pytest.raises(TrackParseError, match="magic"):
        Kn5Metadata.read(path)


@pytest.mark.skipif(not _REAL.is_file(), reason="AC1 sample not extracted")
def test_read_real_kn5() -> None:
    meta = Kn5Metadata.read(_REAL)
    assert meta.version == 6
    assert len(meta.textures) == 34
    assert len(meta.materials) == 47
    assert meta.materials[0].name == "MIRRORS"
    assert meta.has_geometry
