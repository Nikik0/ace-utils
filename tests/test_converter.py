"""Integration test for the end-to-end converter pipeline."""

from __future__ import annotations

import struct
from pathlib import Path

from src.converter import TrackConverter
from src.packer.kspkg import KspkgReader

_SURFACES = (
    "[SURFACE_0]\nKEY=ROAD\nFRICTION=1.0\nIS_VALID_TRACK=1\n"
    "[SURFACE_1]\nKEY=GRASS\nFRICTION=0.7\nIS_VALID_TRACK=0\n"
)
_MAP = "[PARAMETERS]\nWIDTH=1600\nHEIGHT=1866\nX_OFFSET=1\nZ_OFFSET=2\nMARGIN=10\n"
_HINTS = "[HINT_0]\nSTART=0\nEND=1\nVALUE=1\n"
_MODELS = "[MODEL_0]\nFILE=missing.kn5\nPOSITION=0,0,0\nROTATION=0,0,0\n"
_UI = '{"name": "Test Track", "length": "1km", "tags": ["test"]}'


def _fast_lane() -> bytes:
    points = [((0.0, 0.0, 0.0), 0.0, 0), ((1.0, 0.0, 0.0), 1.0, 1), ((2.0, 0.0, 0.0), 2.0, 2)]
    data = struct.pack("<4i", 7, len(points), 0, 0)
    for position, length, point_id in points:
        data += struct.pack("<3ffi", *position, length, point_id)
    data += struct.pack("<i", len(points))
    for index in range(len(points)):
        data += struct.pack("<18f", *([float(index)] * 18))
    data += struct.pack("<i", 0)
    return data


def _make_track(root: Path) -> None:
    (root / "downhill" / "data").mkdir(parents=True)
    (root / "downhill" / "ai").mkdir()
    (root / "ui" / "downhill").mkdir(parents=True)
    (root / "downhill" / "data" / "surfaces.ini").write_text(_SURFACES)
    (root / "downhill" / "data" / "map.ini").write_text(_MAP)
    (root / "downhill" / "data" / "ai_hints.ini").write_text(_HINTS)
    (root / "downhill" / "ai" / "fast_lane.ai").write_bytes(_fast_lane())
    (root / "models_downhill.ini").write_text(_MODELS)
    (root / "ui" / "downhill" / "ui_track.json").write_text(_UI)


def test_converter_end_to_end(tmp_path: Path) -> None:
    ac1 = tmp_path / "mytrack"
    _make_track(ac1)
    build = tmp_path / "build"
    kspkg = tmp_path / "mytrack.kspkg"

    result = TrackConverter(ac1, output_dir=build, kspkg_path=kspkg).run()

    assert result.track_id == "mytrack"
    assert result.layout == "downhill"
    assert (build / "content/tracks/mytrack/mytrack.track").is_file()
    assert (build / "content/tracks/mytrack/mytrack.scene").is_file()
    assert (build / "content/tracks/mytrack/containers/layout_downhill.scene").is_file()
    assert (build / "content/tracks/mytrack/textures/NULL.texture").is_file()
    assert (
        build / "content/tracks/mytrack/layouts/layout_downhill.ideal_line.aisplinedata"
    ).is_file()
    assert kspkg.is_file()

    with KspkgReader(kspkg) as reader:
        paths = {entry.path for entry in reader.entries}
    assert r"content\tracks\mytrack\mytrack.track" in paths
    assert r"content\tracks\mytrack\layouts\layout_downhill.ideal_line.aisplinedata" in paths
