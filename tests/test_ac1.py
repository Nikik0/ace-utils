"""Tests for the AC1 track folder parser."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from src.parser.ac1 import parse_track
from src.parser.errors import TrackParseError

_REAL = Path("test_data/ac1/ek_tsubaki_line")

_SURFACES = (
    "[SURFACE_0]\nKEY=ROAD\nFRICTION=1.0\nIS_VALID_TRACK=1\n\n"
    "[SURFACE_1]\nKEY=GRASS\nFRICTION=0.7\n"
)
_MAP = "[PARAMETERS]\nWIDTH=1600\nHEIGHT=1866\nX_OFFSET=185.8\nZ_OFFSET=1716.3\nMARGIN=10.1\n"
_HINTS = "[HINT_0]\nSTART=0\nEND=1\nVALUE=1\n"
_MODELS = "[MODEL_0]\nFILE=model.kn5\nPOSITION=1,2,3\nROTATION=0,0,0\n"
_UI = '{"name": "Test Track", "tags": ["touge"], "author": "me", "year": 2022}'


def _make_track(root: Path, layouts: list[str]) -> None:
    for name in layouts:
        data_dir = root / name / "data"
        data_dir.mkdir(parents=True)
        (data_dir / "surfaces.ini").write_text(_SURFACES)
        (data_dir / "map.ini").write_text(_MAP)
        (data_dir / "ai_hints.ini").write_text(_HINTS)
        ai_dir = root / name / "ai"
        ai_dir.mkdir()
        ai_dir.joinpath("fast_lane.ai").write_bytes(
            struct.pack("<4i", 7, 0, 0, 0) + struct.pack("<i", 0) + struct.pack("<i", 0)
        )
        (root / f"models_{name}.ini").write_text(_MODELS)
        ui_dir = root / "ui" / name
        ui_dir.mkdir(parents=True)
        (ui_dir / "ui_track.json").write_text(_UI)
    (root / "collision.kn5").write_bytes(b"sc6969")


def test_parse_synthetic_track(tmp_path: Path) -> None:
    root = tmp_path / "mytrack"
    _make_track(root, ["layoutA", "layoutB"])
    track = parse_track(root)
    assert track.id == "mytrack"
    assert set(track.layouts) == {"layoutA", "layoutB"}
    assert track.collision == root / "collision.kn5"

    layout = track.layouts["layoutA"]
    assert [s.key for s in layout.surfaces] == ["ROAD", "GRASS"]
    assert layout.surfaces[1].friction == pytest.approx(0.7)
    assert layout.map_parameters is not None
    assert layout.map_parameters.width == 1600
    assert layout.models[0].file == "model.kn5"
    assert layout.models[0].position == (1.0, 2.0, 3.0)
    assert layout.ui is not None
    assert layout.ui.name == "Test Track"
    assert layout.ui.year == 2022


def test_no_layouts_raises(tmp_path: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(TrackParseError, match="no layouts"):
        parse_track(root)


def test_missing_folder_raises(tmp_path: Path) -> None:
    with pytest.raises(TrackParseError, match="not found"):
        parse_track(tmp_path / "nope")


@pytest.mark.skipif(not _REAL.is_dir(), reason="AC1 sample not extracted")
def test_parse_real_track() -> None:
    track = parse_track(_REAL)
    assert track.id == "ek_tsubaki_line"
    assert set(track.layouts) == {
        "downhill_idas",
        "downhill_real",
        "freeroam",
        "uphill_idas",
        "uphill_real",
    }
    real = track.layouts["downhill_real"]
    assert [s.key for s in real.surfaces] == ["ROAD", "PITLANE", "GRASS", "RUMBLE", "GRAVEL"]
    assert real.fast_lane is not None
    assert len(real.fast_lane.points) == 4862
    assert real.ui is not None
    assert real.ui.name == "EK Tsubaki Line Real Downhill"
