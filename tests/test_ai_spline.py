"""Tests for the AC1 fast_lane.ai AI spline parser."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from src.parser.ai_spline import AiSpline
from src.parser.errors import TrackParseError

_REAL = Path("test_data/ac1/ek_tsubaki_line/downhill_real/ai/fast_lane.ai")


def _build(*, points: list[tuple[tuple[float, float, float], float, int]], grid: bool) -> bytes:
    data = struct.pack("<4i", 7, len(points), 4242, 0)
    for position, length, point_id in points:
        data += struct.pack("<3ffi", *position, length, point_id)
    data += struct.pack("<i", len(points))
    for index in range(len(points)):
        data += struct.pack("<18f", *([float(index)] * 18))
    data += struct.pack("<i", 1 if grid else 0)
    if grid:
        data += struct.pack("<3f", 10.0, 20.0, 30.0)
        data += struct.pack("<3f", 0.0, 0.0, 0.0)
        data += struct.pack("<if", 3, 0.5)
        data += struct.pack("<i", 1)  # 1 item
        data += struct.pack("<i", 2)  # 2 subs
        data += struct.pack("<i", 2) + struct.pack("<2i", 7, 8)
        data += struct.pack("<i", 1) + struct.pack("<i", 9)
    return data


def test_parse_synthetic_without_grid(tmp_path: Path) -> None:
    path = tmp_path / "fast_lane.ai"
    path.write_bytes(
        _build(points=[((0.0, 0.0, 0.0), 0.0, 0), ((1.0, 2.0, 3.0), 1.5, 1)], grid=False)
    )
    spline = AiSpline.read(path)
    assert spline.version == 7
    assert spline.lap_time == 4242
    assert [p.id for p in spline.points] == [0, 1]
    assert spline.points[1].position == (1.0, 2.0, 3.0)
    assert spline.points[1].length == pytest.approx(1.5)
    assert len(spline.extra) == 2
    assert spline.extra[1].speed == pytest.approx(1.0)
    assert spline.grid is None


def test_parse_synthetic_with_grid(tmp_path: Path) -> None:
    path = tmp_path / "fast_lane.ai"
    path.write_bytes(_build(points=[((0.0, 0.0, 0.0), 0.0, 0)], grid=True))
    spline = AiSpline.read(path)
    assert spline.grid is not None
    assert spline.grid.neighbors_considered == 3
    assert spline.grid.items == (((7, 8), (9,)),)


def test_trailing_bytes_raise(tmp_path: Path) -> None:
    path = tmp_path / "fast_lane.ai"
    path.write_bytes(_build(points=[((0.0, 0.0, 0.0), 0.0, 0)], grid=False) + b"\x00\x00")
    with pytest.raises(TrackParseError, match="trailing"):
        AiSpline.read(path)


def test_bad_version_raises(tmp_path: Path) -> None:
    path = tmp_path / "fast_lane.ai"
    path.write_bytes(struct.pack("<4i", 6, 0, 0, 0))
    with pytest.raises(TrackParseError, match="version"):
        AiSpline.read(path)


@pytest.mark.skipif(not _REAL.is_file(), reason="AC1 sample not extracted")
def test_read_real_spline() -> None:
    spline = AiSpline.read(_REAL)
    assert len(spline.points) == 4862
    assert len(spline.extra) == 4862
    assert spline.grid is not None
    assert spline.positions[0] == pytest.approx((294.0185, 131.8292, -1706.04), abs=1e-3)
