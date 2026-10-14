"""Tests for the UI map, track-layout binary and timelines zone converters."""

from __future__ import annotations

import json
import struct
from typing import Any

import pytest
from src.ace_formats import new_message
from src.convert.spawn import Marker
from src.convert.track_layout import build_track_layout
from src.convert.ui import build_map_svg, build_previews_json
from src.convert.zones import build_timelines_scene
from src.parser.ai_spline import AiPoint, AiSpline


def _spline() -> AiSpline:
    points = tuple(
        AiPoint(position=(float(i), 1.0, float(i * 2)), length=float(i), id=i) for i in range(30)
    )
    return AiSpline(version=7, lap_time=0, sample_count=0, points=points, extra=(), grid=None)


def test_build_map_svg() -> None:
    svg, layout_image = build_map_svg(_spline(), size=256)
    assert svg.startswith(b"<svg")
    assert b"<path" in svg
    assert layout_image["k"] > 0
    assert set(layout_image) == {"margin", "minX", "minY", "k", "offsetX", "offsetY"}


def test_build_map_svg_empty() -> None:
    empty = AiSpline(version=7, lap_time=0, sample_count=0, points=(), extra=(), grid=None)
    svg, layout_image = build_map_svg(empty)
    assert svg.startswith(b"<svg")
    assert layout_image["k"] == pytest.approx(1.0)


def test_build_previews_json() -> None:
    document = json.loads(build_previews_json("mytrack", "downhill", layout_image={"k": 2.0}))
    assert document["trackId"] == "mytrack"
    assert document["layout"] == "downhill"
    assert document["svg"] is True
    assert document["layoutImage"]["k"] == 2.0


def test_build_track_layout_binary() -> None:
    data = build_track_layout(_spline(), points=8)
    version = struct.unpack("<i", data[:4])[0]
    start = struct.unpack("<3f", data[4:16])
    count = struct.unpack("<i", data[32:36])[0]
    assert version == 1
    assert start == (0.0, 1.0, 0.0)
    assert count == 8
    assert len(data) == 36 + count * 16


def test_build_timelines_scene() -> None:
    markers = [
        Marker("AC_AB_START_L", (0.0, 0.0, 0.0), 0.0),
        Marker("AC_AB_START_R", (10.0, 0.0, 0.0), 0.0),
        Marker("AC_AB_FINISH_L", (0.0, 0.0, 900.0), 0.0),
        Marker("AC_AB_FINISH_R", (10.0, 0.0, 900.0), 0.0),
    ]
    scene = build_timelines_scene(markers, "timelines_downhill")
    assert scene is not None
    msg = new_message(".scene")
    msg.ParseFromString(scene.SerializeToString())
    actor = msg.scene_actors[0]
    assert actor.name == "timelines_downhill"
    assert list(actor.tags) == ["Zone"]
    zones = actor.actor.zones
    assert len(zones.data.geometrics) == 2
    assert len(zones.data.instances) == 1
    assert zones.data.geometrics[0].color.x == pytest.approx(1.0)
    assert zones.data.instances[0].WhichOneof("type") == "timelines"


def test_build_timelines_scene_none_without_markers() -> None:
    assert build_timelines_scene([Marker("AC_START_0", (0.0, 0.0, 0.0), 0.0)], "x") is None


def test_zone_geometric_has_four_points() -> None:
    markers = [
        Marker("AC_AB_START_L", (0.0, 0.0, 0.0), 0.0),
        Marker("AC_AB_START_R", (10.0, 0.0, 0.0), 0.0),
    ]
    scene = build_timelines_scene(markers, "t")
    assert scene is not None
    msg = new_message(".scene")
    msg.ParseFromString(scene.SerializeToString())
    geometry: Any = msg.scene_actors[0].actor.zones.data.geometrics[0]
    assert len(geometry.points) == 4
    assert geometry.renderHeight == pytest.approx(0.8)
