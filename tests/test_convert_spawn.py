"""Tests for marker -> ACE spawn-point scene conversion."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest
from src.ace_formats import new_message
from src.convert.spawn import (
    TYPE_GRID,
    TYPE_HOTLAP,
    TYPE_PIT,
    build_spawn_scene,
    collect_markers,
    select_markers,
    spawn_scenes,
)
from src.parser.kn5_model import Kn5Model, Kn5Node, Kn5NodeClass

_REAL = Path("test_data/ac1/ek_tsubaki_line")
_REF_GRID = Path("test_data/ace_ref/content/tracks/ek_tsubak_hv/containers/spawnpoints_grid.scene")

_IDENTITY = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0)


def _translation(x: float, y: float, z: float, yaw: float = 0.0) -> tuple[float, ...]:
    cos, sin = math.cos(yaw), math.sin(yaw)
    # row-major Y rotation with translation in elements 12..14
    return (cos, 0.0, -sin, 0.0, 0.0, 1.0, 0.0, 0.0, sin, 0.0, cos, 0.0, x, y, z, 1.0)


def _node(name: str, transform: tuple[float, ...]) -> Kn5Node:
    return Kn5Node(node_class=Kn5NodeClass.BASE, name=name, active=True, transform=transform)


def _model() -> Kn5Model:
    root = Kn5Node(
        node_class=Kn5NodeClass.BASE,
        name="root",
        active=True,
        transform=_IDENTITY,
        children=(
            _node("AC_START_0", _translation(1.0, 2.0, 3.0)),
            _node("AC_START_1", _translation(4.0, 5.0, 6.0)),
            _node("AC_PIT_0", _translation(7.0, 8.0, 9.0)),
            _node("AC_HOTLAP_START_0", _translation(10.0, 11.0, 12.0)),
        ),
    )
    return Kn5Model(version=6, extra=0, textures=(), materials=(), root=root)


def _decode(data: bytes) -> Any:
    msg = new_message(".scene")
    msg.ParseFromString(data)
    return msg


def test_collect_and_select_markers() -> None:
    markers = collect_markers(_model())
    expected = ["AC_START_0", "AC_START_1", "AC_PIT_0", "AC_HOTLAP_START_0"]
    assert [m.name for m in markers] == expected
    starts = select_markers(markers, "AC_START_")
    assert [m.name for m in starts] == ["AC_START_0", "AC_START_1"]
    assert starts[1].position == (4.0, 5.0, 6.0)


def test_marker_yaw() -> None:
    model = Kn5Model(
        version=6,
        extra=0,
        textures=(),
        materials=(),
        root=Kn5Node(
            node_class=Kn5NodeClass.BASE,
            name="root",
            active=True,
            transform=_IDENTITY,
            children=(_node("AC_START_0", _translation(0.0, 0.0, 0.0, math.radians(45.0))),),
        ),
    )
    marker = collect_markers(model)[0]
    assert marker.yaw_degrees == pytest.approx(45.0, abs=1e-4)


def test_build_spawn_scene() -> None:
    markers = select_markers(collect_markers(_model()), "AC_START_")
    message = build_spawn_scene(markers, type_code=TYPE_GRID, name_prefix="starting_grid")
    scene = _decode(message.SerializeToString())
    assert [a.name for a in scene.scene_actors] == ["starting_grid_0", "starting_grid_1"]
    actor = scene.scene_actors[1]
    assert actor.actor.WhichOneof("actorType") == "starting_position"
    assert list(actor.actor.starting_position.types) == [TYPE_GRID]
    assert actor.actor.starting_position.slot_number == 1
    assert actor.transform.position.x == pytest.approx(4.0)
    assert list(actor.tags) == ["Start Pos"]


def test_spawn_scenes_split() -> None:
    scenes = spawn_scenes(_model())
    assert len(scenes["grid"].scene_actors) == 2
    assert len(scenes["pitlane"].scene_actors) == 1
    assert len(scenes["hotlap"].scene_actors) == 1
    assert scenes["hotlap"].scene_actors[0].actor.starting_position.types[0] == TYPE_HOTLAP
    assert scenes["pitlane"].scene_actors[0].actor.starting_position.types[0] == TYPE_PIT


@pytest.mark.skipif(not (_REAL / "real_downhill.kn5").is_file(), reason="sample track missing")
def test_real_grid_matches_reference() -> None:
    model = Kn5Model.read(_REAL / "real_downhill.kn5", load_textures=False)
    scene = spawn_scenes(model)["grid"]
    reference = _decode(_REF_GRID.read_bytes())
    assert len(scene.scene_actors) == len(reference.scene_actors)
    for actual, expected in zip(scene.scene_actors, reference.scene_actors, strict=False):
        point = expected.transform.position
        assert actual.transform.position.x == pytest.approx(point.x, abs=1e-3)
        assert actual.transform.position.z == pytest.approx(point.z, abs=1e-3)
        assert actual.transform.rotation.y == pytest.approx(
            expected.transform.rotation.y, abs=0.05
        )
