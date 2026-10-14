"""Tests for KN5 geometry conversion into MeshData / SceneData."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from src.ace_formats import new_message
from src.convert.mesh import (
    IDENTITY,
    GroupedMesh,
    build_mesh_data,
    group_by_material,
    iter_world_meshes,
    mat_mul,
    transform_normal,
    transform_point,
)
from src.convert.scene import StaticMeshEntry, build_static_mesh_scene, guid_for
from src.parser.kn5_model import Kn5Model, Kn5Node, Kn5NodeClass, Kn5Vertex

_REAL = Path("test_data/ac1/ek_tsubaki_line")


def _vertex(x: float, y: float, z: float) -> Kn5Vertex:
    return Kn5Vertex(position=(x, y, z), normal=(0.0, 0.0, 1.0), uv=(x, y), tangent=(1.0, 0.0, 0.0))


def _model() -> Kn5Model:
    mesh_a = Kn5Node(
        node_class=Kn5NodeClass.MESH,
        name="A",
        active=True,
        vertices=(_vertex(0, 0, 0), _vertex(1, 0, 0), _vertex(0, 1, 0)),
        indices=(0, 1, 2),
        material_id=0,
        cast_shadows=True,
        is_visible=True,
        is_renderable=True,
        lod_out=250.0,
    )
    mesh_b = Kn5Node(
        node_class=Kn5NodeClass.MESH,
        name="B",
        active=True,
        vertices=(_vertex(2, 0, 0), _vertex(3, 0, 0), _vertex(2, 1, 0)),
        indices=(0, 1, 2),
        material_id=0,
        is_visible=True,
        is_renderable=True,
        lod_out=100.0,
    )
    # base node translated by (10, 0, 0)
    translated = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 0.0, 1.0)
    group = Kn5Node(
        node_class=Kn5NodeClass.BASE,
        name="group",
        active=True,
        transform=translated,
        children=(mesh_a, mesh_b),
    )
    root = Kn5Node(
        node_class=Kn5NodeClass.BASE,
        name="root",
        active=True,
        transform=IDENTITY,
        children=(group,),
    )
    return Kn5Model(version=6, extra=0, textures=(), materials=(), root=root)


def test_transform_point_translation() -> None:
    matrix = (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 5.0, 6.0, 7.0, 1.0)
    assert transform_point(matrix, (1.0, 2.0, 3.0)) == (6.0, 8.0, 10.0)


def test_transform_normal_ignores_translation_and_normalizes() -> None:
    matrix = (2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 9.0, 9.0, 9.0, 1.0)
    assert transform_normal(matrix, (1.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)


def test_mat_mul_identity() -> None:
    assert mat_mul(IDENTITY, IDENTITY) == IDENTITY


def test_iter_world_meshes_bakes_transform() -> None:
    meshes = iter_world_meshes(_model())
    assert [m.name for m in meshes] == ["A", "B"]
    assert meshes[0].vertices[1].position == (11.0, 0.0, 0.0)
    assert meshes[1].vertices[0].position == (12.0, 0.0, 0.0)


def test_group_by_material_merges() -> None:
    groups = group_by_material(iter_world_meshes(_model()))
    assert len(groups) == 1
    assert groups[0].material == "MATERIAL_0"
    assert len(groups[0].parts) == 2
    assert groups[0].cast_shadows is True
    assert groups[0].lod_out == 250.0


def _decode(extension: str, data: bytes) -> Any:
    msg = new_message(extension)
    msg.ParseFromString(data)
    return msg


def test_build_mesh_data() -> None:
    group = group_by_material(iter_world_meshes(_model()))[0]
    data = build_mesh_data(group, "content/tracks/t/materials/m.material").SerializeToString()
    mesh = _decode(".mesh", data)
    assert mesh.isVisible is True
    assert mesh.lodOut == pytest.approx(250.0)
    lod = mesh.lods[0]
    assert len(lod.batches) == 1
    assert lod.batches[0].startIndex == 0
    assert lod.batches[0].indexCount == 6
    assert list(lod.indices) == [0, 1, 2, 3, 4, 5]
    assert lod.batches[0].material.startswith("content/tracks/t/materials/")
    assert mesh.boundsMin.x == pytest.approx(10.0)
    assert mesh.boundsMax.x == pytest.approx(13.0)


def test_build_mesh_data_empty_group() -> None:
    empty = GroupedMesh(name="empty", material="M", parts=())
    data = build_mesh_data(empty, "path").SerializeToString()
    mesh = _decode(".mesh", data)
    assert len(mesh.lods[0].positions) == 0


def test_guid_for_deterministic_nonzero() -> None:
    assert guid_for("ROAD") == guid_for("ROAD")
    assert guid_for("ROAD") != guid_for("GRASS")
    assert guid_for("ROAD") != 0


def test_build_static_mesh_scene() -> None:
    entry = StaticMeshEntry(
        name="ROAD_0",
        mesh_path="content/tracks/t/static_mesh/ROAD_0.mesh",
        surface_id="ROAD",
        bounds_min=(1.0, 2.0),
        bounds_max=(3.0, 4.0),
    )
    scene = _decode(".scene", build_static_mesh_scene([entry]).SerializeToString())
    assert len(scene.scene_actors) == 1
    actor = scene.scene_actors[0]
    assert actor.name == "ROAD_0"
    assert actor.actor.WhichOneof("actorType") == "static_mesh"
    assert actor.actor.static_mesh.mesh_path.endswith("ROAD_0.mesh")
    assert actor.actor.static_mesh.surface_id == "ROAD"
    assert actor.transform.scale.x == pytest.approx(1.0)
    assert actor.bounds.active_distance == pytest.approx(254.0)


@pytest.mark.skipif(not (_REAL / "ek_tsubaki_line.kn5").is_file(), reason="sample track missing")
def test_real_kn5_geometry_counts() -> None:
    model = Kn5Model.read(_REAL / "ek_tsubaki_line.kn5", load_textures=False)
    meshes = iter_world_meshes(model)
    assert len(meshes) > 1000
    assert len(model.materials) > 10
    total_vertices = sum(len(mesh.vertices) for mesh in meshes)
    assert total_vertices > 100000
    groups = group_by_material(meshes)
    assert len(groups) == len({mesh.material for mesh in meshes})
