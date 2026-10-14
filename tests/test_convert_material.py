"""Tests for the ACE MaterialData converter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from src.ace_formats import new_message
from src.convert.material import (
    build_material_data,
    dds_texture_name,
    material_texture_resolver,
)
from src.parser.kn5_model import Kn5Material, Kn5ShaderProperty, Kn5TextureMapping

_REAL = Path("test_data/ac1/ek_tsubaki_line")


def _material(*mappings: tuple[str, int, str], alpha_tested: bool = False) -> Kn5Material:
    return Kn5Material(
        name="TEST",
        shader="ksPerPixelMultiMap",
        blend_mode=0,
        alpha_tested=alpha_tested,
        depth_mode=0,
        properties=(
            Kn5ShaderProperty("ksSpecularEXP", 50.0, (0.0, 0.0), (0.0, 0.0, 0.0), (0.0,) * 4),
        ),
        mappings=tuple(
            Kn5TextureMapping(name=name, slot=slot, texture=texture)
            for name, slot, texture in mappings
        ),
    )


def _decode(message: Any) -> Any:
    msg = new_message(".material")
    msg.ParseFromString(message.SerializeToString())
    return msg


def _resources(msg: Any) -> dict[str, str]:
    return {entry.key: entry.value.texture for entry in msg.resources}


def test_dds_texture_name() -> None:
    assert dds_texture_name("tsubaki_day_dry.dds") == "tsubaki_day_dry"
    assert dds_texture_name("a\\b\\p_forest_a.dds") == "p_forest_a"
    assert dds_texture_name("NULL.dds") == "NULL"


def test_build_material_opaque_uses_uberbiplanar() -> None:
    resolver = material_texture_resolver("mytrack")
    material = _material(("txDiffuse", 0, "road.dds"), ("txNormal", 1, "road_NORM.dds"))
    msg = _decode(build_material_data(material, resolver))

    assert msg.materialType == "UberBiplanarMaterial"
    resources = _resources(msg)
    expected = r"content\tracks\mytrack\textures\road.texture"
    assert resources["txDiffuse"] == expected
    assert resources["Base_BaseColorMap"] == expected
    assert resources["NormalMap"] == r"content\tracks\mytrack\textures\road_NORM.texture"


def test_build_material_alpha_uses_fence() -> None:
    material = _material(("txDiffuse", 0, "bush.dds"), alpha_tested=True)
    msg = _decode(build_material_data(material, material_texture_resolver("t")))
    assert msg.materialType == "Fence"


def test_build_material_without_maps_clears_textures() -> None:
    msg = _decode(build_material_data(_material(), material_texture_resolver("t")))
    assert _resources(msg)["txDiffuse"] == ""
    assert _resources(msg)["Base_BaseColorMap"] == ""


@pytest.mark.skipif(not (_REAL / "ek_tsubaki_line.kn5").is_file(), reason="sample track missing")
def test_real_material_texture_mapping() -> None:
    from src.parser.kn5_model import Kn5Model

    model = Kn5Model.read(_REAL / "ek_tsubaki_line.kn5", load_textures=False)
    road = next(mat for mat in model.materials if mat.name == "ROAD")
    msg = _decode(build_material_data(road, material_texture_resolver("ek_tsubaki_line")))
    assert msg.materialType == "UberBiplanarMaterial"
    assert _resources(msg)["txDiffuse"] == (
        r"content\tracks\ek_tsubaki_line\textures\tsubaki_day_dry_crs_a_OPAQUE_C_DXT5_color.texture"
    )
