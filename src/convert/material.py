"""Convert AC1 KN5 materials into ACE ``MaterialData`` assets.

AC1 KN5 materials are a fixed-function shader model; ACE uses PBR material
templates (``UberBiplanarMaterial`` for opaque surfaces, ``Fence`` for
alpha-tested foliage). Each template carries ~290 variables; we reuse a
captured template and substitute the texture resources.
"""

from __future__ import annotations

import base64
import re
from collections.abc import Callable
from typing import Any, cast

from google.protobuf.message import Message

from src import mapping
from src.ace_formats import new_message
from src.convert._material_templates import TEMPLATES
from src.parser.kn5_model import Kn5Material

TextureResolver = Callable[[str], str]

TEXTURE_KEYS = ("txDiffuse", "Base_BaseColorMap", "NormalMap", "Base_NormalMap")

# EvoForge's ``wVV``: simple opaque surfaces (dynamic road, distant scenery,
# physics-only collision meshes, mirrors, emissive signals) use the plain PBR
# ``DefaultMaterial``; alpha-tested foliage uses ``Fence``; everything else
# (ground/splat surfaces) uses ``UberBiplanarMaterial``.
_DEFAULT_MATERIAL = re.compile(r"MESH$|MOUNTAIN|COLLISION|MIRROR|SIGNAL", re.IGNORECASE)


def dds_texture_name(filename: str) -> str:
    """ACE texture asset name for an AC1 DDS file (basename without extension)."""
    return mapping.sanitize_name(filename.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0])


def _template_name(material: Kn5Material) -> str:
    if material.alpha_tested:
        return "Fence"
    if _DEFAULT_MATERIAL.search(material.name):
        return "DefaultMaterial"
    return "UberBiplanarMaterial"


def build_material_data(
    material: Kn5Material,
    texture_resolver: TextureResolver,
    *,
    render_order: int | None = None,
) -> Message:
    """Serialize one AC1 KN5 material as an ACE ``MaterialData`` message."""
    template_name = _template_name(material)
    msg = new_message(".material")
    msg.ParseFromString(base64.b64decode(TEMPLATES[template_name]))
    if render_order is not None:
        cast(Any, msg).renderOrder = render_order

    if template_name == "UberBiplanarMaterial":
        # EvoForge derives the albedo/roughness from the AC1 fixed-function
        # material instead of the template defaults.
        ks_diffuse = _prop(material, "ksDiffuse", 0.2)
        ks_ambient = _prop(material, "ksAmbient", 0.2)
        basecolor = min(1.0, (ks_diffuse + ks_ambient) * 2.0)
        for entry in cast(Any, msg).variables:
            if entry.key in ("Basecolor", "Base_Basecolor"):
                entry.value.asVector4.x = basecolor
                entry.value.asVector4.y = basecolor
                entry.value.asVector4.z = basecolor
                entry.value.asVector4.w = 1.0
            elif entry.key in (
                "Roughness",
                "Base_Roughness",
                "Red_Roughness",
                "Green_Roughness",
                "Blue_Roughness",
            ):
                entry.value.asFloat = 0.9

    diffuse = material.mapping("txDiffuse")
    normal = material.mapping("txNormal")
    diffuse_path = texture_resolver(diffuse.texture) if diffuse else ""
    normal_path = texture_resolver(normal.texture) if normal else ""

    values = {
        "txDiffuse": diffuse_path,
        "Base_BaseColorMap": diffuse_path,
        "NormalMap": normal_path,
        "Base_NormalMap": normal_path,
    }
    resources = cast(Any, msg).resources
    for entry in resources:
        entry.value.texture = values.get(entry.key, "")
    return msg


def _prop(material: Kn5Material, name: str, default: float) -> float:
    prop = material.property(name)
    return float(prop.value_a) if prop is not None else default


def material_texture_resolver(track_id: str) -> TextureResolver:
    """Return a resolver mapping AC1 DDS names to ACE texture asset paths."""
    return lambda filename: mapping.as_message_path(
        mapping.texture_path(track_id, dds_texture_name(filename))
    )
