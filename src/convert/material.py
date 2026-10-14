"""Convert AC1 KN5 materials into ACE ``MaterialData`` assets.

AC1 KN5 materials are a fixed-function shader model; ACE uses PBR material
templates (``UberBiplanarMaterial`` for opaque surfaces, ``Fence`` for
alpha-tested foliage). Each template carries ~290 variables; we reuse a
captured template and substitute the texture resources.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any, cast

from google.protobuf.message import Message

from src import mapping
from src.ace_formats import new_message
from src.convert._material_templates import TEMPLATES
from src.parser.kn5_model import Kn5Material

TextureResolver = Callable[[str], str]

TEXTURE_KEYS = ("txDiffuse", "Base_BaseColorMap", "NormalMap", "Base_NormalMap")


def dds_texture_name(filename: str) -> str:
    """ACE texture asset name for an AC1 DDS file (basename without extension)."""
    return mapping.sanitize_name(filename.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0])


def _template_name(material: Kn5Material) -> str:
    return "Fence" if material.alpha_tested else "UberBiplanarMaterial"


def build_material_data(
    material: Kn5Material,
    texture_resolver: TextureResolver,
    *,
    render_order: int | None = None,
) -> Message:
    """Serialize one AC1 KN5 material as an ACE ``MaterialData`` message."""
    msg = new_message(".material")
    msg.ParseFromString(base64.b64decode(TEMPLATES[_template_name(material)]))
    if render_order is not None:
        cast(Any, msg).renderOrder = render_order

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


def material_texture_resolver(track_id: str) -> TextureResolver:
    """Return a resolver mapping AC1 DDS names to ACE texture asset paths."""
    return lambda filename: mapping.as_message_path(
        mapping.texture_path(track_id, dds_texture_name(filename))
    )
