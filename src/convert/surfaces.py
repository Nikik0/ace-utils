"""Build the ACE root-scene surface-definition actor from AC1 surfaces."""

from __future__ import annotations

from typing import Any, cast

from google.protobuf.message import Message

from src.ace_formats import new_message
from src.convert.scene import guid_for
from src.parser.ac1 import Surface

COLLIDER_CATEGORY_SURFACE = 1
_SURFACE_TYPE_ASPHALT = 0
_SURFACE_TYPE_GRASS = 3
_SURFACE_TYPE_SAND = 4
_SURFACE_TYPE_FLAT_KERB = 5


def surface_type(key: str) -> int:
    upper = key.upper()
    if "GRASS" in upper:
        return _SURFACE_TYPE_GRASS
    if "GRAVEL" in upper or "SAND" in upper:
        return _SURFACE_TYPE_SAND
    if "RUMBLE" in upper or "KERB" in upper:
        return _SURFACE_TYPE_FLAT_KERB
    return _SURFACE_TYPE_ASPHALT


def build_surfaces_scene(surfaces: tuple[Surface, ...]) -> Message:
    """Create a ``SceneData`` holding one surface-definition actor."""
    scene = cast(Any, new_message(".scene"))
    actor = scene.scene_actors.add()
    actor.name = "AC1 surfaces"
    actor.enabled = True
    actor.tags.append("Surface Def")
    actor.transform.scale.x = 1.0
    actor.transform.scale.y = 1.0
    actor.transform.scale.z = 1.0
    actor.guid = guid_for("AC1 surfaces")
    actor.active_distance_override = 100000.0

    definitions = actor.actor.surfaces.SurfaceDefinitionData
    for surface in surfaces:
        entry = definitions.add()
        entry.surfaceId = surface.key
        entry.collisionCategory = COLLIDER_CATEGORY_SURFACE
        entry.gripMod = surface.friction
        entry.isValidTrack = surface.is_valid_track
        entry.dirtAdditiveK = 1.0 if surface.dirt_additive else 0.0
        entry.sinHeight = surface.sin_height
        entry.sinLength = surface.sin_length
        entry.vibrationGain = surface.vibration_gain
        entry.vibrationLength = surface.vibration_length
        entry.wearMult = 1.0
        entry.frictionRain = -1.0
        entry.frictionRainGamma = 1.0
        entry.surface_type = surface_type(surface.key)
    return cast(Message, scene)
