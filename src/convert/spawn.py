"""Convert AC1 marker nodes (from a KN5) into ACE spawn-point scenes.

AC1 encodes start/pit/hotlap positions as empty base nodes named ``AC_START_n``,
``AC_PIT_n`` and ``AC_HOTLAP_START_n``. Their node transform gives the world
position and yaw. ACE wants one ``SceneData`` per group whose actors carry a
``StartingPositionData``.

Marker rotations use the row-vector convention; yaw in degrees is
``atan2(-m02, m00)``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, cast

from google.protobuf.message import Message

from src.ace_formats import new_message
from src.convert.scene import guid_for
from src.parser.kn5_model import Kn5Model, Kn5NodeClass

# StartingPositionType enum values (Scene.proto).
TYPE_GRID = 3
TYPE_PIT = 4
TYPE_HOTLAP = 1

START_TAGS = ("Start Pos",)


@dataclass(frozen=True, slots=True)
class Marker:
    name: str
    position: tuple[float, float, float]
    yaw_degrees: float


def collect_markers(model: Kn5Model) -> list[Marker]:
    """Extract position+yaw for every base node directly under the root."""
    markers: list[Marker] = []
    for node in model.root.children:
        if node.node_class != Kn5NodeClass.BASE or len(node.transform) < 16:
            continue
        matrix = node.transform
        markers.append(
            Marker(
                name=node.name,
                position=(matrix[12], matrix[13], matrix[14]),
                yaw_degrees=math.degrees(math.atan2(-matrix[2], matrix[0])),
            )
        )
    return markers


def _sort_key(marker: Marker) -> tuple[int, str]:
    match = re.search(r"(\d+)$", marker.name)
    return (int(match.group(1)), marker.name) if match else (1 << 30, marker.name)


def select_markers(markers: list[Marker], prefix: str) -> list[Marker]:
    """Markers whose name starts with ``prefix``, ordered by numeric suffix."""
    return sorted((m for m in markers if m.name.startswith(prefix)), key=_sort_key)


def build_spawn_scene(
    markers: list[Marker],
    *,
    type_code: int,
    name_prefix: str,
    slot_start: int = 0,
) -> Message:
    """Build a ``SceneData`` of starting positions from markers."""
    scene = cast(Any, new_message(".scene"))
    for index, marker in enumerate(markers):
        slot = slot_start + index
        actor = scene.scene_actors.add()
        actor.name = f"{name_prefix}_{slot}"
        actor.enabled = True
        actor.tags.extend(START_TAGS)
        actor.transform.position.x = marker.position[0]
        actor.transform.position.y = marker.position[1]
        actor.transform.position.z = marker.position[2]
        actor.transform.rotation.y = marker.yaw_degrees
        actor.transform.scale.x = 1.0
        actor.transform.scale.y = 1.0
        actor.transform.scale.z = 1.0
        actor.guid = guid_for(actor.name)
        actor.bounds.active_distance = 250.0
        position = actor.actor.starting_position
        position.tags.SetInParent()
        position.types.append(type_code)
        position.slot_number = slot
        position.group_number = 0
    return cast(Message, scene)


def spawn_scenes(model: Kn5Model) -> dict[str, Message]:
    """Build the grid/pitlane/hotlap scenes for one KN5 model."""
    markers = collect_markers(model)
    return {
        "grid": build_spawn_scene(
            select_markers(markers, "AC_START_"), type_code=TYPE_GRID, name_prefix="starting_grid"
        ),
        "pitlane": build_spawn_scene(
            select_markers(markers, "AC_PIT_"), type_code=TYPE_PIT, name_prefix="pit_spawn"
        ),
        "hotlap": build_spawn_scene(
            select_markers(markers, "AC_HOTLAP_START_"),
            type_code=TYPE_HOTLAP,
            name_prefix="starting_position_hotlap",
        ),
    }
