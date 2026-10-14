"""Build the ACE ``timelines`` zone scene from AC1 start/finish markers.

AC1 stores start/finish helper markers as ``AC_AB_START_L/_R`` and
``AC_AB_FINISH_L/_R`` base nodes. ACE wants a ``ZoneActorData`` whose
geometrics are the timing lines and whose instance is a ``Timelines`` marker.
"""

from __future__ import annotations

import math
from typing import Any, cast

from google.protobuf.message import Message

from src.ace_formats import new_message
from src.convert.scene import guid_for
from src.convert.spawn import Marker, collect_markers, select_markers
from src.parser.kn5_model import Kn5Model

_LINE_WIDTH = 0.5


def _perpendicular(
    start: tuple[float, float, float], end: tuple[float, float, float]
) -> tuple[float, float, float]:
    dx, dz = end[0] - start[0], end[2] - start[2]
    length = math.hypot(dx, dz) or 1.0
    return (-dz / length, 0.0, dx / length)


def _geometric(zone_data: Any, left: Marker, right: Marker, color_index: int) -> None:
    geometry = zone_data.geometrics.add()
    normal = _perpendicular(left.position, right.position)
    for point in (left, right):
        node = geometry.points.add()
        node.position.x, node.position.y, node.position.z = point.position
        node.scale.x = 1.0
        node.scale.y = 1.0
        node.scale.z = 1.0
    for point in (right, left):
        node = geometry.points.add()
        node.position.x = point.position[0] + normal[0] * _LINE_WIDTH
        node.position.y = point.position[1]
        node.position.z = point.position[2] + normal[2] * _LINE_WIDTH
        node.scale.x = 1.0
        node.scale.y = 1.0
        node.scale.z = 1.0
    if color_index == 0:
        geometry.color.x = 1.0
    elif color_index == 1:
        geometry.color.y = 1.0
    else:
        geometry.color.z = 1.0
    geometry.renderHeight = 0.8


def build_timelines_scene(markers: list[Marker], name: str) -> Message | None:
    """Build a ``timelines`` zone scene, or ``None`` if no markers are present."""
    lines: list[tuple[Marker, Marker]] = []
    for prefix in ("AC_AB_START_", "AC_AB_FINISH_"):
        left = select_markers(markers, prefix + "L")
        right = select_markers(markers, prefix + "R")
        if left and right:
            lines.append((left[0], right[0]))
    if not lines:
        return None

    scene = cast(Any, new_message(".scene"))
    actor = scene.scene_actors.add()
    actor.name = name
    actor.enabled = True
    actor.tags.append("Zone")
    actor.transform.scale.x = 1.0
    actor.transform.scale.y = 1.0
    actor.transform.scale.z = 1.0
    actor.guid = guid_for(name)

    zone = actor.actor.zones
    zone.debug_options.node_drawing_size = 0.22
    zone.debug_options.segment_thickness = 2
    for index, (line_left, line_right) in enumerate(lines):
        _geometric(zone.data, line_left, line_right, index)
    zone.data.instances.add().timelines.SetInParent()
    return cast(Message, scene)


def timelines_scene_for(model: Kn5Model, layout: str) -> Message | None:
    """Convenience wrapper: build the timelines scene for one KN5 model."""
    return build_timelines_scene(collect_markers(model), f"timelines_{layout}")
