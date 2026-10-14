"""Build ACE ``SceneData`` containers referencing static meshes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

from google.protobuf.message import Message

from src.ace_formats import new_message

DEFAULT_TAGS = ("SMesh", "collidable", "track_layout")


def guid_for(name: str) -> int:
    """Deterministic non-zero 64-bit guid from an actor name (FNV-1a)."""
    value = 1469598103934665603
    for byte in name.encode("utf-8"):
        value ^= byte
        value = (value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return value or 1


@dataclass(frozen=True, slots=True)
class StaticMeshEntry:
    name: str
    mesh_path: str
    surface_id: str = ""
    bounds_min: tuple[float, float] = (0.0, 0.0)
    bounds_max: tuple[float, float] = (0.0, 0.0)
    active_distance: float = 254.0
    active_distance_override: float = 100000.0
    tags: tuple[str, ...] = field(default=DEFAULT_TAGS)
    physics_only: bool = False


def build_static_mesh_scene(entries: list[StaticMeshEntry]) -> Message:
    """Create a ``SceneData`` whose actors instantiate the given meshes."""
    scene = cast(Any, new_message(".scene"))
    for entry in entries:
        actor = scene.scene_actors.add()
        actor.name = entry.name
        actor.enabled = True
        actor.tags.extend(entry.tags)
        actor.transform.scale.x = 1.0
        actor.transform.scale.y = 1.0
        actor.transform.scale.z = 1.0
        actor.guid = guid_for(entry.name)
        actor.bounds.bounds_min.x, actor.bounds.bounds_min.y = entry.bounds_min
        actor.bounds.bounds_max.x, actor.bounds.bounds_max.y = entry.bounds_max
        actor.bounds.active_distance = entry.active_distance
        actor.active_distance_override = entry.active_distance_override
        actor.actor.static_mesh.mesh_path = entry.mesh_path
        actor.actor.static_mesh.surface_id = entry.surface_id
        actor.actor.static_mesh.physics_only = entry.physics_only
    return cast(Message, scene)


def add_track_layout_actor(
    scene: Message,
    *,
    name: str,
    layout_path: str,
    start: tuple[float, float, float],
    end: tuple[float, float, float],
    is_closed: bool = True,
) -> None:
    """Append a ``TrackLayoutData`` actor to an existing scene."""
    actors = cast(Any, scene).scene_actors
    actor = actors.add()
    actor.name = name
    actor.enabled = True
    actor.tags.append("Track Layout")
    actor.transform.scale.x = 1.0
    actor.transform.scale.y = 1.0
    actor.transform.scale.z = 1.0
    actor.guid = guid_for(name)
    actor.bounds.active_distance = 250.0
    actor.active_distance_override = 100000.0
    data = actor.actor.track_layout
    data.path = layout_path
    data.start_position.x, data.start_position.y, data.start_position.z = start
    data.end_position.x, data.end_position.y, data.end_position.z = end
    data.is_closed = is_closed
    data.show_middle_segments = True
    data.use_for_tv_cameras = True
