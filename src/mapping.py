"""AC1 -> ACE internal path and constant mappings.

Every internal ACE path and tunable default lives here so the converters stay
free of ad-hoc string logic (track-converter skill rule 1). Internal paths use
forward slashes; the packer converts them to backslashes for the file table.
"""

from __future__ import annotations

import re

TRACKS_PREFIX = "content/tracks"
UIRESOURCES_PREFIX = "uiresources"
TRACKMAP_IMAGES = f"{UIRESOURCES_PREFIX}/images/trackmaps"
TRACK_IMAGES = f"{UIRESOURCES_PREFIX}/images/tracks"

# Heuristic defaults matching the EvoForge reference output for ek_tsubak_hv.
DYNAMIC_TRACK_ENABLED = True
BASE_GRIP = 0.98
TRACK_EXTENSION_M = 6000.0
TRACK_ALTITUDE_M = 55.0


def sanitize_name(name: str) -> str:
    """Make an AC1 asset name safe for ACE internal paths.

    Spaces and unsupported characters become underscores (``BUSHES A`` ->
    ``BUSHES_A``), matching the reference package naming.
    """
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", name).strip("_")
    return cleaned or "unnamed"


def as_message_path(path: str) -> str:
    """ACE message fields embed Windows-style backslash paths, not POSIX ones."""
    return path.replace("/", "\\")


def track_dir(track_id: str) -> str:
    return f"{TRACKS_PREFIX}/{track_id}"


def track_file(track_id: str) -> str:
    return f"{track_dir(track_id)}/{track_id}.track"


def track_scene(track_id: str) -> str:
    return f"{track_dir(track_id)}/{track_id}.scene"


def mesh_path(track_id: str, name: str) -> str:
    return f"{track_dir(track_id)}/static_mesh/{name}.mesh"


def material_path(track_id: str, name: str) -> str:
    return f"{track_dir(track_id)}/materials/{name}.material"


def texture_path(track_id: str, name: str) -> str:
    return f"{track_dir(track_id)}/textures/{name}.texture"


def texture_mips_path(track_id: str, name: str) -> str:
    return f"{track_dir(track_id)}/textures/{name}.texturemips"


def container_scene(track_id: str, name: str) -> str:
    return f"{track_dir(track_id)}/containers/{name}.scene"


def ui_map_svg(track_id: str) -> str:
    return f"{track_dir(track_id)}/ui/map.svg"


def ui_previews_json(track_id: str) -> str:
    return f"{track_dir(track_id)}/ui/previews.json"


def ui_photo_texture(track_id: str) -> str:
    return f"{track_dir(track_id)}/ui/photo.texture"


def ui_map_texture(track_id: str) -> str:
    return f"{track_dir(track_id)}/ui/map.texture"


def texture_mips_for(texture_arc: str) -> str:
    return texture_arc + "mips"


def layout_container_scene(track_id: str, layout: str) -> str:
    return container_scene(track_id, f"layout_{layout}")


def layout_aispline(track_id: str, layout: str) -> str:
    return f"{track_dir(track_id)}/layouts/layout_{layout}.ideal_line.aisplinedata"


def layout_control_points(track_id: str, layout: str) -> str:
    return f"{track_dir(track_id)}/layouts/layout_{layout}.trackcontrolpoints"


def layout_reference(track_id: str, layout: str) -> str:
    return f"{track_dir(track_id)}/layouts/layout_{layout}.carcategory_none.reference"


def layout_track_layout(track_id: str, layout: str) -> str:
    return f"{track_dir(track_id)}/layouts/{layout}.track_layout"


def trackmap_svg(track_id: str, layout: str) -> str:
    return f"{TRACKMAP_IMAGES}/{track_id}-{layout}.svg"


def trackmap_texture(track_id: str, layout: str) -> str:
    return f"{TRACKMAP_IMAGES}/{track_id}-{layout}.texture"


def track_preview_texture(track_id: str, layout: str) -> str:
    return f"{TRACK_IMAGES}/{track_id}-{layout}.texture"
