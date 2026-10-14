"""Assetto Corsa 1 track folder parser.

An AC1 track is a folder whose name is the track id, containing one or more
layout subfolders. Each layout holds ``data/*.ini`` and ``ai/fast_lane.ai``;
shared models are listed in ``models_<layout>.ini`` at the root, and per-layout
UI metadata lives in ``ui/<layout>/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.parser.ai_spline import AiSpline
from src.parser.errors import TrackParseError
from src.parser.ini import IniFile


def _vec3(value: str | None) -> tuple[float, float, float]:
    if not value:
        return (0.0, 0.0, 0.0)
    try:
        x, y, z = (float(part) for part in value.split(","))
    except ValueError as exc:
        raise TrackParseError("<ini>", f"invalid vector {value!r}") from exc
    return (x, y, z)


@dataclass(frozen=True, slots=True)
class Surface:
    """One ``[SURFACE_n]`` block from ``data/surfaces.ini``."""

    key: str
    friction: float
    damping: float
    is_valid_track: bool
    is_pitlane: bool
    black_flag_time: float
    sin_height: float
    sin_length: float
    vibration_gain: float
    vibration_length: float
    wav: str
    ff_effect: str
    dirt_additive: bool


@dataclass(frozen=True, slots=True)
class MapParameters:
    """``data/map.ini`` ``[PARAMETERS]``."""

    width: int
    height: int
    x_offset: float
    z_offset: float
    margin: float
    scale_factor: float
    drawing_size: float


@dataclass(frozen=True, slots=True)
class AiHint:
    start: float
    end: float
    value: float


@dataclass(frozen=True, slots=True)
class Model:
    file: str
    position: tuple[float, float, float]
    rotation: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class TrackUi:
    name: str
    description: str
    tags: tuple[str, ...]
    country: str
    city: str
    length: str
    width: str
    pitboxes: str
    author: str
    version: str
    year: int | None
    preview: Path | None
    outline: Path | None


@dataclass(frozen=True, slots=True)
class AcLayout:
    name: str
    directory: Path
    surfaces: tuple[Surface, ...]
    map_parameters: MapParameters | None
    ai_hints: tuple[AiHint, ...]
    models: tuple[Model, ...]
    fast_lane: AiSpline | None
    ui: TrackUi | None


@dataclass(frozen=True, slots=True)
class AcTrack:
    id: str
    root: Path
    layouts: dict[str, AcLayout] = field(default_factory=dict)
    collision: Path | None = None


def _parse_surfaces(path: Path) -> tuple[Surface, ...]:
    ini = IniFile.read(path)
    surfaces: list[Surface] = []
    for section in sorted(ini.sections, key=_surface_sort_key):
        keys = ini.sections[section]
        if "KEY" not in keys:
            continue
        surfaces.append(
            Surface(
                key=keys["KEY"],
                friction=ini.get_float(section, "FRICTION"),
                damping=ini.get_float(section, "DAMPING"),
                is_valid_track=ini.get_bool(section, "IS_VALID_TRACK"),
                is_pitlane=ini.get_bool(section, "IS_PITLANE"),
                black_flag_time=ini.get_float(section, "BLACK_FLAG_TIME"),
                sin_height=ini.get_float(section, "SIN_HEIGHT"),
                sin_length=ini.get_float(section, "SIN_LENGTH"),
                vibration_gain=ini.get_float(section, "VIBRATION_GAIN"),
                vibration_length=ini.get_float(section, "VIBRATION_LENGTH"),
                wav=ini.get(section, "WAV", "") or "",
                ff_effect=ini.get(section, "FF_EFFECT", "") or "",
                dirt_additive=ini.get_bool(section, "DIRT_ADDITIVE"),
            )
        )
    if not surfaces:
        raise TrackParseError(path, "no surfaces defined")
    return tuple(surfaces)


def _surface_sort_key(section: str) -> tuple[int, str]:
    suffix = section.replace("SURFACE_", "")
    return (int(suffix), "") if suffix.isdigit() else (1 << 30, section)


def _parse_map(path: Path) -> MapParameters | None:
    if not path.is_file():
        return None
    ini = IniFile.read(path)
    if "PARAMETERS" not in ini.sections:
        return None
    return MapParameters(
        width=ini.get_int("PARAMETERS", "WIDTH"),
        height=ini.get_int("PARAMETERS", "HEIGHT"),
        x_offset=ini.get_float("PARAMETERS", "X_OFFSET"),
        z_offset=ini.get_float("PARAMETERS", "Z_OFFSET"),
        margin=ini.get_float("PARAMETERS", "MARGIN"),
        scale_factor=ini.get_float("PARAMETERS", "SCALE_FACTOR", 1.0),
        drawing_size=ini.get_float("PARAMETERS", "DRAWING_SIZE", 10.0),
    )


def _parse_ai_hints(path: Path) -> tuple[AiHint, ...]:
    if not path.is_file():
        return ()
    ini = IniFile.read(path)
    hints = [
        AiHint(
            start=ini.get_float(section, "START"),
            end=ini.get_float(section, "END"),
            value=ini.get_float(section, "VALUE"),
        )
        for section in sorted(ini.sections)
        if section.startswith("HINT_")
    ]
    return tuple(hints)


def _parse_models(path: Path) -> tuple[Model, ...]:
    if not path.is_file():
        return ()
    ini = IniFile.read(path)
    models = [
        Model(
            file=ini.get(section, "FILE", "") or "",
            position=_vec3(ini.get(section, "POSITION")),
            rotation=_vec3(ini.get(section, "ROTATION")),
        )
        for section in sorted(ini.sections)
        if section.startswith("MODEL_")
    ]
    return tuple(models)


def _parse_ui(directory: Path) -> TrackUi | None:
    json_path = directory / "ui_track.json"
    if not json_path.is_file():
        return None
    try:
        data: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8-sig"))
    except (ValueError, OSError) as exc:
        raise TrackParseError(json_path, "invalid ui_track.json") from exc

    def text(key: str) -> str:
        return str(data.get(key, ""))

    preview = directory / "preview.png"
    outline = directory / "outline.png"
    year = data.get("year")
    return TrackUi(
        name=text("name"),
        description=text("description"),
        tags=tuple(str(tag) for tag in data.get("tags", [])),
        country=text("country"),
        city=text("city"),
        length=text("length"),
        width=text("width"),
        pitboxes=text("pitboxes"),
        author=text("author"),
        version=text("version"),
        year=int(year) if isinstance(year, int | float) else None,
        preview=preview if preview.is_file() else None,
        outline=outline if outline.is_file() else None,
    )


def _parse_layout(root: Path, name: str) -> AcLayout:
    directory = root / name
    data_dir = directory / "data"
    fast_lane_path = directory / "ai" / "fast_lane.ai"
    return AcLayout(
        name=name,
        directory=directory,
        surfaces=_parse_surfaces(data_dir / "surfaces.ini"),
        map_parameters=_parse_map(data_dir / "map.ini"),
        ai_hints=_parse_ai_hints(data_dir / "ai_hints.ini"),
        models=_parse_models(root / f"models_{name}.ini"),
        fast_lane=AiSpline.read(fast_lane_path) if fast_lane_path.is_file() else None,
        ui=_parse_ui(root / "ui" / name),
    )


def parse_track(root: str | Path) -> AcTrack:
    """Parse an AC1 track folder into a structured model."""
    path = Path(root)
    if not path.is_dir():
        raise TrackParseError(path, "track folder not found")

    layout_names = sorted(
        child.name for child in path.iterdir() if (child / "data" / "surfaces.ini").is_file()
    )
    if not layout_names:
        raise TrackParseError(path, "no layouts found (no */data/surfaces.ini)")

    layouts = {name: _parse_layout(path, name) for name in layout_names}
    collision = path / "collision.kn5"
    return AcTrack(
        id=path.name,
        root=path,
        layouts=layouts,
        collision=collision if collision.is_file() else None,
    )
