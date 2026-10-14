"""Assetto Corsa 1 ``fast_lane.ai`` AI spline parser (format version 7).

Layout (little-endian, recovered from gro-ove/actools ``AcTools/AiFile``)::

    int32 version        # 7
    int32 numPoints
    int32 lapTime
    int32 sampleCount
    AiPoint[numPoints]       # vec3 position, float length, int32 id (20 bytes)
    int32 numExtra
    AiPointExtra[numExtra]   # 18 floats: speed..grade
    int32 hasGrid
    AiSplineGrid grid        # only when hasGrid != 0
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from src.parser.errors import TrackParseError

_VEC3 = struct.Struct("<3f")
_POINT = struct.Struct("<3ffi")  # position, length, id
_EXTRA = struct.Struct("<18f")
_HEADER = struct.Struct("<4i")


@dataclass(frozen=True, slots=True)
class AiPoint:
    position: tuple[float, float, float]
    length: float
    id: int


@dataclass(frozen=True, slots=True)
class AiPointExtra:
    speed: float
    gas: float
    brake: float
    obsolete_lat_g: float
    radius: float
    side_left: float
    side_right: float
    camber: float
    direction: float
    normal: tuple[float, float, float]
    length: float
    forward_vector: tuple[float, float, float]
    tag: float
    grade: float

    @property
    def width(self) -> float:
        return self.side_left + self.side_right


@dataclass(frozen=True, slots=True)
class AiSplineGrid:
    max_extreme: tuple[float, float, float]
    min_extreme: tuple[float, float, float]
    neighbors_considered: int
    sampling_density: float
    items: tuple[tuple[tuple[int, ...], ...], ...]


@dataclass(frozen=True, slots=True)
class AiSpline:
    version: int
    lap_time: int
    sample_count: int
    points: tuple[AiPoint, ...]
    extra: tuple[AiPointExtra, ...]
    grid: AiSplineGrid | None

    @property
    def positions(self) -> list[tuple[float, float, float]]:
        return [point.position for point in self.points]

    @classmethod
    def read(cls, path: str | Path) -> AiSpline:
        cursor = _Cursor(Path(path).read_bytes(), path)
        version, num_points, lap_time, sample_count = cursor.header()
        if version != 7:
            raise TrackParseError(path, f"unsupported AI spline version {version}")
        points = tuple(cursor.point() for _ in range(num_points))
        num_extra = cursor.i32()
        extra = tuple(cursor.extra() for _ in range(num_extra))
        grid = cursor.grid() if cursor.i32() != 0 else None
        if cursor.remaining:
            raise TrackParseError(path, f"{cursor.remaining} trailing bytes after AI spline")
        return cls(
            version=version,
            lap_time=lap_time,
            sample_count=sample_count,
            points=points,
            extra=extra,
            grid=grid,
        )


class _Cursor:
    def __init__(self, data: bytes, path: str | Path) -> None:
        self.data = data
        self.path = path
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    def _read(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise TrackParseError(self.path, f"unexpected end of file at offset {self.offset}")
        chunk = self.data[self.offset : end]
        self.offset = end
        return chunk

    def header(self) -> tuple[int, int, int, int]:
        return _HEADER.unpack(self._read(_HEADER.size))

    def point(self) -> AiPoint:
        x, y, z, length, point_id = _POINT.unpack(self._read(_POINT.size))
        return AiPoint(position=(x, y, z), length=length, id=point_id)

    def extra(self) -> AiPointExtra:
        values = _EXTRA.unpack(self._read(_EXTRA.size))
        return AiPointExtra(
            speed=values[0],
            gas=values[1],
            brake=values[2],
            obsolete_lat_g=values[3],
            radius=values[4],
            side_left=values[5],
            side_right=values[6],
            camber=values[7],
            direction=values[8],
            normal=(values[9], values[10], values[11]),
            length=values[12],
            forward_vector=(values[13], values[14], values[15]),
            tag=values[16],
            grade=values[17],
        )

    def i32(self) -> int:
        return int(struct.unpack("<i", self._read(4))[0])

    def f32(self) -> float:
        return float(struct.unpack("<f", self._read(4))[0])

    def vec3(self) -> tuple[float, float, float]:
        return _VEC3.unpack(self._read(_VEC3.size))

    def grid(self) -> AiSplineGrid:
        max_extreme = self.vec3()
        min_extreme = self.vec3()
        neighbors = self.i32()
        density = self.f32()
        items = tuple(self._grid_item() for _ in range(self.i32()))
        return AiSplineGrid(
            max_extreme=max_extreme,
            min_extreme=min_extreme,
            neighbors_considered=neighbors,
            sampling_density=density,
            items=items,
        )

    def _grid_item(self) -> tuple[tuple[int, ...], ...]:
        return tuple(self._grid_sub() for _ in range(self.i32()))

    def _grid_sub(self) -> tuple[int, ...]:
        return tuple(self.i32() for _ in range(self.i32()))
