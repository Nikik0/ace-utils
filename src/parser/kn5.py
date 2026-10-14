"""Assetto Corsa 1 ``.kn5`` header/metadata reader.

Only the header, texture table and material table are parsed; node geometry is
out of scope for now (geometry conversion is a later milestone). Layout follows
gro-ove/actools ``AcTools/Kn5File``::

    char[6] magic "sc6969"
    int32 version
    int32 extra            # present only when version > 5
    int32 textureCount
    texture[textureCount]: int32 active, string name, uint32 length, byte[length] data
    int32 materialCount
    material[materialCount]: ...
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from src.parser.errors import TrackParseError

KN5_MAGIC = b"sc6969"
_PROPERTY_BYTES = 40  # float + vec2 + vec3 + vec4


@dataclass(frozen=True, slots=True)
class Kn5Texture:
    name: str
    length: int
    active: bool


@dataclass(frozen=True, slots=True)
class Kn5Material:
    name: str
    shader: str


@dataclass(frozen=True, slots=True)
class Kn5Metadata:
    version: int
    extra: int
    textures: tuple[Kn5Texture, ...]
    materials: tuple[Kn5Material, ...]
    geometry_bytes: int

    @property
    def has_geometry(self) -> bool:
        return self.geometry_bytes > 0

    @classmethod
    def read(cls, path: str | Path) -> Kn5Metadata:
        cursor = _Cursor(Path(path).read_bytes(), path)
        if cursor.take(6) != KN5_MAGIC:
            raise TrackParseError(path, "not a KN5 file (bad magic)")
        version = cursor.i32()
        extra = cursor.i32() if version > 5 else 0
        textures = tuple(cursor.texture() for _ in range(cursor.i32()))
        materials = tuple(cursor.material() for _ in range(cursor.i32()))
        return cls(
            version=version,
            extra=extra,
            textures=textures,
            materials=materials,
            geometry_bytes=cursor.remaining,
        )


class _Cursor:
    def __init__(self, data: bytes, path: str | Path) -> None:
        self.data = data
        self.path = path
        self.offset = 0

    @property
    def remaining(self) -> int:
        return len(self.data) - self.offset

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise TrackParseError(self.path, f"unexpected end of file at offset {self.offset}")
        chunk = self.data[self.offset : end]
        self.offset = end
        return chunk

    def skip(self, size: int) -> None:
        self.take(size)

    def i32(self) -> int:
        return int(struct.unpack("<i", self.take(4))[0])

    def u32(self) -> int:
        return int(struct.unpack("<I", self.take(4))[0])

    def string(self) -> str:
        length = self.i32()
        if length < 0:
            raise TrackParseError(self.path, f"negative string length at offset {self.offset}")
        return self.take(length).decode("utf-8", "replace")

    def texture(self) -> Kn5Texture:
        active = self.i32()
        name = self.string()
        length = self.u32()
        self.skip(length)
        return Kn5Texture(name=name, length=length, active=active == 1)

    def material(self) -> Kn5Material:
        name = self.string()
        shader = self.string()
        self.skip(6)  # blend mode (byte) + alpha tested (byte) + depth mode (int32)
        for _ in range(self.i32()):
            self.string()
            self.skip(_PROPERTY_BYTES)
        for _ in range(self.i32()):
            self.string()
            self.skip(4)
            self.string()
        return Kn5Material(name=name, shader=shader)
