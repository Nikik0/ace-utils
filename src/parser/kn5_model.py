"""Assetto Corsa 1 ``.kn5`` geometry reader.

Binary layout recovered from gro-ove/actools ``AcTools/Kn5File`` (MIT). Only
the reader is ported; ``.kn5`` files are never written back.

Header::

    char[6] magic "sc6969"
    int32   version
    int32   extra                # only when version > 5
    int32   textureCount
    Texture[textureCount]        # int32 active, string name, uint32 length, byte[length]
    int32   materialCount
    Material[materialCount]      # see _read_material
    Node root                    # recursive, see _read_node

Each node::

    int32  nodeClass             # 1 base, 2 mesh, 3 skinned mesh
    string name
    int32  childCount            # children follow the node, pre-order
    byte   active
    base:    float[16] transform
    mesh:    byte castShadows, byte isVisible, byte isTransparent,
             uint32 vertexCount, Vertex[vertexCount] (44 bytes each),
             uint32 indexCount, uint16[indexCount],
             uint32 materialId, uint32 layer, float lodIn, float lodOut,
             vec3 boundsCenter, float boundsRadius, byte isRenderable
    skinned: as mesh but with bones and per-vertex weights (76 bytes each)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path

from src.parser.errors import TrackParseError

KN5_MAGIC = b"sc6969"

_V2 = struct.Struct("<2f")
_V3 = struct.Struct("<3f")
_V4 = struct.Struct("<4f")
_MATRIX = struct.Struct("<16f")
_I32 = struct.Struct("<i")
_U32 = struct.Struct("<I")
_F32 = struct.Struct("<f")


class Kn5NodeClass(IntEnum):
    BASE = 1
    MESH = 2
    SKINNED_MESH = 3


@dataclass(frozen=True, slots=True)
class Kn5Vertex:
    position: tuple[float, float, float]
    normal: tuple[float, float, float]
    uv: tuple[float, float]
    tangent: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Kn5Weight:
    weights: tuple[float, float, float, float]
    indices: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Kn5Bone:
    name: str
    transform: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class Kn5ShaderProperty:
    name: str
    value_a: float
    value_b: tuple[float, float]
    value_c: tuple[float, float, float]
    value_d: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Kn5TextureMapping:
    name: str
    slot: int
    texture: str


@dataclass(frozen=True, slots=True)
class Kn5Material:
    name: str
    shader: str
    blend_mode: int
    alpha_tested: bool
    depth_mode: int
    properties: tuple[Kn5ShaderProperty, ...]
    mappings: tuple[Kn5TextureMapping, ...]

    def property(self, name: str) -> Kn5ShaderProperty | None:
        for prop in self.properties:
            if prop.name == name:
                return prop
        return None

    def mapping(self, name: str) -> Kn5TextureMapping | None:
        for mapping in self.mappings:
            if mapping.name == name:
                return mapping
        return None


@dataclass(frozen=True, slots=True)
class Kn5Texture:
    name: str
    active: bool
    length: int
    data: bytes | None = None


@dataclass(frozen=True, slots=True)
class Kn5Node:
    node_class: Kn5NodeClass
    name: str
    active: bool
    transform: tuple[float, ...] = ()
    children: tuple[Kn5Node, ...] = ()
    cast_shadows: bool = False
    is_visible: bool = False
    is_transparent: bool = False
    is_renderable: bool = False
    vertices: tuple[Kn5Vertex, ...] = ()
    weights: tuple[Kn5Weight, ...] = ()
    bones: tuple[Kn5Bone, ...] = ()
    indices: tuple[int, ...] = ()
    material_id: int = 0
    layer: int = 0
    lod_in: float = 0.0
    lod_out: float = 0.0
    bounds_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    bounds_radius: float = 0.0

    @property
    def is_mesh(self) -> bool:
        return self.node_class in (Kn5NodeClass.MESH, Kn5NodeClass.SKINNED_MESH)


@dataclass(frozen=True, slots=True)
class Kn5Model:
    version: int
    extra: int
    textures: tuple[Kn5Texture, ...]
    materials: tuple[Kn5Material, ...]
    root: Kn5Node
    path: Path = field(default=Path("."))

    def mesh_nodes(self) -> list[Kn5Node]:
        """All mesh nodes in depth-first order."""
        result: list[Kn5Node] = []

        def walk(node: Kn5Node) -> None:
            if node.is_mesh:
                result.append(node)
            for child in node.children:
                walk(child)

        walk(self.root)
        return result

    @classmethod
    def read(cls, path: str | Path, *, load_textures: bool = False) -> Kn5Model:
        source = Path(path)
        cursor = _Cursor(source.read_bytes(), source)
        if cursor.take(6) != KN5_MAGIC:
            raise TrackParseError(source, "not a KN5 file (bad magic)")
        version = cursor.i32()
        extra = cursor.i32() if version > 5 else 0
        textures = tuple(cursor.texture(load_textures) for _ in range(cursor.i32()))
        materials = tuple(cursor.material() for _ in range(cursor.i32()))
        root = cursor.node()
        return cls(
            version=version,
            extra=extra,
            textures=textures,
            materials=materials,
            root=root,
            path=source,
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
        if size < 0 or end > len(self.data):
            raise TrackParseError(
                self.path, f"unexpected end of file at offset {self.offset}"
            )
        chunk = self.data[self.offset : end]
        self.offset = end
        return chunk

    def skip(self, size: int) -> None:
        self.take(size)

    def u8(self) -> int:
        return self.take(1)[0]

    def boolean(self) -> bool:
        return self.u8() != 0

    def i32(self) -> int:
        return int(_I32.unpack(self.take(4))[0])

    def u32(self) -> int:
        return int(_U32.unpack(self.take(4))[0])

    def f32(self) -> float:
        return float(_F32.unpack(self.take(4))[0])

    def string(self) -> str:
        length = self.i32()
        if length < 0:
            raise TrackParseError(self.path, f"negative string length at offset {self.offset}")
        return self.take(length).decode("utf-8", "replace").rstrip("\x00")

    def vec2(self) -> tuple[float, float]:
        return _V2.unpack(self.take(_V2.size))

    def vec3(self) -> tuple[float, float, float]:
        return _V3.unpack(self.take(_V3.size))

    def vec4(self) -> tuple[float, float, float, float]:
        return _V4.unpack(self.take(_V4.size))

    def matrix(self) -> tuple[float, ...]:
        return _MATRIX.unpack(self.take(_MATRIX.size))

    def texture(self, load: bool) -> Kn5Texture:
        active = self.i32()
        name = self.string()
        length = self.u32()
        data = self.take(length) if load else None
        if not load:
            self.skip(length)
        return Kn5Texture(name=name, active=active == 1, length=length, data=data)

    def material(self) -> Kn5Material:
        name = self.string()
        shader = self.string()
        blend_mode = self.u8()
        alpha_tested = self.boolean()
        depth_mode = self.i32()
        properties = tuple(self.shader_property() for _ in range(self.i32()))
        mappings = tuple(self.texture_mapping() for _ in range(self.i32()))
        return Kn5Material(
            name=name,
            shader=shader,
            blend_mode=blend_mode,
            alpha_tested=alpha_tested,
            depth_mode=depth_mode,
            properties=properties,
            mappings=mappings,
        )

    def shader_property(self) -> Kn5ShaderProperty:
        return Kn5ShaderProperty(
            name=self.string(),
            value_a=self.f32(),
            value_b=self.vec2(),
            value_c=self.vec3(),
            value_d=self.vec4(),
        )

    def texture_mapping(self) -> Kn5TextureMapping:
        return Kn5TextureMapping(name=self.string(), slot=self.i32(), texture=self.string())

    def vertex(self) -> Kn5Vertex:
        return Kn5Vertex(
            position=self.vec3(),
            normal=self.vec3(),
            uv=self.vec2(),
            tangent=self.vec3(),
        )

    def node(self) -> Kn5Node:
        node_class = Kn5NodeClass(self.i32())
        name = self.string()
        child_count = self.i32()
        active = self.boolean()

        if node_class == Kn5NodeClass.BASE:
            transform = self.matrix()
            children = tuple(self.node() for _ in range(child_count))
            return Kn5Node(
                node_class=node_class,
                name=name,
                active=active,
                transform=transform,
                children=children,
            )

        cast_shadows = self.boolean()
        is_visible = self.boolean()
        is_transparent = self.boolean()

        bones: tuple[Kn5Bone, ...] = ()
        weights: tuple[Kn5Weight, ...] = ()
        if node_class == Kn5NodeClass.SKINNED_MESH:
            bones = tuple(
                Kn5Bone(name=self.string(), transform=self.matrix()) for _ in range(self.u32())
            )

        vertex_count = self.u32()
        vertices = tuple(self.vertex() for _ in range(vertex_count))
        if node_class == Kn5NodeClass.SKINNED_MESH:
            weights = tuple(
                Kn5Weight(weights=self.vec4(), indices=self.vec4())
                for _ in range(vertex_count)
            )

        indices = tuple(self._u16() for _ in range(self.u32()))
        material_id = self.u32()
        layer = self.u32()
        lod_in = self.f32()
        lod_out = self.f32()

        bounds_center: tuple[float, float, float] = (0.0, 0.0, 0.0)
        bounds_radius = 0.0
        is_renderable = True
        if node_class == Kn5NodeClass.MESH:
            bounds_center = self.vec3()
            bounds_radius = self.f32()
            is_renderable = self.boolean()

        children = tuple(self.node() for _ in range(child_count))
        return Kn5Node(
            node_class=node_class,
            name=name,
            active=active,
            children=children,
            cast_shadows=cast_shadows,
            is_visible=is_visible,
            is_transparent=is_transparent,
            is_renderable=is_renderable,
            vertices=vertices,
            weights=weights,
            bones=bones,
            indices=indices,
            material_id=material_id,
            layer=layer,
            lod_in=lod_in,
            lod_out=lod_out,
            bounds_center=bounds_center,
            bounds_radius=bounds_radius,
        )

    def _u16(self) -> int:
        return int(struct.unpack("<H", self.take(2))[0])
