"""Convert AC1 KN5 model geometry into ACE ``MeshData`` assets.

KN5 transforms use the DirectX row-vector convention: a point is transformed as
``[x y z 1] . M`` (row-major, translation in elements 12..14). Mesh nodes carry
no transform of their own; their world matrix is the product of enclosing base
node matrices. Vertices are baked into world space so scene actors stay at
identity, matching the reference packages.
"""

from __future__ import annotations

import math
import re
import struct
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, cast

from google.protobuf.message import Message

from src.ace_formats import new_message
from src.mapping import sanitize_name
from src.parser.kn5_model import Kn5Model, Kn5Node, Kn5NodeClass, Kn5Vertex

IDENTITY: tuple[float, ...] = (
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.0, 0.0, 0.0, 1.0,
)

# EvoForge drops flat billboard/impostor submeshes (AC1 foliage) and rebuilds
# vegetation as native 3D assets.
_BILLBOARD = re.compile(r"billboard|impostor", re.IGNORECASE)
_FOLIAGE = re.compile(r"tree|bush|grass|foliage|plant|flat|shrub", re.IGNORECASE)

# Diagnostic switch: when True, meshes carry only positions/normals/uv/indices.
MINIMAL_ATTRS = False

# EvoForge classifies flat foliage/impostor meshes (>80 % of vertical area in two
# of six orientation bins) and drops them, rebuilding vegetation as native assets.
DROP_BILLBOARDS = True

# EvoForge's default vertex-colour word (field 13), written as 4 x uint32 per
# vertex. Reinterpreted as floats this is ~0.00392 (== 1/255).
_COLOR_VALUE = struct.unpack("<f", struct.pack("<I", 0x3B808081))[0]


def mat_mul(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    """Standard row-major 4x4 product ``left . right``."""
    out = [0.0] * 16
    for row in range(4):
        for col in range(4):
            out[row * 4 + col] = sum(left[row * 4 + k] * right[k * 4 + col] for k in range(4))
    return tuple(out)


def transform_point(
    matrix: tuple[float, ...], point: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z = point
    return (
        x * matrix[0] + y * matrix[4] + z * matrix[8] + matrix[12],
        x * matrix[1] + y * matrix[5] + z * matrix[9] + matrix[13],
        x * matrix[2] + y * matrix[6] + z * matrix[10] + matrix[14],
    )


def transform_normal(
    matrix: tuple[float, ...], normal: tuple[float, float, float]
) -> tuple[float, float, float]:
    x, y, z = normal
    nx = x * matrix[0] + y * matrix[4] + z * matrix[8]
    ny = x * matrix[1] + y * matrix[5] + z * matrix[9]
    nz = x * matrix[2] + y * matrix[6] + z * matrix[10]
    length = (nx * nx + ny * ny + nz * nz) ** 0.5
    if length < 1e-8:
        return (0.0, 0.0, 1.0)
    return (nx / length, ny / length, nz / length)


def _unit(
    x: float, y: float, z: float, fallback: tuple[float, float, float]
) -> tuple[float, float, float]:
    length = (x * x + y * y + z * z) ** 0.5
    if length < 1e-9:
        return fallback
    return (x / length, y / length, z / length)


def _weld_vectors(
    arrays: list[tuple[list[float], list[float], list[float]]],
    positions: list[float],
    angle_deg: float = 45.0,
) -> None:
    """EvoForge's ``weldNormals``: smooth per-vertex vectors for vertices that
    share a position (rounded to 0.0001) and point within ``angle_deg`` of each
    other. Hard edges (larger angle) stay faceted."""
    count = len(positions) // 3
    if count <= 0:
        return
    cos_thresh = math.cos(angle_deg * math.pi / 180.0)
    keys = [
        (
            round(positions[i * 3] * 10000),
            round(positions[i * 3 + 1] * 10000),
            round(positions[i * 3 + 2] * 10000),
        )
        for i in range(count)
    ]
    for ax, ay, az in arrays:
        sums: dict[tuple[int, int, int], list[float]] = {}
        for i in range(count):
            total = sums.get(keys[i])
            if total is None:
                sums[keys[i]] = [ax[i], ay[i], az[i]]
            else:
                total[0] += ax[i]
                total[1] += ay[i]
                total[2] += az[i]
        for i in range(count):
            total = sums[keys[i]]
            total_len = math.hypot(total[0], total[1], total[2])
            own_len = math.hypot(ax[i], ay[i], az[i])
            if total_len < 1e-12 or own_len < 1e-12:
                continue
            cosine = (
                total[0] * ax[i] + total[1] * ay[i] + total[2] * az[i]
            ) / (total_len * own_len)
            if cosine < cos_thresh:
                continue
            ax[i], ay[i], az[i] = total[0], total[1], total[2]


def recompute_normals_tangents(
    positions: list[float], uvs: list[float], indices: list[int]
) -> tuple[list[float], list[float]]:
    """Recompute vertex normals (area-weighted) and tangents (UV gradient).

    Mirrors EvoForge's ``hd``: the KN5 normals/tangents are not trusted for track
    meshes; both are derived from the baked geometry and UVs.
    """
    count = len(positions) // 3
    nxs = [0.0] * count
    nys = [0.0] * count
    nzs = [0.0] * count
    txs = [0.0] * count
    tys = [0.0] * count
    tzs = [0.0] * count
    bxs = [0.0] * count
    bys = [0.0] * count
    bzs = [0.0] * count
    for i in range(0, len(indices) - 2, 3):
        a, b, c = indices[i], indices[i + 1], indices[i + 2]
        ax, ay, az = positions[a * 3], positions[a * 3 + 1], positions[a * 3 + 2]
        bx, by, bz = positions[b * 3], positions[b * 3 + 1], positions[b * 3 + 2]
        cx, cy, cz = positions[c * 3], positions[c * 3 + 1], positions[c * 3 + 2]
        e1x, e1y, e1z = bx - ax, by - ay, bz - az
        e2x, e2y, e2z = cx - ax, cy - ay, cz - az
        fx, fy, fz = e1y * e2z - e1z * e2y, e1z * e2x - e1x * e2z, e1x * e2y - e1y * e2x
        for v in (a, b, c):
            nxs[v] += fx
            nys[v] += fy
            nzs[v] += fz
        au, av = uvs[a * 2], uvs[a * 2 + 1]
        bu, bv = uvs[b * 2], uvs[b * 2 + 1]
        cu, cv = uvs[c * 2], uvs[c * 2 + 1]
        du1, dv1 = bu - au, bv - av
        du2, dv2 = cu - au, cv - av
        det = du1 * dv2 - du2 * dv1
        if abs(det) < 1e-12:
            continue
        r = 1.0 / det
        sx = (dv2 * e1x - dv1 * e2x) * r
        sy = (dv2 * e1y - dv1 * e2y) * r
        sz = (dv2 * e1z - dv1 * e2z) * r
        for v in (a, b, c):
            txs[v] += sx
            tys[v] += sy
            tzs[v] += sz
        bx = (du1 * e2x - du2 * e1x) * r
        by = (du1 * e2y - du2 * e1y) * r
        bz = (du1 * e2z - du2 * e1z) * r
        for v in (a, b, c):
            bxs[v] += bx
            bys[v] += by
            bzs[v] += bz

    _weld_vectors([(nxs, nys, nzs), (txs, tys, tzs), (bxs, bys, bzs)], positions)

    normals: list[float] = []
    tangents: list[float] = []
    for v in range(count):
        nx, ny, nz = _unit(nxs[v], nys[v], nzs[v], (0.0, 1.0, 0.0))
        dot = txs[v] * nx + tys[v] * ny + tzs[v] * nz
        ox, oy, oz = txs[v] - nx * dot, tys[v] - ny * dot, tzs[v] - nz * dot
        tx, ty, tz = _unit(ox, oy, oz, (1.0, 0.0, 0.0) if abs(nx) < 0.9 else (0.0, 1.0, 0.0))
        cx = ny * tz - nz * ty
        cy = nz * tx - nx * tz
        cz = nx * ty - ny * tx
        w = -1.0 if (cx * bxs[v] + cy * bys[v] + cz * bzs[v]) < 0.0 else 1.0
        normals.extend((nx, ny, nz))
        tangents.extend((tx, ty, tz, w))
    return normals, tangents


@dataclass(frozen=True, slots=True)
class WorldMesh:
    """One KN5 mesh node with vertices baked into world space."""

    name: str
    material: str
    vertices: tuple[Kn5Vertex, ...]
    indices: tuple[int, ...]
    cast_shadows: bool
    lod_out: float
    flip_winding: bool = False


@dataclass(frozen=True, slots=True)
class GroupedMesh:
    """Mesh nodes that share one material, ready to serialize as one asset."""

    name: str
    material: str
    parts: tuple[WorldMesh, ...]

    @property
    def cast_shadows(self) -> bool:
        return any(part.cast_shadows for part in self.parts)

    @property
    def lod_out(self) -> float:
        return max((part.lod_out for part in self.parts), default=0.0)


def _material_name(model: Kn5Model, material_id: int) -> str:
    if 0 <= material_id < len(model.materials):
        return model.materials[material_id].name
    return f"MATERIAL_{material_id}"


def _determinant3(matrix: tuple[float, ...]) -> float:
    return (
        matrix[0] * (matrix[5] * matrix[10] - matrix[6] * matrix[9])
        - matrix[4] * (matrix[1] * matrix[10] - matrix[2] * matrix[9])
        + matrix[8] * (matrix[1] * matrix[6] - matrix[2] * matrix[5])
    )


def _billboard_ratio(vertices: tuple[Kn5Vertex, ...], indices: tuple[int, ...]) -> float:
    """Port of EvoForge's ``SmV``: share of vertical area in the top two of six
    orientation bins. ``>= 0.8`` means the mesh is a flat billboard/impostor."""
    import math

    if len(indices) < 3:
        return 1.0
    bins = [0.0] * 6
    total = 0.0
    vertical = 0.0
    for t in range(0, len(indices) - 2, 3):
        i, p, a = indices[t], indices[t + 1], indices[t + 2]
        ix, iy, iz = vertices[i].position
        px, py, pz = vertices[p].position
        ax, ay, az = vertices[a].position
        jx, jy, jz = px - ix, py - iy, pz - iz
        kx, ky, kz = ax - ix, ay - iy, az - iz
        cx, cy, cz = jy * kz - jz * ky, jz * kx - jx * kz, jx * ky - jy * kx
        length = math.sqrt(cx * cx + cy * cy + cz * cz)
        if length <= 1e-9:
            continue
        area = length / 2.0
        total += area
        if abs(cy / length) >= 0.5:
            continue
        vertical += area
        angle = math.atan2(cx, cz)
        if angle < 0.0:
            angle += math.pi
        bins[min(5, int(angle / math.pi * 6))] += area
    if total <= 0.0 or vertical <= 0.0:
        return 1.0
    top = sorted(bins, reverse=True)
    return (top[0] + top[1]) / vertical


def iter_world_meshes(model: Kn5Model, *, lift: float = 0.0) -> list[WorldMesh]:
    """Bake every mesh node of a KN5 model into world space.

    ``lift`` raises every vertex along Y (EvoForge's ``track/altitude`` step).
    """
    result: list[WorldMesh] = []

    def walk(node: Kn5Node, parent: tuple[float, ...]) -> None:
        world = (
            mat_mul(tuple(node.transform), parent)
            if node.node_class == Kn5NodeClass.BASE
            else parent
        )
        if node.is_mesh and not _BILLBOARD.search(node.name):
            baked = [
                (
                    transform_point(world, vertex.position),
                    transform_normal(world, vertex.normal),
                    vertex.uv,
                    vertex.tangent,
                )
                for vertex in node.vertices
            ]
            vertices = tuple(
                Kn5Vertex(
                    position=(position[0], position[1] + lift, position[2]),
                    normal=normal,
                    uv=uv,
                    tangent=tangent,
                )
                for position, normal, uv, tangent in baked
            )
            material = _material_name(model, node.material_id)
            foliage = bool(_FOLIAGE.search(node.name) or _FOLIAGE.search(material))
            if (
                DROP_BILLBOARDS
                and foliage
                and _billboard_ratio(vertices, tuple(node.indices)) >= 0.8
            ):
                for child in node.children:
                    walk(child, world)
                return
            result.append(
                WorldMesh(
                    name=node.name,
                    material=_material_name(model, node.material_id),
                    vertices=vertices,
                    indices=tuple(node.indices),
                    cast_shadows=node.cast_shadows,
                    lod_out=node.lod_out,
                    flip_winding=_determinant3(world) < 0.0,
                )
            )
        for child in node.children:
            walk(child, world)

    walk(model.root, IDENTITY)
    return result


def group_by_material(meshes: list[WorldMesh]) -> list[GroupedMesh]:
    """Merge world meshes that share a material into single assets."""
    groups: OrderedDict[str, list[WorldMesh]] = OrderedDict()
    for mesh in meshes:
        groups.setdefault(mesh.material, []).append(mesh)
    return [
        GroupedMesh(name=sanitize_name(material), material=material, parts=tuple(parts))
        for material, parts in groups.items()
    ]


def _split_part(part: WorldMesh, max_verts: int) -> list[WorldMesh]:
    """Split one mesh into chunks of at most ``max_verts`` vertices."""
    parts: list[WorldMesh] = []
    new_verts: list[Kn5Vertex] = []
    new_idx: list[int] = []
    vmap: dict[int, int] = {}

    def flush() -> None:
        nonlocal new_verts, new_idx, vmap
        if new_idx:
            parts.append(
                WorldMesh(
                    name=part.name,
                    material=part.material,
                    vertices=tuple(new_verts),
                    indices=tuple(new_idx),
                    cast_shadows=part.cast_shadows,
                    lod_out=part.lod_out,
                    flip_winding=part.flip_winding,
                )
            )
        new_verts, new_idx, vmap = [], [], {}

    for t in range(0, len(part.indices), 3):
        tri = part.indices[t : t + 3]
        added = {vi for vi in tri if vi not in vmap}
        if new_verts and len(new_verts) + len(added) > max_verts:
            flush()
        for vi in tri:
            if vi not in vmap:
                vmap[vi] = len(new_verts)
                new_verts.append(part.vertices[vi])
            new_idx.append(vmap[vi])
    flush()
    return parts


def split_groups(groups: list[GroupedMesh], max_verts: int = 12000) -> list[GroupedMesh]:
    """Cap each mesh group at ``max_verts`` vertices (EvoForge splits at 12000)."""
    out: list[GroupedMesh] = []
    for group in groups:
        index = 0
        chunk: list[WorldMesh] = []
        chunk_verts = 0
        for part in group.parts:
            sub_parts = _split_part(part, max_verts) if len(part.vertices) > max_verts else [part]
            for sub in sub_parts:
                if chunk and chunk_verts + len(sub.vertices) > max_verts:
                    name = f"{group.name}_{index}"
                    out.append(GroupedMesh(name=name, material=group.material, parts=tuple(chunk)))
                    index += 1
                    chunk, chunk_verts = [], 0
                chunk.append(sub)
                chunk_verts += len(sub.vertices)
        if chunk:
            name = f"{group.name}_{index}"
            out.append(GroupedMesh(name=name, material=group.material, parts=tuple(chunk)))
    return out


def tile_groups(groups: list[GroupedMesh], cell_size: float) -> list[GroupedMesh]:
    """Split each group into ``cell_size`` metre grid tiles (EvoForge's ``ROAD_x_z``)."""
    import math

    out: list[GroupedMesh] = []
    for group in groups:
        tiles: OrderedDict[tuple[int, int], list[WorldMesh]] = OrderedDict()
        for part in group.parts:
            buckets: dict[tuple[int, int], list[int]] = {}
            for t in range(0, len(part.indices), 3):
                tri = part.indices[t : t + 3]
                cx = sum(part.vertices[v].position[0] for v in tri) / 3.0
                cz = sum(part.vertices[v].position[2] for v in tri) / 3.0
                key = (math.floor(cx / cell_size), math.floor(cz / cell_size))
                buckets.setdefault(key, []).extend(tri)
            for key, tris in buckets.items():
                remap: dict[int, int] = {}
                verts: list[Kn5Vertex] = []
                idx: list[int] = []
                for vi in tris:
                    j = remap.get(vi)
                    if j is None:
                        j = len(verts)
                        remap[vi] = j
                        verts.append(part.vertices[vi])
                    idx.append(j)
                tiles.setdefault(key, []).append(
                    WorldMesh(
                        name=part.name,
                        material=part.material,
                        vertices=tuple(verts),
                        indices=tuple(idx),
                        cast_shadows=part.cast_shadows,
                        lod_out=part.lod_out,
                        flip_winding=part.flip_winding,
                    )
                )
        for (tx, tz), parts in tiles.items():
            out.append(
                GroupedMesh(
                    name=f"{group.name}_{tx}_{tz}",
                    material=group.material,
                    parts=tuple(parts),
                )
            )
    return out


def _lod_out(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    *,
    cull_height: float = 0.0,
) -> float:
    """Port of EvoForge's ``Ib``: derive the LOD-out distance from the mesh bounds."""
    def span(axis: int) -> float:
        value = bounds_max[axis] - bounds_min[axis]
        return value if math.isfinite(value) and value > 0.0 else 0.0

    z = cull_height if cull_height > 0.0 else span(1)
    h = max(span(0), span(2))
    k = min(2000.0, round(min(h * 1.5, h + 250.0)))
    if z >= 8.0:
        return max(min(2500.0, max(800.0, round(z * 25.0))), k)
    if h < 150.0:
        return min(400.0, 250.0 + round(h))
    return max(250.0, k)


def _has_large_uvs(group: GroupedMesh) -> bool:
    """EvoForge sets ``fullPrecisionUvs`` when any UV coordinate exceeds +-8."""
    for part in group.parts:
        for vertex in part.vertices:
            u, v = vertex.uv
            if u > 8.0 or u < -8.0 or v > 8.0 or v < -8.0:
                return True
    return False


def build_mesh_data(
    group: GroupedMesh,
    material_path: str,
    *,
    uv_scale: tuple[float, float] = (1.0, 1.0),
    ground_level: bool = False,
    full_precision_uvs: bool = False,
) -> Message:
    """Serialize a material group as a ``MeshData`` message.

    The ACE vertex layout is fixed: position(3), normal(3), uv(2), tangent(4),
    colour(4), uv1(2), uv2(2), uv3(2). All arrays must be present with matching
    counts or the engine's resource loader faults. ``uv_scale`` multiplies the
    texture coordinates (AC1 UVs repeat more densely than ACE expects).
    """
    scale_u, scale_v = uv_scale
    msg = cast(Any, new_message(".mesh"))
    msg.isVisible = True
    msg.isRenderable = True
    msg.irradianceVolumeGroundLevel = ground_level
    msg.useFullPrecisionUvs = full_precision_uvs

    lod = msg.lods.add()

    batch = lod.batches.add()
    batch.name = group.name
    batch.material = material_path

    weld: dict[tuple[Any, ...], int] = {}
    emitted: list[Kn5Vertex] = []
    for part in group.parts:
        local: list[int] = []
        for vertex in part.vertices:
            key = (vertex.position, vertex.normal, vertex.uv, vertex.tangent)
            index = weld.get(key)
            if index is None:
                index = len(emitted)
                weld[key] = index
                emitted.append(vertex)
            local.append(index)

        if part.flip_winding:
            for tri in range(0, len(part.indices), 3):
                a, b, c = part.indices[tri : tri + 3]
                lod.indices.extend((local[a], local[c], local[b]))
        else:
            lod.indices.extend(local[index] for index in part.indices)

    batch.indexCount = len(lod.indices)

    positions: list[float] = []
    uvs: list[float] = []
    for vertex in emitted:
        positions.extend(vertex.position)
        uvs.extend((vertex.uv[0] * scale_u, vertex.uv[1] * scale_v))
    lod.positions.extend(positions)
    lod.texcoords.extend(uvs)
    vertex_count = len(emitted)
    if not MINIMAL_ATTRS:
        normals, tangents = recompute_normals_tangents(positions, uvs, list(lod.indices))
        lod.normals.extend(normals)
        lod.tangents.extend(tangents)
        lod.colors.extend([_COLOR_VALUE] * (vertex_count * 4))
        lod.texcoords1.extend([0.0] * (vertex_count * 2))
        lod.texcoords2.extend([0.0] * (vertex_count * 2))
        lod.texcoords3.extend([0.0] * (vertex_count * 2))
    else:
        for vertex in emitted:
            lod.normals.extend(vertex.normal)
            lod.tangents.extend((*vertex.tangent, 1.0))

    if lod.positions:
        xs = lod.positions[0::3]
        ys = lod.positions[1::3]
        zs = lod.positions[2::3]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        min_z, max_z = min(zs), max(zs)
        msg.boundsMin.x, msg.boundsMin.y, msg.boundsMin.z = min_x, min_y, min_z
        msg.boundsMax.x, msg.boundsMax.y, msg.boundsMax.z = max_x, max_y, max_z
        msg.lodOut = _lod_out((min_x, min_y, min_z), (max_x, max_y, max_z))
        lod.boundsMin.x, lod.boundsMin.y, lod.boundsMin.z = min_x, min_y, min_z
        lod.boundsMax.x, lod.boundsMax.y, lod.boundsMax.z = max_x, max_y, max_z

    return cast(Message, msg)
