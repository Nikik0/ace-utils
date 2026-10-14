"""End-to-end AC1 -> ACE track converter.

Builds a complete ACE build tree from an AC1 track folder and packs it into a
``.kspkg``::

    python -m src.converter <ac1_track_path> [--layout NAME] [--out DIR]
        [--track-id ID] [--kspkg PATH] [--install]

Only writes below the chosen output directory (and, with ``--install``, the
``Saved Games/ACE/mods`` folder).
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from loguru import logger

from src import mapping
from src.convert import ai as ai_convert
from src.convert import track_layout as track_layout_convert
from src.convert import ui as ui_convert
from src.convert import zones as zones_convert
from src.convert.material import build_material_data, dds_texture_name, material_texture_resolver
from src.convert.mesh import (
    GroupedMesh,
    WorldMesh,
    build_mesh_data,
    group_by_material,
    iter_world_meshes,
    split_groups,
    tile_groups,
)
from src.convert.scene import (
    StaticMeshEntry,
    add_track_layout_actor,
    build_static_mesh_scene,
    guid_for,
)
from src.convert.spawn import spawn_scenes
from src.convert.surfaces import build_surfaces_scene
from src.convert.texture import (
    AceTexture,
    is_srgb_name,
    null_texture,
    solid_texture,
    texture_from_image_file,
)
from src.convert.track import convert_track
from src.packer.install import default_mods_dir, install_kspkg
from src.packer.kspkg import KspkgWriter
from src.parser.ac1 import AcLayout, AcTrack, parse_track
from src.parser.kn5_model import Kn5Material, Kn5Model


@dataclass(frozen=True, slots=True)
class ConversionResult:
    track_id: str
    layout: str
    build_dir: Path
    kspkg_path: Path
    file_count: int
    total_bytes: int


class TrackConverter:
    def __init__(
        self,
        ac1_track_path: str | Path,
        *,
        layout: str | None = None,
        output_dir: str | Path | None = None,
        track_id: str | None = None,
        kspkg_path: str | Path | None = None,
        install: bool = False,
        mods_dir: str | Path | None = None,
        uv_scale: tuple[float, float] = (1.0, 1.0),
        exclude_materials: str | None = None,
    ) -> None:
        self.ac1_track_path = Path(ac1_track_path)
        self.layout = layout
        self.output_dir = Path(output_dir) if output_dir is not None else None
        self.track_id = track_id
        self.kspkg_path = Path(kspkg_path) if kspkg_path is not None else None
        self.install = install
        self.mods_dir = Path(mods_dir) if mods_dir is not None else None
        self.uv_scale = uv_scale
        self.exclude_materials = exclude_materials

    def run(self) -> ConversionResult:
        track = parse_track(self.ac1_track_path)
        track_id = self.track_id or track.id
        if not track.layouts:
            raise ValueError(f"no layouts in {self.ac1_track_path}")
        layout = self.layout or next(iter(track.layouts))
        if layout not in track.layouts:
            raise ValueError(f"layout {layout!r} not in {sorted(track.layouts)}")
        ac_layout = track.layouts[layout]

        build_dir = self.output_dir or Path("out") / track_id
        if build_dir.exists():
            shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True)

        self._write(
            build_dir,
            mapping.track_file(track_id),
            convert_track(track).SerializeToString(),
        )

        models = self._load_models(track, ac_layout)
        materials = self._materials_by_name(models)

        entries: list[StaticMeshEntry] = []
        self._write_textures(build_dir, track, track_id, materials)
        self._write_materials(build_dir, track_id, materials)
        surface_keys = {s.key.upper() for s in ac_layout.surfaces}
        lift = _compute_lift(models, surface_keys)
        if lift:
            logger.info("track/altitude: raising track by {:.1f} m", lift)
        self._write_meshes(build_dir, track_id, models, entries, surface_keys, lift, self.uv_scale)

        near_entries = [e for e in entries if e.surface_id]
        far_entries = [e for e in entries if not e.surface_id]
        layout_scene = build_static_mesh_scene(near_entries)
        if ac_layout.fast_lane is not None:
            positions = ac_layout.fast_lane.positions
            add_track_layout_actor(
                layout_scene,
                name=f"layout_{layout}",
                layout_path=mapping.as_message_path(
                    mapping.layout_track_layout(track_id, layout)
                ),
                start=positions[0] if positions else (0.0, 0.0, 0.0),
                end=positions[-1] if positions else (0.0, 0.0, 0.0),
            )
            self._write_bytes(
                build_dir,
                mapping.layout_track_layout(track_id, layout),
                track_layout_convert.build_track_layout(ac_layout.fast_lane),
            )
        self._write(
            build_dir,
            mapping.layout_container_scene(track_id, layout),
            layout_scene.SerializeToString(),
        )
        spawn_containers = self._write_spawns(build_dir, track_id, models, lift)
        has_timelines = self._write_timelines(build_dir, track_id, layout, models)
        self._write_root_scene(
            build_dir,
            track_id,
            layout,
            ac_layout,
            near_entries,
            far_entries,
            spawn_containers,
            has_timelines,
        )

        if ac_layout.fast_lane is not None:
            self._write_ai(build_dir, track_id, layout, ac_layout)
            self._write_ui(build_dir, track_id, layout, ac_layout)

        file_count, total_bytes = self._pack(build_dir, track_id)
        kspkg_path = self.kspkg_path or Path("dist") / f"{track_id}.kspkg"
        if self.install:
            install_kspkg(kspkg_path, self.mods_dir or default_mods_dir())
        logger.info("converted {} [{}] -> {} ({} files)", track_id, layout, kspkg_path, file_count)
        return ConversionResult(
            track_id=track_id,
            layout=layout,
            build_dir=build_dir,
            kspkg_path=kspkg_path,
            file_count=file_count,
            total_bytes=total_bytes,
        )

    # -- steps ---------------------------------------------------------------

    def _load_models(self, track: AcTrack, layout: AcLayout) -> list[tuple[str, Kn5Model]]:
        models: list[tuple[str, Kn5Model]] = []
        for model in layout.models:
            path = track.root / model.file
            if path.suffix.lower() != ".kn5" or not path.is_file():
                continue
            models.append((model.file, Kn5Model.read(path, load_textures=False)))
        return models

    def _materials_by_name(self, models: list[tuple[str, Kn5Model]]) -> dict[str, Kn5Material]:
        materials: dict[str, Kn5Material] = {}
        for _, model in models:
            for material in model.materials:
                materials.setdefault(material.name, material)
        return materials

    def _write_textures(
        self,
        build_dir: Path,
        track: AcTrack,
        track_id: str,
        materials: dict[str, Kn5Material],
    ) -> None:
        sources = self._find_dds(track.root)
        referenced: set[str] = {"NULL"}
        for material in materials.values():
            for name in ("txDiffuse", "txNormal"):
                mapping_entry = material.mapping(name)
                if mapping_entry:
                    referenced.add(dds_texture_name(mapping_entry.texture))

        fallback = null_texture()
        for name in sorted(referenced):
            source = sources.get(name.lower())
            if source is not None:
                texture = texture_from_image_file(source, max_size=1024, srgb=is_srgb_name(name))
            else:
                texture = fallback
            self._write_texture(build_dir, track_id, name, texture)

    def _write_texture(
        self, build_dir: Path, track_id: str, name: str, texture: AceTexture
    ) -> None:
        self._write_texture_asset(build_dir, mapping.texture_path(track_id, name), texture)

    def _write_texture_asset(
        self, build_dir: Path, base_arc: str, texture: AceTexture
    ) -> None:
        self._write_bytes(build_dir, base_arc, texture.metadata.SerializeToString())
        self._write_bytes(build_dir, mapping.texture_mips_for(base_arc), texture.mip_data)

    def _write_materials(
        self, build_dir: Path, track_id: str, materials: dict[str, Kn5Material]
    ) -> None:
        resolver = material_texture_resolver(track_id)
        names = list(materials)
        for index, name in enumerate(names):
            material = materials[name]
            render_order = index - len(names)
            message = build_material_data(material, resolver, render_order=render_order)
            path = mapping.material_path(track_id, mapping.sanitize_name(name))
            self._write_bytes(build_dir, path, message.SerializeToString())

    def _write_meshes(
        self,
        build_dir: Path,
        track_id: str,
        models: list[tuple[str, Kn5Model]],
        entries: list[StaticMeshEntry],
        surface_keys: set[str],
        lift: float = 0.0,
        uv_scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        for filename, model in models:
            physics_only = "collision" in filename.lower()
            world = iter_world_meshes(model, lift=lift)
            cell = max(32.0, _world_span(world) / 8.0)
            for group in split_groups(tile_groups(group_by_material(world), cell)):
                if self.exclude_materials and re.search(
                    self.exclude_materials, group.material, re.IGNORECASE
                ):
                    continue
                material_path = mapping.as_message_path(
                    mapping.material_path(track_id, mapping.sanitize_name(group.material))
                )
                mesh_path = mapping.mesh_path(track_id, group.name)
                mesh_message = build_mesh_data(group, material_path, uv_scale=uv_scale)
                self._write_bytes(build_dir, mesh_path, mesh_message.SerializeToString())
                bounds = _group_bounds(group)
                upper = group.material.upper()
                surface = next(
                    (
                        key
                        for key in sorted(surface_keys, key=len, reverse=True)
                        if upper == key or upper.startswith(f"{key} ")
                    ),
                    "",
                )
                entries.append(
                    StaticMeshEntry(
                        name=group.name,
                        mesh_path=mapping.as_message_path(mesh_path),
                        surface_id=surface if surface in surface_keys else "",
                        bounds_min=(bounds[0], bounds[2]),
                        bounds_max=(bounds[3], bounds[5]),
                        physics_only=physics_only,
                    )
                )

    def _write_root_scene(
        self,
        build_dir: Path,
        track_id: str,
        layout: str,
        ac_layout: AcLayout,
        entries: list[StaticMeshEntry],
        far_entries: list[StaticMeshEntry],
        spawn_containers: list[str],
        has_timelines: bool,
    ) -> None:
        scene = build_surfaces_scene(ac_layout.surfaces)
        actors = cast(Any, scene).scene_actors

        for name in spawn_containers:
            self._add_container(actors, track_id, name, on_demand=False)
        if has_timelines:
            self._add_container(actors, track_id, "timelines", on_demand=False)
        self._add_container(actors, track_id, f"layout_{layout}", on_demand=True)

        # EvoForge places meshes without a surface key directly in the root scene.
        for entry in far_entries:
            far = actors.add()
            far.name = entry.name
            far.enabled = True
            far.tags.append("SMesh")
            far.transform.scale.x = 1.0
            far.transform.scale.y = 1.0
            far.transform.scale.z = 1.0
            far.guid = guid_for(entry.name)
            far.bounds.bounds_min.x, far.bounds.bounds_min.y = entry.bounds_min
            far.bounds.bounds_max.x, far.bounds.bounds_max.y = entry.bounds_max
            far.bounds.active_distance = entry.active_distance
            far.active_distance_override = 100000.0
            far.actor.static_mesh.mesh_path = entry.mesh_path

        all_entries = entries + far_entries
        if all_entries:
            xs = [e.bounds_min[0] for e in all_entries] + [e.bounds_max[0] for e in all_entries]
            zs = [e.bounds_min[1] for e in all_entries] + [e.bounds_max[1] for e in all_entries]
            gi = actors.add()
            gi.name = "global_illumination_bounds"
            gi.enabled = True
            gi.tags.append("GlobalIllumination Bounds")
            gi.transform.position.x = (min(xs) + max(xs)) / 2
            gi.transform.position.z = (min(zs) + max(zs)) / 2
            gi.transform.scale.x = max(1.0, max(xs) - min(xs))
            gi.transform.scale.y = 320.0
            gi.transform.scale.z = max(1.0, max(zs) - min(zs))
            gi.guid = guid_for(gi.name)
            gi.active_distance_override = 100000.0
            gi.actor.macro_irradiance_volume.SetInParent()

            # Irradiance probe grid (EvoForge emits one per macro cell).
            nx = max(1, min(24, int((max(xs) - min(xs)) // 64.0) + 1))
            nz = max(1, min(24, int((max(zs) - min(zs)) // 64.0) + 1))
            for ix in range(nx):
                for iz in range(nz):
                    probe = actors.add()
                    probe.name = f"irradiance_volume_{ix}_{iz}"
                    probe.enabled = True
                    probe.tags.append("Irradiance Probe")
                    probe.transform.position.x = min(xs) + (ix + 0.5) * (max(xs) - min(xs)) / nx
                    probe.transform.position.z = min(zs) + (iz + 0.5) * (max(zs) - min(zs)) / nz
                    probe.transform.scale.x = 1.0
                    probe.transform.scale.y = 1.0
                    probe.transform.scale.z = 1.0
                    probe.guid = guid_for(probe.name)
                    probe.bounds.active_distance = 250.0
                    probe.active_distance_override = 100000.0
                    iv = probe.actor.irradiance_volume
                    iv.grid_x = ix
                    iv.grid_z = iz

        info = actors.add()
        info.name = "track_info"
        info.enabled = True
        info.tags.append("Track Info")
        info.transform.scale.x = 1.0
        info.transform.scale.y = 1.0
        info.transform.scale.z = 1.0
        info.guid = guid_for("track_info")
        info.active_distance_override = 100000.0
        scene_info = info.actor.track_info.sceneInfo
        scene_info.trackType = 1  # SceneTrackTypeEnum_Circuit
        scene_info.displayName = ac_layout.ui.name if ac_layout.ui else track_id

        self._write(
            build_dir, mapping.track_scene(track_id), scene.SerializeToString()
        )

    @staticmethod
    def _add_container(
        actors: Any, track_id: str, name: str, *, on_demand: bool
    ) -> None:
        container = actors.add()
        container.name = name
        container.enabled = True
        container.tags.append("Container")
        container.transform.scale.x = 1.0
        container.transform.scale.y = 1.0
        container.transform.scale.z = 1.0
        container.guid = guid_for(name)
        container.bounds.active_distance = 250.0
        container.active_distance_override = 100000.0
        container.actor.container.scene_path = mapping.as_message_path(
            mapping.container_scene(track_id, name)
        )
        container.actor.container.loaded_on_demand = on_demand

    def _write_spawns(
        self,
        build_dir: Path,
        track_id: str,
        models: list[tuple[str, Kn5Model]],
        lift: float = 0.0,
    ) -> list[str]:
        written: list[str] = []
        for _, model in models:
            for name, scene in spawn_scenes(model, lift=lift).items():
                if not cast(Any, scene).scene_actors:
                    continue
                short = f"spawnpoints_{name}"
                self._write(
                    build_dir,
                    mapping.container_scene(track_id, short),
                    scene.SerializeToString(),
                )
                written.append(short)
        return written

    def _write_timelines(
        self, build_dir: Path, track_id: str, layout: str, models: list[tuple[str, Kn5Model]]
    ) -> bool:
        for _, model in models:
            scene = zones_convert.timelines_scene_for(model, layout)
            if scene is not None:
                self._write(
                    build_dir,
                    mapping.container_scene(track_id, "timelines"),
                    scene.SerializeToString(),
                )
                return True
        return False

    def _write_ai(
        self, build_dir: Path, track_id: str, layout: str, ac_layout: AcLayout
    ) -> None:
        assert ac_layout.fast_lane is not None
        spline = ac_layout.fast_lane
        self._write_bytes(
            build_dir,
            mapping.layout_aispline(track_id, layout),
            ai_convert.build_ai_spline(
                spline, track_id=track_id, layout=layout
            ).SerializeToString(),
        )
        self._write_bytes(
            build_dir,
            mapping.layout_control_points(track_id, layout),
            ai_convert.build_control_points(
                spline, track_id=track_id, layout=layout
            ).SerializeToString(),
        )
        self._write_bytes(
            build_dir,
            mapping.layout_reference(track_id, layout),
            ai_convert.build_reference(spline).SerializeToString(),
        )

    def _write_ui(
        self, build_dir: Path, track_id: str, layout: str, ac_layout: AcLayout
    ) -> None:
        assert ac_layout.fast_lane is not None
        svg, layout_image = ui_convert.build_map_svg(ac_layout.fast_lane)
        self._write_bytes(build_dir, mapping.ui_map_svg(track_id), svg)
        self._write_bytes(build_dir, mapping.trackmap_svg(track_id, layout), svg)
        previews = ui_convert.build_previews_json(track_id, layout, layout_image=layout_image)
        self._write_bytes(build_dir, mapping.ui_previews_json(track_id), previews)

        photo = self._ui_image(ac_layout, "preview.png", (32, 32, 48))
        map_image = self._ui_image(ac_layout, "map.png", (16, 16, 16))
        self._write_texture_asset(build_dir, mapping.ui_photo_texture(track_id), photo)
        self._write_texture_asset(build_dir, mapping.ui_map_texture(track_id), map_image)
        self._write_texture_asset(
            build_dir, mapping.track_preview_texture(track_id, layout), photo
        )
        self._write_texture_asset(
            build_dir, mapping.trackmap_texture(track_id, layout), map_image
        )

    @staticmethod
    def _ui_image(ac_layout: AcLayout, filename: str, fallback: tuple[int, int, int]) -> AceTexture:
        ui_dir = ac_layout.directory.parent / "ui" / ac_layout.name
        candidates = [ui_dir / filename, ac_layout.directory / filename]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    return texture_from_image_file(candidate)
                except Exception as exc:
                    logger.warning("UI image {!s} unusable: {}", candidate, exc)
        return solid_texture(size=32, rgba=(*fallback, 255))

    def _pack(self, build_dir: Path, track_id: str) -> tuple[int, int]:
        kspkg_path = self.kspkg_path or Path("dist") / f"{track_id}.kspkg"
        kspkg_path.parent.mkdir(parents=True, exist_ok=True)
        writer = KspkgWriter()
        count = writer.add_tree(build_dir)
        writer.write(kspkg_path)
        total = sum(p.stat().st_size for p in build_dir.rglob("*") if p.is_file())
        return count, total

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _find_dds(root: Path) -> dict[str, Path]:
        sources: dict[str, Path] = {}
        for path in root.rglob("*.dds"):
            sources.setdefault(_texture_name(path.name), path)
        return sources
    def _write(self, build_dir: Path, arc_path: str, data: bytes) -> None:
        self._write_bytes(build_dir, arc_path, data)

    @staticmethod
    def _write_bytes(build_dir: Path, arc_path: str, data: bytes) -> None:
        target = build_dir / arc_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _texture_name(filename: str) -> str:
    return dds_texture_name(filename).lower()


def _group_bounds(group: GroupedMesh) -> tuple[float, float, float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for part in group.parts:
        for vertex in part.vertices:
            xs.append(vertex.position[0])
            ys.append(vertex.position[1])
            zs.append(vertex.position[2])
    if not xs:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (min(xs), min(ys), min(zs), max(xs), max(ys), max(zs))


def _compute_lift(
    models: list[tuple[str, Kn5Model]], surface_keys: set[str]
) -> float:
    """EvoForge ``track/altitude``: raise the track so its lowest drivable surface
    sits at Y=0. Drivable = material matching a surface key, excluding walls/rails."""
    if not surface_keys:
        return 0.0
    wall = re.compile(r"WALL|BARRIER|FENCE|RAIL|GUARD", re.IGNORECASE)
    lowest = float("inf")
    for _, model in models:
        for mesh in iter_world_meshes(model):
            upper = mesh.material.upper()
            surface = next(
                (
                    key
                    for key in sorted(surface_keys, key=len, reverse=True)
                    if upper == key or upper.startswith(f"{key} ")
                ),
                "",
            )
            if not surface or wall.search(surface):
                continue
            for vertex in mesh.vertices:
                if vertex.position[1] < lowest:
                    lowest = vertex.position[1]
    if lowest != float("inf") and lowest < -0.05:
        return -lowest
    return 0.0


def _world_span(meshes: list[WorldMesh]) -> float:
    """Largest horizontal extent of a model's baked geometry, in metres."""
    xs: list[float] = []
    zs: list[float] = []
    for mesh in meshes:
        for vertex in mesh.vertices:
            xs.append(vertex.position[0])
            zs.append(vertex.position[2])
    if not xs:
        return 0.0
    return max(max(xs) - min(xs), max(zs) - min(zs))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="src.converter", description="Convert an AC1 track to ACE"
    )
    parser.add_argument("track", type=Path)
    parser.add_argument("--layout", default=None)
    parser.add_argument("--out", type=Path, default=None, help="build tree directory")
    parser.add_argument("--track-id", default=None)
    parser.add_argument("--kspkg", type=Path, default=None)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--mods", type=Path, default=None)
    parser.add_argument("--uv-scale", default="1,1", help="texture coordinate scale U,V")
    parser.add_argument("--exclude-materials", default=None, help="regex of materials to skip")
    args = parser.parse_args(argv)

    uv_parts = [float(part) for part in args.uv_scale.split(",")]
    uv_scale = (uv_parts[0], uv_parts[1]) if len(uv_parts) == 2 else (uv_parts[0], uv_parts[0])

    result = TrackConverter(
        args.track,
        layout=args.layout,
        output_dir=args.out,
        track_id=args.track_id,
        kspkg_path=args.kspkg,
        install=args.install,
        mods_dir=args.mods,
        uv_scale=uv_scale,
        exclude_materials=args.exclude_materials,
    ).run()
    print(f"{result.kspkg_path} ({result.file_count} files, {result.total_bytes / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
