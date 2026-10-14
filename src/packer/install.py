"""Install .kspkg packages into the Assetto Corsa EVO mods folder."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from loguru import logger


def default_mods_dir() -> Path:
    """Locate ``Saved Games/ACE/mods``.

    Honours ``ACE_MODS_DIR`` when set. On WSL the Windows user profile differs
    from ``Path.home()`` (``/home/<user>``), so the ``/mnt/c/Users/<user>``
    profile is detected when present.
    """
    override = os.environ.get("ACE_MODS_DIR")
    if override:
        return Path(override)

    wsl_users = Path("/mnt/c/Users")
    if wsl_users.is_dir():
        current = (os.environ.get("USER") or os.environ.get("USERNAME") or "").lower()
        reserved = {"public", "default", "default user", "all users"}
        candidates = [
            entry
            for entry in wsl_users.iterdir()
            if entry.is_dir() and entry.name.lower() not in reserved
        ]
        profile = next((c for c in candidates if c.name.lower() == current), None)
        if profile is None and len(candidates) == 1:
            profile = candidates[0]
        if profile is not None:
            return profile / "Saved Games" / "ACE" / "mods"

    return Path.home() / "Saved Games" / "ACE" / "mods"


def install_kspkg(source: str | Path, mods_dir: str | Path | None = None) -> Path:
    """Copy a ``.kspkg`` file into the mods folder. Returns the installed path."""
    src = Path(source)
    if not src.is_file():
        raise FileNotFoundError(f"kspkg not found: {src}")
    if src.suffix.lower() != ".kspkg":
        raise ValueError(f"not a .kspkg file: {src}")
    target_dir = Path(mods_dir) if mods_dir is not None else default_mods_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    dest = target_dir / src.name
    shutil.copy2(src, dest)
    logger.info("installed {} -> {}", src.name, dest)
    return dest


def install_directory(source: str | Path, mods_dir: str | Path | None = None) -> Path:
    """Copy an unpacked mod folder into the mods folder (merge-existing)."""
    src = Path(source)
    if not src.is_dir():
        raise FileNotFoundError(f"mod folder not found: {src}")
    target_dir = Path(mods_dir) if mods_dir is not None else default_mods_dir()
    dest = target_dir / src.name
    shutil.copytree(src, dest, dirs_exist_ok=True)
    logger.info("installed folder {} -> {}", src.name, dest)
    return dest
