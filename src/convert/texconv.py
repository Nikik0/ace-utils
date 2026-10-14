"""BC7/BC4 texture compression via Microsoft DirectXTex ``texconv``.

ACE's native texture format is BC7 (colour/alpha) and BC4 (single channel);
AC1 textures are BC1/BC3, which the engine renders differently. When
``texconv`` is available we compress to BC7 to match. The executable path is
taken from ``TEXCONV_PATH`` or found on ``PATH``; on WSL Windows paths are used
so the Windows binary can read the staged files.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

BC7_SRGB = "BC7_UNORM_SRGB"
BC7_LINEAR = "BC7_UNORM"
BC4_LINEAR = "BC4_UNORM"


def find_texconv() -> Path | None:
    """Locate a ``texconv`` executable, or None if unavailable."""
    override = os.environ.get("TEXCONV_PATH")
    if override and Path(override).is_file():
        return Path(override)
    found = shutil.which("texconv") or shutil.which("texconv.exe")
    return Path(found) if found else None


def _is_wsl() -> bool:
    return Path("/mnt/c").is_dir() and "microsoft" in _release().lower()


def _release() -> str:
    try:
        return Path("/proc/version").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _windows_path(path: Path) -> str:
    resolved = path.resolve()
    parts = resolved.parts
    if len(parts) >= 3 and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        rest = "\\".join(parts[3:])
        return f"{drive}:\\{rest}"
    return str(resolved)


def _stage_dir() -> Path:
    """A temporary directory the Windows texconv can read/write (WSL-aware)."""
    if _is_wsl():
        base = Path("/mnt/c/Users/Public/Documents/.ace-utils-texconv")
        base.mkdir(parents=True, exist_ok=True)
        return base
    return Path(tempfile.mkdtemp(prefix="ace-texconv-"))


def make_dds(image_path: str | Path, format_name: str, *, mip_count: int = 0) -> bytes:
    """Compress an image to a DDS with texconv and return the DDS bytes.

    ``mip_count=0`` lets texconv generate the full mip chain.
    """
    texconv = find_texconv()
    if texconv is None:
        raise RuntimeError("texconv not found (set TEXCONV_PATH)")

    work = _stage_dir()
    suffix = Path(image_path).suffix or ".png"
    staged = work / f"src{suffix}"
    shutil.copyfile(image_path, staged)
    output = work / "src.dds"
    if output.exists():
        output.unlink()

    args = [str(texconv.resolve()), "-f", format_name, "-y", "-o", _windows_path(work)]
    args += ["-m", str(mip_count)] if mip_count else []
    args.append(_windows_path(staged))
    subprocess.run(args, check=True, capture_output=True)
    if not output.is_file():
        raise RuntimeError(f"texconv did not produce {output}")
    return output.read_bytes()
