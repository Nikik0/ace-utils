"""Minimal DirectDraw Surface (``.dds``) reader for AC1 textures.

Only what ACE conversion needs: header fields and the per-mip block data of
BC-compressed images. AC1 track textures are typically DXT1/DXT5 (BC1/BC3).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from src.parser.errors import TrackParseError

DDS_MAGIC = b"DDS "
_HEADER = struct.Struct("<7I")
_FOURCC_OFFSET = 84
_MIPCOUNT_OFFSET = 28
_HEADER_SIZE = 128

# fourCC -> (name, bytes per 4x4 block)
_BLOCK_FORMATS: dict[bytes, tuple[str, int]] = {
    b"DXT1": ("BC1", 8),
    b"DXT3": ("BC2", 16),
    b"DXT5": ("BC3", 16),
    b"ATI1": ("BC4", 8),
    b"BC4U": ("BC4", 8),
    b"ATI2": ("BC5", 16),
    b"BC5U": ("BC5", 16),
}

# DXGI_FORMAT ids (DDS DX10 header) -> (name, bytes per 4x4 block)
_DXGI: dict[int, tuple[str, int]] = {
    70: ("BC1", 8),
    71: ("BC1", 8),
    74: ("BC2", 16),
    77: ("BC3", 16),
    80: ("BC4", 8),
    83: ("BC5", 16),
    95: ("BC6H", 16),
    98: ("BC7", 16),
    99: ("BC7", 16),
}


@dataclass(frozen=True, slots=True)
class DdsImage:
    width: int
    height: int
    mip_count: int
    format: str
    block_bytes: int
    mips: tuple[bytes, ...]
    path: Path


def _surface_size(width: int, height: int, block_bytes: int) -> int:
    return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * block_bytes


def read_dds(path: str | Path) -> DdsImage:
    """Parse a BC-compressed ``.dds`` file into its mip chain."""
    source = Path(path)
    data = source.read_bytes()
    if data[:4] != DDS_MAGIC:
        raise TrackParseError(source, "not a DDS file (bad magic)")
    if len(data) < _HEADER_SIZE:
        raise TrackParseError(source, "DDS file shorter than header")

    height, width = struct.unpack("<2I", data[12:20])
    fourcc = data[_FOURCC_OFFSET : _FOURCC_OFFSET + 4]
    mip_count = struct.unpack("<I", data[_MIPCOUNT_OFFSET : _MIPCOUNT_OFFSET + 4])[0]

    if fourcc == b"DX10":
        dxgi = struct.unpack("<I", data[_HEADER_SIZE : _HEADER_SIZE + 4])[0]
        if dxgi not in _DXGI:
            raise TrackParseError(source, f"unsupported DXGI format {dxgi}")
        name, block_bytes = _DXGI[dxgi]
        offset = _HEADER_SIZE + 20
    elif fourcc in _BLOCK_FORMATS:
        name, block_bytes = _BLOCK_FORMATS[fourcc]
        offset = _HEADER_SIZE
    else:
        raise TrackParseError(source, f"unsupported DDS pixel format {fourcc!r}")

    if mip_count <= 0:
        mip_count = 1

    mips: list[bytes] = []
    mip_width, mip_height = width, height
    for _ in range(mip_count):
        size = _surface_size(mip_width, mip_height, block_bytes)
        if offset + size > len(data):
            raise TrackParseError(source, f"truncated mip data at offset {offset}")
        mips.append(data[offset : offset + size])
        offset += size
        mip_width = max(1, mip_width // 2)
        mip_height = max(1, mip_height // 2)

    return DdsImage(
        width=width,
        height=height,
        mip_count=mip_count,
        format=name,
        block_bytes=block_bytes,
        mips=tuple(mips),
        path=source,
    )
