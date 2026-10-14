"""Tests for DDS parsing and ACE texture conversion."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any

import pytest
from src.ace_formats import new_message
from src.convert.texture import (
    TILE_BYTES,
    convert_dds,
    convert_texture_file,
    encode_bc1,
    encode_bc3,
    encode_bc4,
    is_srgb_name,
    solid_texture,
    texture_from_image_file,
    texture_from_png,
    tile_geometry,
    tile_image,
)
from src.parser.dds import read_dds

_SKINS = Path("test_data/ac1/ek_tsubaki_line/skins/cm_skins/Winter")
_REF_MIPS = Path("test_data/ace_ref/content/tracks/ek_tsubak_hv/textures/da_light.texturemips")

_BLOCK_BYTES = {"DXT1": 8, "DXT5": 16}


def _surface(width: int, height: int, block_bytes: int) -> int:
    return max(1, (width + 3) // 4) * max(1, (height + 3) // 4) * block_bytes


def _make_dds(width: int, height: int, mip_count: int, fourcc: bytes) -> bytes:
    block_bytes = _BLOCK_BYTES[fourcc.decode()]
    header = bytearray(128)
    header[0:4] = b"DDS "
    struct.pack_into("<I", header, 4, 124)
    struct.pack_into("<I", header, 12, height)
    struct.pack_into("<I", header, 16, width)
    struct.pack_into("<I", header, 28, mip_count)
    struct.pack_into("<I", header, 76, 32)
    struct.pack_into("<I", header, 80, 0x4)
    header[84:88] = fourcc

    body = bytearray()
    w, h = width, height
    for _ in range(mip_count):
        size = _surface(w, h, block_bytes)
        body += bytes((i % 251) for i in range(size))
        w = max(1, w // 2)
        h = max(1, h // 2)
    return bytes(header) + bytes(body)


def _write(tmp_path: Path, data: bytes, name: str = "test.dds") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_read_dds_fields(tmp_path: Path) -> None:
    image = read_dds(_write(tmp_path, _make_dds(16, 16, 5, b"DXT1")))
    assert image.width == 16
    assert image.height == 16
    assert image.mip_count == 5
    assert image.format == "BC1"
    assert image.block_bytes == 8
    assert len(image.mips) == 5


def test_read_dds_rejects_bad_magic(tmp_path: Path) -> None:
    path = _write(tmp_path, b"NOPE" + b"\x00" * 200)
    with pytest.raises(Exception, match="not a DDS"):
        read_dds(path)


def test_read_dds_rejects_unknown_format(tmp_path: Path) -> None:
    data = bytearray(_make_dds(4, 4, 1, b"DXT1"))
    data[84:88] = b"XXXX"
    with pytest.raises(Exception, match="unsupported DDS pixel format"):
        read_dds(_write(tmp_path, bytes(data)))


def test_tile_geometry() -> None:
    assert tile_geometry(8) == (512, 256)
    assert tile_geometry(16) == (256, 256)


def test_tile_image_places_blocks_at_tile_pitch() -> None:
    # 16x16 BC1 -> 4x4 blocks, tile is 128x64 blocks wide
    mip = bytes(range(0, 256))[: 16 * 8]
    tiled, tiles = tile_image(mip, 16, 16, 8)
    assert tiles == 1
    assert len(tiled) == TILE_BYTES
    # block (0,0) at 0, block (0,1) at 8, row 1 starts at 128 blocks * 8 = 1024
    assert tiled[0:8] == mip[0:8]
    assert tiled[8:16] == mip[8:16]
    assert tiled[1024:1032] == mip[4 * 8 : 5 * 8]
    assert tiled[3104:3112] == b"\x00" * 8


def test_tile_image_16byte_blocks() -> None:
    mip = bytes((i % 200) for i in range(_surface(256, 256, 16)))
    tiled, tiles = tile_image(mip, 256, 256, 16)
    assert tiles == 1
    assert len(tiled) == TILE_BYTES
    assert tiled == mip


def test_is_srgb_name() -> None:
    assert is_srgb_name("road.dds")
    assert not is_srgb_name("road_NORM.dds")
    assert not is_srgb_name("flat_spec.dds")


def _decode(data: bytes) -> Any:
    msg = new_message(".texture")
    msg.ParseFromString(data)
    return msg


def test_convert_dds_metadata(tmp_path: Path) -> None:
    image = read_dds(_write(tmp_path, _make_dds(16, 16, 5, b"DXT1"), "da_light.dds"))
    texture = convert_dds(image, srgb=True)
    meta = _decode(texture.metadata.SerializeToString())
    assert meta.width == 16
    assert meta.height == 16
    assert meta.mipLevels == 5
    assert meta.format == 13  # BC1UNormSrgb
    assert meta.isTiled is True
    assert len(texture.mip_data) % TILE_BYTES == 0


def test_convert_dds_bc3_linear(tmp_path: Path) -> None:
    image = read_dds(_write(tmp_path, _make_dds(256, 256, 1, b"DXT5")))
    texture = convert_dds(image, srgb=False)
    meta = _decode(texture.metadata.SerializeToString())
    assert meta.format == 16  # BC3UNorm
    assert meta.isTiled is True
    assert len(texture.mip_data) == 65536


@pytest.mark.skipif(not (_SKINS / "o_road_a.dds").is_file(), reason="skin textures missing")
def test_convert_real_skin_texture() -> None:
    texture = convert_texture_file(_SKINS / "o_road_a.dds")
    meta = _decode(texture.metadata.SerializeToString())
    assert meta.width == 256
    assert meta.height == 256
    assert meta.format == 17  # BC3UNormSrgb (colour name)
    assert len(texture.mip_data) > 0


@pytest.mark.skipif(not _REF_MIPS.is_file(), reason="reference mips missing")
def test_reference_tile_geometry_matches() -> None:
    # da_light: 16x16 BC1, 5 mips, one 64 KiB tile each.
    assert _REF_MIPS.stat().st_size == 5 * TILE_BYTES
    mip = bytes((i % 251) for i in range(_surface(16, 16, 8)))
    tiled, _ = tile_image(mip, 16, 16, 8)
    assert len(tiled) == TILE_BYTES


def test_encode_bc1_solid_red() -> None:
    rgba = bytes((255, 0, 0, 255)) * (4 * 4)
    block = encode_bc1(rgba, 4, 4)
    assert len(block) == 8
    c0 = struct.unpack("<H", block[0:2])[0]
    # c0 is the max corner; red packs to 0xF800
    assert (c0 >> 11) & 0x1F == 0x1F


def test_encode_bc1_size() -> None:
    rgba = bytes((10, 20, 30, 255)) * (8 * 8)
    assert len(encode_bc1(rgba, 8, 8)) == 4 * 8  # 4 blocks, 8 bytes each


def test_solid_texture() -> None:
    texture = solid_texture(size=4, rgba=(255, 0, 0, 255))
    meta = _decode(texture.metadata.SerializeToString())
    assert meta.width == 4
    assert meta.mipLevels == 3
    assert meta.format == 13  # BC1UNormSrgb
    assert meta.isTiled is True
    assert len(texture.mip_data) % TILE_BYTES == 0


def test_texture_from_png(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "preview.png"
    Image.new("RGB", (64, 32), (10, 120, 200)).save(path)
    texture = texture_from_png(path, max_size=32)
    meta = _decode(texture.metadata.SerializeToString())
    assert (meta.width, meta.height) == (32, 16)
    assert meta.mipLevels >= 5
    assert len(texture.mip_data) > 0


def test_encode_bc4_endpoints() -> None:
    block = encode_bc4([200 - i for i in range(16)])
    assert len(block) == 8
    assert block[0] == 200
    assert block[1] == 185


def test_encode_bc3_size() -> None:
    rgba = bytes((10, 20, 30, 128)) * (4 * 4)
    assert len(encode_bc3(rgba, 4, 4)) == 16


def test_texture_from_png_with_alpha_format(tmp_path: Path) -> None:
    from PIL import Image

    path = tmp_path / "map.png"
    Image.new("RGBA", (16, 16), (255, 255, 255, 0)).save(path)
    texture = texture_from_image_file(path, max_size=16)
    meta = _decode(texture.metadata.SerializeToString())
    # BC7 (with texconv) or BC3 fallback — both carry alpha in ACE
    assert meta.format in (17, 34)
    assert meta.isTiled is True
