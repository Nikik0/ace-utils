"""Convert AC1 DDS textures into ACE ``.texture`` metadata + ``.texturemips``.

ACE stores mip data in 64 KiB tiles (DX12 tiled-resource style). Tiles are
always 65536 bytes: 512x256 texels for 8-byte blocks (BC1/BC4) and 256x256 for
16-byte blocks (BC2/BC3/BC5/BC6H/BC7). Within a subresource, tiles are stored
row-major; within a tile, blocks are stored row-major.

AC1 textures are already BC-compressed, so blocks are passed through unchanged
with the matching ``PixelFormat`` (no re-encoding). UI images (PNG) are encoded
to BC1.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from google.protobuf.message import Message
from loguru import logger

from src.ace_formats import new_message
from src.convert.material import dds_texture_name
from src.parser.dds import DdsImage, read_dds

TILE_BYTES = 65536

# When False, mip data is written linear and the engine builds the tiled
# resource itself (set isTiled=false). When True, data is pre-tiled — which is
# what ACE's own content (and the reference) uses.
TILED = True

# PixelFormat enum values (Renderer.proto).
_PIXEL_FORMATS: dict[str, tuple[int, int]] = {
    # name: (linear, srgb)
    "BC1": (12, 13),
    "BC2": (14, 15),
    "BC3": (16, 17),
    "BC4": (27, 27),
    "BC5": (35, 35),
    "BC6H": (26, 32),
    "BC7": (33, 34),
}

# TextureMetadata.PixelDataType, keyed by (format name, srgb).
_PIXEL_DATA_TYPES: dict[tuple[str, bool], int] = {
    ("BC1", True): 8,  # SrgbColor_BC1_NoAlpha
    ("BC1", False): 5,  # LinearColor_BC1_NoAlpha
    ("BC3", True): 7,  # SrgbColor_BC3_Alpha
    ("BC3", False): 6,  # LinearColor_BC3_Alpha
    ("BC4", True): 2,  # LinearGreyscale_BC4
    ("BC4", False): 2,
    ("BC5", True): 9,  # NormalMap_BC5
    ("BC5", False): 9,
    ("BC6H", True): 3,  # NormalMap_BC6H
    ("BC6H", False): 3,
    ("BC7", True): 0,  # SrgbColor_BC7_Alpha
    ("BC7", False): 1,  # LinearColor_BC7_Alpha
}


def pixel_data_type(format_name: str, *, srgb: bool) -> int:
    return _PIXEL_DATA_TYPES.get((format_name, srgb), 0)


@dataclass(frozen=True, slots=True)
class AceTexture:
    """An ACE texture asset: metadata message plus tiled mip bytes."""

    metadata: Message
    mip_data: bytes


def tile_geometry(block_bytes: int) -> tuple[int, int]:
    """Tile size in texels that yields a 64 KiB tile for the given block size."""
    if block_bytes == 16:
        return 256, 256
    return 512, 256


def tile_image(
    mip: bytes, width: int, height: int, block_bytes: int
) -> tuple[bytes, int]:
    """Arrange one mip's linear blocks into ACE 64 KiB tiles.

    Returns the tiled bytes and the number of tiles used.
    """
    tile_w, tile_h = tile_geometry(block_bytes)
    tile_blocks_x = tile_w // 4
    tile_blocks_y = tile_h // 4
    blocks_x = max(1, (width + 3) // 4)
    blocks_y = max(1, (height + 3) // 4)
    tiles_x = -(-blocks_x // tile_blocks_x)
    tiles_y = -(-blocks_y // tile_blocks_y)

    tile_size = tile_blocks_x * tile_blocks_y * block_bytes
    out = bytearray(tiles_x * tiles_y * tile_size)
    for by in range(blocks_y):
        for bx in range(blocks_x):
            src = (by * blocks_x + bx) * block_bytes
            tile = (by // tile_blocks_y) * tiles_x + (bx // tile_blocks_x)
            within = (by % tile_blocks_y) * tile_blocks_x + (bx % tile_blocks_x)
            dst = tile * tile_size + within * block_bytes
            out[dst : dst + block_bytes] = mip[src : src + block_bytes]
    return bytes(out), tiles_x * tiles_y


def pixel_format_name(format_name: str, *, srgb: bool) -> int:
    """ACE ``PixelFormat`` value for a DDS block format."""
    if format_name not in _PIXEL_FORMATS:
        raise ValueError(f"unsupported texture format: {format_name}")
    linear, srgb_value = _PIXEL_FORMATS[format_name]
    return srgb_value if srgb else linear


def is_srgb_name(name: str) -> bool:
    """Colour textures are sRGB; normal/spec/roughness maps are linear."""
    lowered = dds_texture_name(name).lower()
    return not any(token in lowered for token in ("_norm", "_nm", "_spec", "_rough", "_ao"))


def convert_dds(image: DdsImage, *, srgb: bool | None = None) -> AceTexture:
    """Convert a parsed DDS image into an ACE texture asset."""
    if srgb is None:
        srgb = is_srgb_name(image.path.name)
    format_value = pixel_format_name(image.format, srgb=srgb)

    if TILED:
        tiled: list[bytes] = []
        offsets: list[int] = []
        counts: list[int] = []
        tile_counter = 0
        width, height = image.width, image.height
        for mip in image.mips:
            offsets.append(tile_counter)
            data, tiles = tile_image(mip, width, height, image.block_bytes)
            tiled.append(data)
            counts.append(tiles)
            tile_counter += tiles
            width = max(1, width // 2)
            height = max(1, height // 2)
        mip_data = b"".join(tiled)
    else:
        offsets, counts = [], []
        mip_data = b"".join(image.mips)

    tile_w, tile_h = tile_geometry(image.block_bytes)
    metadata = _build_metadata(
        width=image.width,
        height=image.height,
        mip_count=image.mip_count,
        format_value=format_value,
        data_type=pixel_data_type(image.format, srgb=srgb),
        tiled=TILED,
        tile_w=tile_w,
        tile_h=tile_h,
        offsets=offsets,
        counts=counts,
        source_name=image.path.name,
    )
    return AceTexture(metadata=metadata, mip_data=mip_data)


def _build_metadata(
    *,
    width: int,
    height: int,
    mip_count: int,
    format_value: int,
    data_type: int,
    tiled: bool,
    tile_w: int,
    tile_h: int,
    offsets: list[int],
    counts: list[int],
    source_name: str,
) -> Message:
    msg = cast(Any, new_message(".texture"))
    msg.width = width
    msg.height = height
    msg.mipLevels = mip_count
    msg.format = format_value
    msg.depthOrArraySize = 1

    msg.sourceImages.importPaths.append(source_name)
    msg.sourceImages.width = width
    msg.sourceImages.height = height
    msg.sourceImages.mipLevels = mip_count
    msg.sourceImages.arraySize = 1
    msg.sourceImages.format = 5  # B8G8R8A8UNorm (import source)
    msg.isTiled = tiled

    msg.conversionSettings.compress = True
    msg.conversionSettings.pixelDataType = data_type
    msg.conversionSettings.alphaToCoverageAlphaReference = 0.5

    if tiled:
        msg.tilingInfo.tileWidthInTexels = tile_w
        msg.tilingInfo.tileHeightInTexels = tile_h
        msg.tilingInfo.tileDepthInTexels = 1
        msg.tilingInfo.tileOffsetForSubresource.extend(offsets)
        msg.tilingInfo.tileCountForSubresource.extend(counts)
        # EvoForge emits one row pitch per *tile* (not per mip), all 1024 B.
        msg.tilingInfo.rowPitchForTile.extend([TILE_BYTES // (tile_h // 4)] * sum(counts))
    return cast(Message, msg)


def convert_texture_file(path: str | Path, *, srgb: bool | None = None) -> AceTexture:
    """Read an AC1 DDS file and convert it into an ACE texture asset."""
    return convert_dds(read_dds(path), srgb=srgb)


def null_texture() -> AceTexture:
    """A 4x4 opaque-black BC1 texture, used as a fallback for missing maps."""
    from src.parser.dds import DdsImage

    image = DdsImage(
        width=4,
        height=4,
        mip_count=1,
        format="BC1",
        block_bytes=8,
        mips=(b"\x00" * 16,),
        path=Path("NULL.dds"),
    )
    return convert_dds(image, srgb=True)


# -- BC1 encoding for generated (PNG) textures -------------------------------


def _pack565(r: int, g: int, b: int) -> int:
    return ((r >> 3) << 11) | ((g >> 2) << 5) | (b >> 3)


def _unpack565(value: int) -> tuple[int, int, int]:
    r = ((value >> 11) & 0x1F) * 255 // 31
    g = ((value >> 5) & 0x3F) * 255 // 63
    b = (value & 0x1F) * 255 // 31
    return (r, g, b)


def _nearest(pixel: tuple[int, int, int], palette: list[tuple[int, int, int]]) -> int:
    best, best_distance = 0, 1 << 30
    for slot, colour in enumerate(palette):
        distance = (
            (pixel[0] - colour[0]) ** 2 + (pixel[1] - colour[1]) ** 2 + (pixel[2] - colour[2]) ** 2
        )
        if distance < best_distance:
            best, best_distance = slot, distance
    return best


def encode_bc1(rgba: bytes, width: int, height: int, *, allow_alpha: bool = False) -> bytes:
    """Encode an RGBA byte buffer as BC1 (DXT1).

    Uses 3-colour + 1-bit alpha when the block has transparent pixels (so alpha
    cut-out foliage works); otherwise 4-colour opaque.
    """
    out = bytearray()
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            block: list[tuple[int, int, int]] = []
            alphas: list[int] = []
            for yy in range(4):
                y = min(by + yy, height - 1)
                for xx in range(4):
                    x = min(bx + xx, width - 1)
                    offset = (y * width + x) * 4
                    block.append((rgba[offset], rgba[offset + 1], rgba[offset + 2]))
                    alphas.append(rgba[offset + 3])

            min_c = _pack565(
                min(p[0] for p in block), min(p[1] for p in block), min(p[2] for p in block)
            )
            max_c = _pack565(
                max(p[0] for p in block), max(p[1] for p in block), max(p[2] for p in block)
            )

            if allow_alpha and any(alpha < 128 for alpha in alphas):
                c0, c1 = min_c, max_c  # 3-colour mode needs c0 <= c1
                if c0 == c1:
                    c1 = min(0xFFFF, c0 + 1)
                p0, p1 = _unpack565(c0), _unpack565(c1)
                mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2, (p0[2] + p1[2]) // 2)
                palette = [p0, p1, mid]
                bits = 0
                for index, pixel in enumerate(block):
                    slot = 3 if alphas[index] < 128 else _nearest(pixel, palette)
                    bits |= slot << (index * 2)
            else:
                c0, c1 = max_c, min_c
                if c0 < c1:
                    c0, c1 = c1, c0
                elif c0 == c1:
                    if c0 == 0:
                        c0 = 1
                    else:
                        c1 = c0 - 1
                p0, p1 = _unpack565(c0), _unpack565(c1)
                third = (
                    (2 * p0[0] + p1[0]) // 3,
                    (2 * p0[1] + p1[1]) // 3,
                    (2 * p0[2] + p1[2]) // 3,
                )
                fourth = (
                    (p0[0] + 2 * p1[0]) // 3,
                    (p0[1] + 2 * p1[1]) // 3,
                    (p0[2] + 2 * p1[2]) // 3,
                )
                palette = [p0, p1, third, fourth]
                bits = 0
                for index, pixel in enumerate(block):
                    bits |= _nearest(pixel, palette) << (index * 2)
            out += struct.pack("<HHI", c0, c1, bits)
    return bytes(out)


def _ace_from_rgba_mips(
    mips: list[tuple[bytes, int, int]], *, srgb: bool, source_name: str
) -> AceTexture:
    # ACE only ships BC1 (no-alpha) / BC4 / BC7 textures; BC3 is not supported,
    # so re-encode everything to BC1 (1-bit alpha where needed).
    has_alpha = any(rgba[index] != 255 for rgba, _, _ in mips for index in range(3, len(rgba), 4))
    # ACE's BC1 is NoAlpha; alpha (foliage cut-outs) needs BC3, which the engine
    # supports. Everything else uses BC1.
    if has_alpha:
        format_name, block_bytes, encode = "BC3", 16, encode_bc3
    else:
        format_name, block_bytes, encode = "BC1", 8, encode_bc1
    format_value = pixel_format_name(format_name, srgb=srgb)

    if TILED:
        tiles_data: list[bytes] = []
        offsets: list[int] = []
        counts: list[int] = []
        counter = 0
        for rgba, width, height in mips:
            data, tiles = tile_image(encode(rgba, width, height), width, height, block_bytes)
            offsets.append(counter)
            counts.append(tiles)
            tiles_data.append(data)
            counter += tiles
        mip_data = b"".join(tiles_data)
    else:
        offsets, counts = [], []
        mip_data = b"".join(encode(rgba, width, height) for rgba, width, height in mips)
    tile_w, tile_h = tile_geometry(block_bytes)
    metadata = _build_metadata(
        width=mips[0][1],
        height=mips[0][2],
        mip_count=len(mips),
        format_value=format_value,
        data_type=pixel_data_type(format_name, srgb=srgb),
        tiled=TILED,
        tile_w=tile_w,
        tile_h=tile_h,
        offsets=offsets,
        counts=counts,
        source_name=source_name,
    )
    return AceTexture(metadata=metadata, mip_data=mip_data)


def encode_bc4(values: list[int]) -> bytes:
    """Encode 16 alpha values as a BC4 (8-value) block."""
    a0, a1 = max(values), min(values)
    if a0 == a1:
        a1 = max(0, a0 - 1)
    palette = [a0, a1] + [((6 - i) * a0 + i * a1) // 7 for i in range(1, 7)]
    bits = 0
    for index, value in enumerate(values):
        best, best_distance = 0, 1 << 30
        for slot, level in enumerate(palette):
            if abs(value - level) < best_distance:
                best, best_distance = slot, abs(value - level)
        bits |= best << (index * 3)
    return bytes((a0, a1)) + bits.to_bytes(6, "little")


def encode_bc3(rgba: bytes, width: int, height: int) -> bytes:
    """Encode RGBA as BC3 (DXT5): BC4 alpha block followed by a BC1 colour block."""
    out = bytearray()
    bc1 = encode_bc1(rgba, width, height)
    block = 0
    for by in range(0, height, 4):
        for bx in range(0, width, 4):
            alpha: list[int] = []
            for yy in range(4):
                y = min(by + yy, height - 1)
                for xx in range(4):
                    x = min(bx + xx, width - 1)
                    alpha.append(rgba[(y * width + x) * 4 + 3])
            out += encode_bc4(alpha)
            out += bc1[block * 8 : block * 8 + 8]
            block += 1
    return bytes(out)


def _largest_power_of_two(value: int) -> int:
    power = 1
    while power * 2 <= value:
        power *= 2
    return max(1, power)


def texture_from_image_file(
    path: str | Path, *, max_size: int = 1024, srgb: bool = True
) -> AceTexture:
    """Encode any Pillow-readable image as an ACE texture.

    Dimensions are forced to powers of two (the engine's tiled resources require
    it). When ``texconv`` is available the image is compressed to ACE's native
    **BC7**; otherwise it falls back to a Pillow re-encode as BC1/BC3.
    """
    from PIL import Image

    source = Path(path)
    image = Image.open(source).convert("RGBA")
    image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    image = image.resize(
        (_largest_power_of_two(image.width), _largest_power_of_two(image.height)),
        Image.Resampling.LANCZOS,
    )

    dds = _texconv_bc7(image, srgb=srgb)
    if dds is not None:
        return dds

    mips: list[tuple[bytes, int, int]] = []
    current = image
    while True:
        mips.append((current.tobytes(), current.width, current.height))
        if current.width == 1 and current.height == 1:
            break
        current = current.resize(
            (max(1, current.width // 2), max(1, current.height // 2)),
            Image.Resampling.BILINEAR,
        )
    return _ace_from_rgba_mips(mips, srgb=srgb, source_name=source.name)


def _texconv_bc7(image: Any, *, srgb: bool) -> AceTexture | None:
    """Compress a PIL image to BC7 with texconv; None if unavailable/failed."""
    from src.convert import texconv

    if texconv.find_texconv() is None:
        return None
    work = texconv._stage_dir()
    png = work / "img.png"
    dds_path = work / "img.dds"
    try:
        image.save(png)
        data = texconv.make_dds(png, texconv.BC7_SRGB if srgb else texconv.BC7_LINEAR)
        dds_path.write_bytes(data)
        return convert_dds(read_dds(dds_path), srgb=srgb)
    except Exception as exc:
        logger.warning("texconv failed for {}: {}", image, exc)
        return None


def solid_texture(
    *, size: int = 4, rgba: tuple[int, int, int, int] = (0, 0, 0, 255), srgb: bool = True
) -> AceTexture:
    """A solid-colour BC1 texture with a full mip chain (no image library needed)."""
    mips: list[tuple[bytes, int, int]] = []
    width = height = size
    while True:
        mips.append((bytes(rgba) * (width * height), width, height))
        if width == 1 and height == 1:
            break
        width, height = max(1, width // 2), max(1, height // 2)
    return _ace_from_rgba_mips(mips, srgb=srgb, source_name="solid")



texture_from_png = texture_from_image_file
