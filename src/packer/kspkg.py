""".kspkg (KSPackage) reader and writer for Assetto Corsa EVO.

Format recovered from refs/ACEvo.Package (C#) and refs/ace-kspkg (Python):

- A KSPackage is a sequence of file blobs followed by a fixed-size file table.
- The file table sits at the end of the file and is 64 MiB on game >= 0.7
  (32 MiB on older builds); it holds 0x100-byte entries up to the first entry
  whose path hash is zero.
- The whole table XORed with the static key 0x9F9721A97D1135C1 (8-byte
  little-endian words, then trailing bytes). Individual file blobs are XORed
  with the same key only when the Encrypted flag is set.
- Entry paths are case-insensitive: the lookup hash is FNV1A64 over the
  UTF-16LE bytes of the lowercased path with '/' replaced by '\\'.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import IntFlag
from pathlib import Path

from loguru import logger

XOR_KEY = 0x9F9721A97D1135C1
TABLE_SIZE_64MB = 0x400_0000
TABLE_SIZE_32MB = 0x200_0000
TABLE_SIZES: tuple[int, ...] = (TABLE_SIZE_64MB, TABLE_SIZE_32MB)
ENTRY_SIZE = 0x100
# File data offsets are aligned so the engine's DirectStorage reader can read
# each asset without an unaligned copy.
DATA_ALIGNMENT = 0x1000
NAME_SIZE = 0xE0

_FNV1A64_OFFSET = 0xCBF29CE484222325
_FNV1A64_PRIME = 0x100000001B3
_MASK64 = 0xFFFF_FFFF_FFFF_FFFF


class KspkgError(Exception):
    """Raised when a .kspkg package cannot be read or written."""


class KspkgFlags(IntFlag):
    """Entry flags stored in the file table."""

    DIRECTORY = 1 << 0
    ENCRYPTED = 1 << 8


def fnv1a64(data: bytes) -> int:
    h = _FNV1A64_OFFSET
    for byte in data:
        h = ((h ^ byte) * _FNV1A64_PRIME) & _MASK64
    return h


def hash_path(path: str) -> int:
    """FNV1A64 of the lowercased, backslash-separated UTF-16LE path bytes."""
    normalized = path.lower().replace("/", "\\")
    return fnv1a64(normalized.encode("utf-16-le"))


def xor_bytes(data: bytes | bytearray, key: int = XOR_KEY) -> bytes:
    """Apply the KSPackage XOR cipher (8-byte LE words, then trailing bytes)."""
    buf = bytearray(data)
    length = len(buf)
    words = length // 8
    if words:
        words_view = memoryview(buf)[: words * 8].cast("Q")
        for i in range(words):
            words_view[i] ^= key
    for i in range(words * 8, length):
        buf[i] ^= (key >> (8 * (i - words * 8))) & 0xFF
    return bytes(buf)


@dataclass(slots=True)
class KspkgEntry:
    """One entry from the KSPackage file table."""

    path: str
    size: int
    offset: int
    flags: KspkgFlags = field(default_factory=lambda: KspkgFlags(0))
    path_hash: int = 0

    @property
    def is_directory(self) -> bool:
        return bool(self.flags & KspkgFlags.DIRECTORY)

    @property
    def is_encrypted(self) -> bool:
        return bool(self.flags & KspkgFlags.ENCRYPTED)


@dataclass(slots=True)
class KspkgFile:
    path: str
    data: bytes
    flags: KspkgFlags = field(default_factory=lambda: KspkgFlags(0))


def _parse_entry(entry: bytes) -> KspkgEntry:
    name_length = struct.unpack_from("<h", entry, 0xE6)[0]
    path_hash = struct.unpack_from("<Q", entry, 0xE8)[0]
    size = struct.unpack_from("<q", entry, 0xF0)[0]
    offset = struct.unpack_from("<q", entry, 0xF8)[0]
    flags = KspkgFlags(struct.unpack_from("<H", entry, 0xE4)[0])
    path = entry[:name_length].decode("ascii", "replace")
    return KspkgEntry(path=path, size=size, offset=offset, flags=flags, path_hash=path_hash)


def _pack_entry(entry: KspkgEntry) -> bytes:
    raw_path = entry.path.encode("ascii")
    if len(raw_path) > NAME_SIZE:
        raise KspkgError(f"path too long for table ({len(raw_path)} > {NAME_SIZE}): {entry.path}")
    buf = bytearray(ENTRY_SIZE)
    buf[0 : len(raw_path)] = raw_path
    struct.pack_into("<i", buf, 0xE0, 0)
    struct.pack_into("<H", buf, 0xE4, int(entry.flags))
    struct.pack_into("<h", buf, 0xE6, len(raw_path))
    struct.pack_into("<Q", buf, 0xE8, entry.path_hash)
    struct.pack_into("<q", buf, 0xF0, entry.size)
    struct.pack_into("<q", buf, 0xF8, entry.offset)
    return bytes(buf)


class KspkgReader:
    """Read a KSPackage file table and its file blobs."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise KspkgError(f"kspkg not found: {self.path}")
        self._stream = self.path.open("rb")
        self.table_size = self._detect_table_size()
        self.entries: list[KspkgEntry] = self._read_table()

    def _detect_table_size(self) -> int:
        for table_size in TABLE_SIZES:
            if table_size > self.path.stat().st_size:
                continue
            self._stream.seek(-table_size, 2)
            first = xor_bytes(self._stream.read(ENTRY_SIZE))
            if struct.unpack_from("<Q", first, 0xE8)[0] == 0:
                continue
            name_length = struct.unpack_from("<h", first, 0xE6)[0]
            if not 0 < name_length <= NAME_SIZE:
                continue
            name = first[:name_length].decode("ascii", "replace")
            if name.lower().startswith("content"):
                return table_size
        raise KspkgError(f"no valid file table found in {self.path}")

    def _read_table(self) -> list[KspkgEntry]:
        self._stream.seek(-self.table_size, 2)
        table = xor_bytes(self._stream.read(self.table_size))
        entries: list[KspkgEntry] = []
        for offset in range(0, self.table_size, ENTRY_SIZE):
            entry = _parse_entry(table[offset : offset + ENTRY_SIZE])
            if entry.path_hash == 0:
                break
            entries.append(entry)
        logger.debug(
            "kspkg {}: {} entries, table 0x{:X}",
            self.path.name,
            len(entries),
            self.table_size,
        )
        return entries

    def read_file(self, entry: KspkgEntry) -> bytes:
        if entry.is_directory:
            return b""
        self._stream.seek(entry.offset, 0)
        data = self._stream.read(entry.size)
        if len(data) != entry.size:
            raise KspkgError(f"short read for {entry.path} in {self.path}")
        if entry.is_encrypted:
            data = xor_bytes(data)
        return data

    def by_path(self, path: str) -> KspkgEntry | None:
        target = hash_path(path)
        for entry in self.entries:
            if entry.path_hash == target:
                return entry
        return None

    def extract_all(self, out_dir: str | Path) -> int:
        out = Path(out_dir)
        count = 0
        for entry in self.entries:
            dest = out / entry.path.replace("\\", "/")
            if entry.is_directory:
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(self.read_file(entry))
            count += 1
        logger.info("extracted {} files from {} to {}", count, self.path.name, out)
        return count

    def close(self) -> None:
        self._stream.close()

    def __enter__(self) -> KspkgReader:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class KspkgWriter:
    """Build a KSPackage from files, then write it to disk."""

    def __init__(self, table_size: int = TABLE_SIZE_64MB) -> None:
        if table_size not in TABLE_SIZES:
            raise KspkgError(f"unsupported table size 0x{table_size:X}")
        self.table_size = table_size
        self._files: list[KspkgFile] = []

    def add_file(self, path: str, data: bytes, *, encrypted: bool = False) -> None:
        flags = KspkgFlags.ENCRYPTED if encrypted else KspkgFlags(0)
        self._files.append(KspkgFile(path=path.replace("/", "\\"), data=data, flags=flags))

    def add_directory(self, path: str) -> None:
        self._files.append(
            KspkgFile(path=path.replace("/", "\\"), data=b"", flags=KspkgFlags.DIRECTORY)
        )

    def add_tree(self, source: str | Path, arc_prefix: str = "") -> int:
        """Add every file under ``source`` using backslash paths.

        ``arc_prefix`` is prepended as an internal directory, e.g.
        ``content/tracks/mytrack``. Returns the number of files added.
        """
        root = Path(source)
        if not root.is_dir():
            raise KspkgError(f"source folder not found: {root}")
        prefix = arc_prefix.replace("/", "\\").strip("\\")
        count = 0
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix().replace("/", "\\")
            arcname = f"{prefix}\\{relative}" if prefix else relative
            self.add_file(arcname, path.read_bytes())
            count += 1
        return count

    def write(self, out_path: str | Path) -> Path:
        out = Path(out_path)
        max_entries = self.table_size // ENTRY_SIZE
        if len(self._files) > max_entries:
            raise KspkgError(f"too many entries for table ({len(self._files)} > {max_entries})")

        blob = bytearray()
        entries: list[KspkgEntry] = []
        for item in self._files:
            pad = (-len(blob)) % DATA_ALIGNMENT
            blob += b"\x00" * pad
            offset = len(blob)
            if not item.flags & KspkgFlags.DIRECTORY:
                payload = xor_bytes(item.data) if item.flags & KspkgFlags.ENCRYPTED else item.data
                blob += payload
                size = len(item.data)
            else:
                size = 0
            entries.append(
                KspkgEntry(
                    path=item.path,
                    size=size,
                    offset=offset,
                    flags=item.flags,
                    path_hash=hash_path(item.path),
                )
            )

        entries.sort(key=lambda e: e.path_hash)
        table = bytearray(self.table_size)
        for index, entry in enumerate(entries):
            table[index * ENTRY_SIZE : (index + 1) * ENTRY_SIZE] = _pack_entry(entry)
        table = bytearray(xor_bytes(bytes(table)))

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("wb") as fh:
            fh.write(blob)
            fh.write(table)
        logger.info("wrote {} entries ({} bytes) to {}", len(entries), out.stat().st_size, out)
        return out
