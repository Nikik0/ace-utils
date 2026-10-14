"""Tests for the .kspkg reader/writer."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.packer.kspkg import (
    XOR_KEY,
    KspkgError,
    KspkgFlags,
    KspkgReader,
    KspkgWriter,
    fnv1a64,
    hash_path,
    xor_bytes,
)


def test_fnv1a64_basis() -> None:
    assert fnv1a64(b"") == 0xCBF29CE484222325


def test_hash_path_matches_reference_package() -> None:
    # Value observed in the EvoForge-produced ek_tsubak_hv.kspkg file table.
    path = r"content\tracks\ek_tsubak_hv\static_mesh\ROAD_GRAVEL_EDGE_ROAD_0_-1.mesh"
    assert hash_path(path) == 0x001584F46E025BB8


def test_hash_path_is_case_insensitive_and_slash_normalizing() -> None:
    assert hash_path(r"Content/Tracks/X/Y.mesh") == hash_path(r"content\tracks\x\y.MESH")


def test_xor_roundtrip() -> None:
    data = bytes(range(256)) * 3 + b"abcde"
    assert xor_bytes(xor_bytes(data)) == data


def test_xor_uses_key_bytes() -> None:
    assert xor_bytes(b"\x00" * 8) == XOR_KEY.to_bytes(8, "little")


def test_writer_reader_roundtrip(tmp_path: Path) -> None:
    out = tmp_path / "test.kspkg"
    writer = KspkgWriter()
    writer.add_file(r"content\tracks\test\test.track", b"track-data")
    writer.add_file(r"content\tracks\test\ui\previews.json", b"{}")
    writer.write(out)

    with KspkgReader(out) as reader:
        assert len(reader.entries) == 2
        by_path = {e.path: e for e in reader.entries}
        assert reader.read_file(by_path[r"content\tracks\test\test.track"]) == b"track-data"
        assert reader.read_file(by_path[r"content\tracks\test\ui\previews.json"]) == b"{}"
        assert out.stat().st_size >= 0x400_0000 + len(b"track-data") + len(b"{}")


def test_encrypted_and_directory_entries(tmp_path: Path) -> None:
    out = tmp_path / "enc.kspkg"
    writer = KspkgWriter()
    writer.add_directory(r"content\tracks\enc")
    writer.add_file(r"content\tracks\enc\secret.bin", b"secret-payload", encrypted=True)
    writer.write(out)

    with KspkgReader(out) as reader:
        entries = {e.path: e for e in reader.entries}
        directory = entries[r"content\tracks\enc"]
        assert directory.is_directory
        assert reader.read_file(directory) == b""
        secret = entries[r"content\tracks\enc\secret.bin"]
        assert secret.is_encrypted
        assert secret.flags & KspkgFlags.ENCRYPTED
        assert reader.read_file(secret) == b"secret-payload"


def test_table_ordered_by_hash(tmp_path: Path) -> None:
    out = tmp_path / "order.kspkg"
    writer = KspkgWriter()
    for i in range(50):
        writer.add_file(rf"content\tracks\order\f{i:02d}.mesh", bytes([i]))
    writer.write(out)
    with KspkgReader(out) as reader:
        hashes = [e.path_hash for e in reader.entries]
    assert hashes == sorted(hashes)


def test_by_path_lookup(tmp_path: Path) -> None:
    out = tmp_path / "lookup.kspkg"
    writer = KspkgWriter()
    writer.add_file(r"content\tracks\lk\a.track", b"a")
    writer.write(out)
    with KspkgReader(out) as reader:
        assert reader.by_path(r"Content/Tracks/LK/A.TRACK") is not None
        assert reader.by_path(r"content\tracks\lk\missing.track") is None


def test_missing_file_raises() -> None:
    with pytest.raises(KspkgError, match="not found"):
        KspkgReader("/nonexistent/path/to.kspkg")


_REFERENCE_KSPKG = Path(
    "/mnt/c/Users/Nikiko/Saved Games/ACE/mods/ek_tsubak_hv.kspkg"
)


@pytest.mark.skipif(not _REFERENCE_KSPKG.is_file(), reason="reference mod not present")
def test_read_reference_package() -> None:
    with KspkgReader(_REFERENCE_KSPKG) as reader:
        assert len(reader.entries) == 3553
        track = reader.by_path(r"content\tracks\ek_tsubak_hv\ek_tsubak_hv.track")
        assert track is not None
        assert len(reader.read_file(track)) == 23
