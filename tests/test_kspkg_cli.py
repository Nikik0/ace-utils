"""Tests for kspkg tree packing, install helpers and the CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from src.packer.__main__ import main
from src.packer.install import default_mods_dir, install_directory, install_kspkg
from src.packer.kspkg import KspkgReader, KspkgWriter


def _make_tree(root: Path) -> None:
    (root / "static_mesh").mkdir(parents=True)
    (root / "static_mesh" / "a.mesh").write_bytes(b"mesh-a")
    (root / "track.track").write_bytes(b"track")
    (root / "ui").mkdir()
    (root / "ui" / "map.svg").write_text("<svg/>")


def test_add_tree_and_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    _make_tree(source)
    out = tmp_path / "t.kspkg"

    writer = KspkgWriter()
    count = writer.add_tree(source, "content/tracks/mytrack")
    assert count == 3
    writer.write(out)

    with KspkgReader(out) as reader:
        assert reader.by_path(r"content\tracks\mytrack\static_mesh\a.mesh") is not None
        assert reader.by_path(r"content\tracks\mytrack\track.track") is not None
        entry = reader.by_path(r"content\tracks\mytrack\ui\map.svg")
        assert entry is not None
        assert reader.read_file(entry) == b"<svg/>"


def test_install_kspkg(tmp_path: Path) -> None:
    source = tmp_path / "mod.kspkg"
    source.write_bytes(b"payload")
    mods = tmp_path / "mods"
    dest = install_kspkg(source, mods)
    assert dest == mods / "mod.kspkg"
    assert dest.read_bytes() == b"payload"


def test_install_kspkg_rejects_non_kspkg(tmp_path: Path) -> None:
    source = tmp_path / "mod.zip"
    source.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"not a .kspkg"):
        install_kspkg(source, tmp_path / "mods")


def test_install_directory(tmp_path: Path) -> None:
    source = tmp_path / "mytrack"
    _make_tree(source)
    dest = install_directory(source, tmp_path / "mods")
    assert (dest / "track.track").is_file()


def test_cli_pack_list_unpack(tmp_path: Path) -> None:
    source = tmp_path / "tree"
    _make_tree(source)
    out = tmp_path / "t.kspkg"

    assert main(["pack", str(source), "-o", str(out), "--prefix", "content/tracks/mytrack"]) == 0
    assert out.is_file()

    assert main(["list", str(out)]) == 0

    unpacked = tmp_path / "unpacked"
    assert main(["unpack", str(out), "-o", str(unpacked)]) == 0
    assert (unpacked / "content" / "tracks" / "mytrack" / "track.track").read_bytes() == b"track"


def test_cli_install_uses_mods_override(tmp_path: Path) -> None:
    source = tmp_path / "mod.kspkg"
    source.write_bytes(b"payload")
    mods = tmp_path / "mods"
    assert main(["install", str(source), "--mods", str(mods)]) == 0
    assert (mods / "mod.kspkg").is_file()


def test_default_mods_dir_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    target = tmp_path / "custom-mods"
    monkeypatch.setenv("ACE_MODS_DIR", str(target))
    assert default_mods_dir() == target


def test_default_mods_dir_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ACE_MODS_DIR", raising=False)
    resolved = default_mods_dir()
    assert resolved.name == "mods"
    assert resolved.parent.name == "ACE"
