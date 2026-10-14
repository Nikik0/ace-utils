"""Round-trip tests for the recovered ACE protobuf schemas.

Proof of schema recovery: every reference asset in the converted track must
parse into its message and survive re-serialization without loss (idempotent).
Extensions whose writer emits fields in canonical order must reproduce the
original bytes exactly.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from src.ace_formats import FORMATS, decode, format_for

REFERENCE_ROOT = Path("test_data/ace_ref")

pytestmark = pytest.mark.skipif(
    not REFERENCE_ROOT.is_dir(), reason="reference tree not extracted (test_data/ace_ref)"
)

BYTE_EXACT_EXTENSIONS = {".track", ".mesh", ".texture", ".trackcontrolpoints", ".aisplinedata"}


def _files(extension: str) -> list[Path]:
    return sorted(REFERENCE_ROOT.rglob(f"*{extension}"))


@pytest.mark.parametrize("fmt", FORMATS, ids=[f.extension for f in FORMATS])
def test_roundtrip_is_lossless(fmt: object) -> None:
    from src.ace_formats import AceFormat

    assert isinstance(fmt, AceFormat)
    files = _files(fmt.extension)
    assert files, f"no reference files for {fmt.extension}"
    cls = fmt.message_class()
    for path in files:
        data = path.read_bytes()
        message = cls()
        message.ParseFromString(data)
        reparsed = cls()
        reparsed.ParseFromString(message.SerializeToString())
        assert message == reparsed, f"lossy round-trip: {path}"


@pytest.mark.parametrize("extension", sorted(BYTE_EXACT_EXTENSIONS))
def test_byte_exact_extensions(extension: str) -> None:
    fmt = format_for(f"x{extension}")
    assert fmt is not None
    cls = fmt.message_class()
    for path in _files(extension):
        message = cls()
        message.ParseFromString(path.read_bytes())
        assert message.SerializeToString() == path.read_bytes(), f"not byte exact: {path}"


def test_format_for_unknown_extension() -> None:
    assert format_for("foo.texturemips") is None
    assert format_for("downhill_real.track_layout") is None


def test_track_metadata_values() -> None:
    track = decode(REFERENCE_ROOT / "content/tracks/ek_tsubak_hv/ek_tsubak_hv.track")
    assert track.dynamicTrack.enabled
    assert track.dynamicTrack.baseGrip == pytest.approx(0.98, rel=1e-6)
    assert track.dynamicTrack.track_extension_m == pytest.approx(6000.0)
    assert track.track_altitude_m == pytest.approx(55.0)


def test_root_scene_has_actors() -> None:
    scene = decode(REFERENCE_ROOT / "content/tracks/ek_tsubak_hv/ek_tsubak_hv.scene")
    names = [actor.name for actor in scene.scene_actors]
    assert "AC1 surfaces" in names
