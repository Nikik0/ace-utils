"""Tests for the ACE ``.track`` converter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from src.ace_formats import new_message
from src.convert.track import build_track_data, convert_track
from src.parser.ac1 import AcTrack

_REF = Path("test_data/ace_ref/content/tracks/ek_tsubak_hv/ek_tsubak_hv.track")


def _parse(data: bytes) -> Any:
    msg = new_message(".track")
    msg.ParseFromString(data)
    return msg


def test_build_track_data_fields() -> None:
    msg = _parse(build_track_data().SerializeToString())
    assert msg.dynamicTrack.enabled is True
    assert msg.dynamicTrack.baseGrip == pytest.approx(0.98)
    assert msg.dynamicTrack.track_extension_m == pytest.approx(6000.0)
    assert msg.track_altitude_m == pytest.approx(55.0)


def test_build_track_data_sets_submessage_presence() -> None:
    msg = _parse(build_track_data().SerializeToString())
    assert msg.dynamicTrack.HasField("track_center")
    assert msg.HasField("aiSplineHintsData")


def test_build_track_data_overrides() -> None:
    msg = build_track_data(enabled=False, base_grip=1.0, track_altitude_m=10.0)
    parsed = _parse(msg.SerializeToString())
    assert parsed.dynamicTrack.enabled is False
    assert parsed.dynamicTrack.baseGrip == pytest.approx(1.0)
    assert parsed.track_altitude_m == pytest.approx(10.0)


def test_convert_track_uses_defaults(tmp_path: Path) -> None:
    track = AcTrack(id="t", root=tmp_path)
    assert convert_track(track).SerializeToString() == build_track_data().SerializeToString()


@pytest.mark.skipif(not _REF.is_file(), reason="reference package not extracted")
def test_build_track_data_matches_reference() -> None:
    assert build_track_data().SerializeToString() == _REF.read_bytes()


def test_new_message_unknown_extension() -> None:
    with pytest.raises(ValueError, match="no protobuf schema"):
        new_message(".nope")
