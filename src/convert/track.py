"""Build the ACE ``.track`` (TrackData) asset."""

from __future__ import annotations

from typing import Any, cast

from google.protobuf.message import Message

from src import mapping
from src.ace_formats import new_message
from src.parser.ac1 import AcTrack


def build_track_data(
    *,
    enabled: bool = mapping.DYNAMIC_TRACK_ENABLED,
    base_grip: float = mapping.BASE_GRIP,
    track_extension_m: float = mapping.TRACK_EXTENSION_M,
    track_altitude_m: float = mapping.TRACK_ALTITUDE_M,
) -> Message:
    """Create a ``TrackData`` message.

    Presence of the empty ``track_center`` and ``aiSplineHintsData`` submessage
    fields is set explicitly so the serialized bytes match the reference
    packages exactly.
    """
    msg = cast(Any, new_message(".track"))
    dynamic = msg.dynamicTrack
    dynamic.enabled = enabled
    dynamic.baseGrip = base_grip
    dynamic.track_center.SetInParent()
    dynamic.track_extension_m = track_extension_m
    msg.aiSplineHintsData.SetInParent()
    msg.track_altitude_m = track_altitude_m
    return cast(Message, msg)


def convert_track(track: AcTrack) -> Message:
    """Build the ``.track`` asset for a parsed AC1 track."""
    del track
    return build_track_data()
