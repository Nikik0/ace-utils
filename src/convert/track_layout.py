"""Best-effort ``.track_layout`` writer.

The reference ``.track_layout`` binary has this prefix (verified by inspection)::

    int32 version   # 1
    vec3  start_position
    vec3  end_position
    int32 reserved  # 0
    int32 count
    count * (vec3 point, float length)

The reference stores its points in a spatial (grid) order whose derivation is
unknown; we write the AI centreline in path order, which is the natural
interpretation of the format.
"""

from __future__ import annotations

import struct

from src.parser.ai_spline import AiSpline

VERSION = 1
DEFAULT_POINTS = 738


def _resample(
    points: list[tuple[float, float, float]], count: int
) -> list[tuple[float, float, float]]:
    if len(points) <= count:
        return points
    step = len(points) / count
    return [points[min(len(points) - 1, int(index * step))] for index in range(count)]


def build_track_layout(spline: AiSpline, *, points: int = DEFAULT_POINTS) -> bytes:
    """Serialize an AC1 AI centreline as a ``.track_layout`` file."""
    positions = _resample(spline.positions, points)
    start = positions[0] if positions else (0.0, 0.0, 0.0)
    end = positions[-1] if positions else (0.0, 0.0, 0.0)

    data = bytearray()
    data += struct.pack("<i", VERSION)
    data += struct.pack("<3f", *start)
    data += struct.pack("<3f", *end)
    data += struct.pack("<i", 0)
    data += struct.pack("<i", len(positions))

    total = 0.0
    previous = start
    for position in positions:
        total += (
            (position[0] - previous[0]) ** 2
            + (position[1] - previous[1]) ** 2
            + (position[2] - previous[2]) ** 2
        ) ** 0.5
        data += struct.pack("<3ff", *position, total)
        previous = position
    return bytes(data)
