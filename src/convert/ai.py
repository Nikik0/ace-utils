"""Best-effort conversion of the AC1 AI spline into ACE AI data.

EvoForge resamples the ideal line and recomputes payloads with its own solver;
we instead carry the AC1 polyline and its geometry-derived fields (sides,
camber, direction, grade) into the ACE messages. The engine can rebuild the
driving line from ``spline_data`` (see ``AISplineData.BuilderCommand``).
"""

from __future__ import annotations

import math
from typing import Any, cast

from google.protobuf.message import Message

from src import mapping
from src.ace_formats import new_message
from src.parser.ai_spline import AiSpline

INT32_MIN = -2147483648


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.dist(a, b)


def _cumulative_lengths(spline: AiSpline) -> list[float]:
    lengths = [0.0]
    total = 0.0
    for index in range(1, len(spline.points)):
        total += _distance(spline.points[index - 1].position, spline.points[index].position)
        lengths.append(total)
    return lengths


def _is_closed(spline: AiSpline) -> bool:
    if len(spline.points) < 3:
        return False
    return _distance(spline.points[0].position, spline.points[-1].position) < 5.0


def _set_vector(target: Any, value: tuple[float, float, float]) -> None:
    target.x, target.y, target.z = value


def _fill_control_points(control_points: Any, spline: AiSpline, total: float) -> None:
    count = len(spline.points)
    if count == 0:
        return
    step = max(1, count // 169)
    last = total or 1.0
    for index in range(0, count, step):
        point = spline.points[index]
        extra = spline.extra[index]
        entry = control_points.control_points.add()
        _set_vector(entry.ideal_line_position, point.position)
        _set_vector(entry.forward_vector, extra.forward_vector)
        entry.side_left = extra.side_left
        entry.side_left_curb = extra.side_left
        entry.side_right = extra.side_right
        entry.side_right_curb = extra.side_right
        entry.side_left_wall = extra.side_left
        entry.side_right_wall = extra.side_right
        entry.signed_radius = extra.radius
        length = point.length if point.length > 0 else step
        entry.length_meters = length
        entry.spline_pos = _cumulative_at(spline, index) / last
        entry.lookahead_multiplier = 1.0
        entry.push_multiplier = 1.0
        entry.push_multiplier_2 = 1.0


def _cumulative_at(spline: AiSpline, index: int) -> float:
    total = 0.0
    for i in range(1, index + 1):
        total += _distance(spline.points[i - 1].position, spline.points[i].position)
    return total


def build_ai_spline(
    spline: AiSpline, *, track_id: str, layout: str
) -> Message:
    """Build the ``.aisplinedata`` message."""
    msg = cast(Any, new_message(".aisplinedata"))
    lengths = _cumulative_lengths(spline)
    total = lengths[-1] if lengths else 0.0

    spline_data = msg.spline_data
    spline_data.is_closed = _is_closed(spline)
    spline_data.length = total
    spline_data.mode = 0
    for index, point in enumerate(spline.points):
        entry = spline_data.points.add()
        _set_vector(entry.point, point.position)
        entry.point_length = lengths[index]
        entry.tag = index

    for extra in spline.extra:
        payload = msg.payloads.add()
        payload.speedMS = extra.speed
        payload.radius = extra.radius
        payload.sides_0 = extra.side_left
        payload.sides_1 = extra.side_right
        payload.boundaries_0 = extra.side_left
        payload.boundaries_1 = extra.side_right
        payload.walls_0 = extra.side_left
        payload.walls_1 = extra.side_right
        payload.sides_with_curbs_0 = extra.side_left
        payload.sides_with_curbs_1 = extra.side_right
        payload.effective_track_limits_0 = extra.side_left
        payload.effective_track_limits_1 = extra.side_right
        payload.camber = extra.camber
        payload.direction = extra.direction
        payload.grade = extra.grade
        payload.gas = extra.gas
        payload.grip = 1.0
        payload.length = extra.length
        _set_vector(payload.normal, extra.normal)
        _set_vector(payload.forwardVector, extra.forward_vector)

    msg.straight_factor = -1.0
    msg.layout_path = mapping.as_message_path(mapping.layout_track_layout(track_id, layout))
    _fill_control_points(msg.control_points, spline, total)
    return cast(Message, msg)


def build_control_points(spline: AiSpline, *, track_id: str, layout: str) -> Message:
    """Build the standalone ``.trackcontrolpoints`` message."""
    msg = cast(Any, new_message(".trackcontrolpoints"))
    total = _cumulative_lengths(spline)[-1] if spline.points else 0.0
    _fill_control_points(msg, spline, total)
    return cast(Message, msg)


def build_reference(spline: AiSpline) -> Message:
    """Build the ``.reference`` (AISplineReferenceData) message."""
    msg = cast(Any, new_message(".reference"))
    lengths = _cumulative_lengths(spline)
    total = lengths[-1] if lengths else 0.0
    msg.is_closed = _is_closed(spline)
    msg.track_meters = total
    msg.reference_laptime = 0
    last = total or 1.0
    for index, extra in enumerate(spline.extra):
        payload = msg.payloads.add()
        payload.spline_position = lengths[index] / last
        payload.reference_speed_ms = extra.speed
        payload.signed_radius = INT32_MIN
        payload.push_multiplier = 1.0
    return cast(Message, msg)
