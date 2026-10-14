"""Generate ACE UI resources (``map.svg``, ``previews.json``) from AC1 data."""

from __future__ import annotations

import datetime
import json
from typing import Any

from src.parser.ai_spline import AiSpline

DEFAULT_MARGIN = 10.0


def build_map_svg(
    spline: AiSpline, *, size: int = 1024, margin: float = DEFAULT_MARGIN
) -> tuple[bytes, dict[str, Any]]:
    """Render the AI centreline as an SVG track map.

    Returns the SVG bytes and the ``layoutImage`` projection used.
    """
    positions = spline.positions
    if not positions:
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}"/>'
        return svg.encode("utf-8"), {
            "margin": margin,
            "minX": 0.0,
            "minY": 0.0,
            "k": 1.0,
            "offsetX": 0.0,
            "offsetY": 0.0,
        }

    xs = [p[0] for p in positions]
    zs = [p[2] for p in positions]
    min_x, max_x = min(xs), max(xs)
    min_z, max_z = min(zs), max(zs)
    span_x = (max_x - min_x) or 1.0
    span_z = (max_z - min_z) or 1.0
    scale = min((size - 2 * margin) / span_x, (size - 2 * margin) / span_z)
    offset_x = (size - scale * span_x) / 2
    offset_z = (size - scale * span_z) / 2

    points = [
        (offset_x + (x - min_x) * scale, size - (offset_z + (z - min_z) * scale))
        for x, z in zip(xs, zs, strict=True)
    ]
    path = "M " + " L ".join(f"{u:.2f},{v:.2f}" for u, v in points)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 {size} {size}">'
        f'<path d="{path}" fill="none" stroke="#ffffff" stroke-width="3" '
        f'stroke-linejoin="round"/></svg>'
    )
    layout_image = {
        "margin": margin,
        "minX": min_x,
        "minY": min_z,
        "k": scale,
        "offsetX": offset_x,
        "offsetY": offset_z,
    }
    return svg.encode("utf-8"), layout_image


def _empty_layout_image() -> dict[str, Any]:
    return {
        "margin": DEFAULT_MARGIN,
        "minX": 0.0,
        "minY": 0.0,
        "k": 1.0,
        "offsetX": 0.0,
        "offsetY": 0.0,
    }


def build_previews_json(
    track_id: str,
    layout: str,
    *,
    layout_image: dict[str, Any] | None = None,
    sources: dict[str, str] | None = None,
) -> bytes:
    """Serialize an ACE ``previews.json`` document."""
    document = {
        "generatedAt": datetime.datetime.now(datetime.UTC).isoformat(),
        "trackId": track_id,
        "layout": layout,
        "sources": sources or {},
        "photo": False,
        "map": False,
        "svg": True,
        "layoutImage": layout_image or _empty_layout_image(),
    }
    return (json.dumps(document, indent="\t") + "\n").encode("utf-8")
