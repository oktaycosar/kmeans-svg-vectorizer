"""
SVG generator from recovered objects — one object = one SVG element.

Generates clean, editable SVG with minimal elements:
rect, circle, ellipse, polygon, path (for complex shapes).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import xml.dom.minidom as md
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

from .primitive_recovery import RecoveredPrimitive


def generate_svg(
    objects: List[Dict[str, Any]],
    width: int,
    height: int,
    output_path: str,
) -> str:
    """Generate clean SVG from recovered objects.

    One logical object → one SVG element (rect/circle/ellipse/polygon/path).

    Args:
        objects: List of object dicts with primitive info.
        width: Image width.
        height: Image height.
        output_path: Output SVG file path.

    Returns:
        Output file path.
    """
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"0 0 {width} {height}",
        "width": str(width),
        "height": str(height),
    })

    # Sort by z-order (background first)
    sorted_objects = sorted(objects, key=lambda o: o.get("z_order", 0))

    element_count = 0

    for obj in sorted_objects:
        primitive = obj.get("primitive")
        color = obj.get("color_hex", "#000000")
        has_fill = obj.get("has_fill", True)
        has_stroke = obj.get("has_stroke", False)
        stroke_color = obj.get("stroke_color", "#000000")
        stroke_width = obj.get("stroke_width", 1.0)

        if primitive is None:
            # Fallback: use contour as path
            contour = obj.get("contour")
            if contour is not None and len(contour) >= 3:
                path_data = _contour_to_path_data(contour)
                _add_path(svg, path_data, color if has_fill else "none",
                         stroke_color if has_stroke else "none", stroke_width)
                element_count += 1
            continue

        ptype = primitive.type
        params = primitive.params

        if ptype in ("rectangle", "rounded_rectangle"):
            rx = params.get("corner_radius", 0)
            ET.SubElement(svg, "rect", {
                "x": f"{params['x']:.1f}",
                "y": f"{params['y']:.1f}",
                "width": f"{params['width']:.1f}",
                "height": f"{params['height']:.1f}",
                "rx": f"{rx:.1f}" if rx > 0 else "0",
                "transform": _rotation_transform(params.get("rotation", 0),
                                                  params.get("center_x", 0),
                                                  params.get("center_y", 0)),
                "fill": color,
            })
            element_count += 1

        elif ptype == "circle":
            ET.SubElement(svg, "circle", {
                "cx": f"{params['center_x']:.1f}",
                "cy": f"{params['center_y']:.1f}",
                "r": f"{params['radius']:.1f}",
                "fill": color,
            })
            element_count += 1

        elif ptype == "ellipse":
            ET.SubElement(svg, "ellipse", {
                "cx": f"{params['center_x']:.1f}",
                "cy": f"{params['center_y']:.1f}",
                "rx": f"{params['semi_major']:.1f}",
                "ry": f"{params['semi_minor']:.1f}",
                "transform": _rotation_transform(params.get("rotation", 0),
                                                  params.get("center_x", 0),
                                                  params.get("center_y", 0)),
                "fill": color,
            })
            element_count += 1

        elif ptype in ("triangle", "quadrilateral", "pentagon", "hexagon", "octagon", "polygon"):
            verts = params.get("vertices", [])
            # Simplify: if too many vertices, re-approximate from contour
            if len(verts) > 12:
                contour = obj.get("contour")
                if contour is not None:
                    perimeter = cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    verts = [(float(pt[0][0]), float(pt[0][1])) for pt in approx]
            if len(verts) >= 3:
                if len(verts) > 12:
                    verts = verts[:12]  # Hard cap
                points_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in verts)
                ET.SubElement(svg, "polygon", {
                    "points": points_str,
                    "fill": color,
                })
                element_count += 1

        else:
            # Freeform/bezier: use contour
            contour = obj.get("contour")
            if contour is not None:
                path_data = _contour_to_path_data(contour)
                _add_path(svg, path_data, color, "none", 0)
                element_count += 1

    # Pretty print
    xml_str = ET.tostring(svg, encoding="unicode")
    dom = md.parseString(xml_str)
    pretty = dom.toprettyxml(indent="  ")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)

    return output_path


def _rotation_transform(angle: float, cx: float, cy: float) -> str:
    """SVG rotation transform string."""
    if abs(angle) < 0.1:
        return ""
    return f"rotate({angle:.1f} {cx:.1f} {cy:.1f})"


def _contour_to_path_data(contour: np.ndarray) -> str:
    """Convert contour to SVG path data."""
    pts = contour.reshape(-1, 2)
    if len(pts) < 2:
        return ""
    parts = [f"M {pts[0][0]:.1f} {pts[0][1]:.1f}"]
    for pt in pts[1:]:
        parts.append(f"L {pt[0]:.1f} {pt[1]:.1f}")
    parts.append("Z")
    return " ".join(parts)


def _add_path(svg, path_data, fill, stroke, stroke_width):
    """Add a path element to SVG."""
    ET.SubElement(svg, "path", {
        "d": path_data,
        "fill": fill,
        "stroke": stroke,
        "stroke-width": str(stroke_width),
        "stroke-linejoin": "round",
    })
