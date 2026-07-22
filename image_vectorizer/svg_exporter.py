"""
SVG exporter for the ImageVectorizer pipeline.

Converts detected primitives (rect, circle, ellipse, polygon, triangle)
into clean SVG elements — one element per detected object.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import xml.dom.minidom as md
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np

from .models import (
    RectanglePrimitive, SquarePrimitive, CirclePrimitive,
    EllipsePrimitive, TrianglePrimitive, PolygonPrimitive,
    LinePrimitive, PolylinePrimitive, PrimitiveBase, Color,
)


def export_svg(primitives: List[PrimitiveBase], width: int, height: int,
               output_path: str, colored: bool = True) -> str:
    """Export detected primitives as clean SVG.

    Args:
        primitives: List of detected primitives.
        width: Image width.
        height: Image height.
        output_path: Output SVG file path.
        colored: Use fill colors from primitives.

    Returns:
        Output file path.
    """
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"0 0 {width} {height}",
        "width": str(width), "height": str(height),
    })

    # Background
    ET.SubElement(svg, "rect", {
        "width": str(width), "height": str(height), "fill": "white",
    })

    counts: Dict[str, int] = {}

    for obj in primitives:
        fill = obj.fill_color.to_hex() if (colored and obj.fill_color) else "none"
        stroke = obj.stroke_color.to_hex() if obj.stroke_color else "#000"
        sw = str(obj.stroke_width)

        if isinstance(obj, (RectanglePrimitive, SquarePrimitive)):
            rx = getattr(obj, 'corner_radius', 0) or 0
            w, h = (obj.width, obj.height) if hasattr(obj, 'width') else (obj.side_length, obj.side_length)
            x, y = obj.x, obj.y
            rot = getattr(obj, 'rotation', 0) or 0
            cx, cy = x + w/2, y + h/2
            t = f"rotate({rot:.1f} {cx:.1f} {cy:.1f})" if abs(rot) > 0.1 else ""
            ET.SubElement(svg, "rect", {
                "x": f"{x:.1f}", "y": f"{y:.1f}",
                "width": f"{w:.1f}", "height": f"{h:.1f}",
                "rx": f"{rx:.1f}", "fill": fill, "stroke": stroke,
                "stroke-width": sw, "transform": t,
            })
            counts["rect"] = counts.get("rect", 0) + 1

        elif isinstance(obj, CirclePrimitive):
            ET.SubElement(svg, "circle", {
                "cx": f"{obj.center_x:.1f}", "cy": f"{obj.center_y:.1f}",
                "r": f"{obj.radius:.1f}", "fill": fill, "stroke": stroke,
                "stroke-width": sw,
            })
            counts["circle"] = counts.get("circle", 0) + 1

        elif isinstance(obj, EllipsePrimitive):
            t = f"rotate({obj.rotation:.1f} {obj.center_x:.1f} {obj.center_y:.1f})" if abs(obj.rotation) > 0.1 else ""
            ET.SubElement(svg, "ellipse", {
                "cx": f"{obj.center_x:.1f}", "cy": f"{obj.center_y:.1f}",
                "rx": f"{obj.semi_major:.1f}", "ry": f"{obj.semi_minor:.1f}",
                "fill": fill, "stroke": stroke, "stroke-width": sw,
                "transform": t,
            })
            counts["ellipse"] = counts.get("ellipse", 0) + 1

        elif isinstance(obj, TrianglePrimitive):
            pts = " ".join(f"{v.x:.1f},{v.y:.1f}" for v in obj.vertices)
            ET.SubElement(svg, "polygon", {
                "points": pts, "fill": fill, "stroke": stroke,
                "stroke-width": sw, "stroke-linejoin": "round",
            })
            counts["triangle"] = counts.get("triangle", 0) + 1

        elif isinstance(obj, PolygonPrimitive):
            # Simplify vertices
            verts = [(p.x, p.y) for p in obj.vertices]
            if len(verts) > 12:
                pts_arr = np.array([[list(v)] for v in verts], dtype=np.int32)
                peri = cv2.arcLength(pts_arr, True)
                approx = cv2.approxPolyDP(pts_arr, 0.015 * peri, True)
                verts = [(float(p[0][0]), float(p[0][1])) for p in approx]
            pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in verts)
            ET.SubElement(svg, "polygon", {
                "points": pts, "fill": fill, "stroke": stroke,
                "stroke-width": sw, "stroke-linejoin": "round",
            })
            counts["polygon"] = counts.get("polygon", 0) + 1

        elif isinstance(obj, LinePrimitive):
            ET.SubElement(svg, "line", {
                "x1": f"{obj.start_x:.1f}", "y1": f"{obj.start_y:.1f}",
                "x2": f"{obj.end_x:.1f}", "y2": f"{obj.end_y:.1f}",
                "stroke": fill if fill != "none" else "#000",
                "stroke-width": sw, "stroke-linecap": "round",
            })
            counts["line"] = counts.get("line", 0) + 1

    # Pretty-print
    xml_str = ET.tostring(svg, encoding="unicode")
    dom = md.parseString(xml_str)
    pretty = dom.toprettyxml(indent="  ")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)

    return output_path
