"""
SVG exporter — converts detected contours to clean SVG paths.

Provides immediate visual feedback on vectorization quality
by rendering OpenCV contours as SVG elements with fill/stroke.
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional
import xml.etree.ElementTree as ET

from .utils import logger


def contours_to_svg(
    contours: List[np.ndarray],
    image_width: int,
    image_height: int,
    output_path: str,
    simplify_epsilon: float = 1.0,
    min_area: float = 20.0,
) -> str:
    """Convert OpenCV contours to an SVG file.

    Each contour becomes a <path> element. Contours are simplified
    using Douglas-Peucker to reduce vertex count.

    Args:
        contours: List of OpenCV contours.
        image_width: Original image width for viewBox.
        image_height: Original image height for viewBox.
        output_path: Output SVG file path.
        simplify_epsilon: Epsilon for approxPolyDP (higher = simpler).
        min_area: Minimum contour area to include.

    Returns:
        Output file path.
    """
    # Register SVG namespace
    ET.register_namespace("", "http://www.w3.org/2000/svg")

    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"0 0 {image_width} {image_height}",
        "width": str(image_width),
        "height": str(image_height),
    })

    # Add a white background
    bg = ET.SubElement(svg, "rect", {
        "width": str(image_width),
        "height": str(image_height),
        "fill": "white",
    })

    path_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue

        # Simplify contour
        perimeter = cv2.arcLength(contour, True)
        epsilon = simplify_epsilon * max(perimeter * 0.001, 0.5)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Build SVG path data
        if len(approx) < 2:
            continue

        path_data = _build_path_data(approx, closed=True)

        # Create path element
        ET.SubElement(svg, "path", {
            "d": path_data,
            "fill": "none",
            "stroke": "black",
            "stroke-width": "1",
            "stroke-linejoin": "round",
            "stroke-linecap": "round",
        })
        path_count += 1

    # Pretty-print and save
    xml_str = ET.tostring(svg, encoding="unicode")
    # Manual pretty-print
    import xml.dom.minidom as md
    dom = md.parseString(xml_str)
    pretty = dom.toprettyxml(indent="  ")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)

    logger.info("SVG exported: %s (%d paths)", output_path, path_count)
    return output_path


def _build_path_data(approx: np.ndarray, closed: bool = True) -> str:
    """Build SVG path 'd' attribute from contour points.

    Args:
        approx: Simplified contour array (N, 1, 2).
        closed: Whether to close the path.

    Returns:
        SVG path data string.
    """
    if len(approx) == 0:
        return ""

    points = approx.reshape(-1, 2)
    parts = []

    # Move to first point
    parts.append(f"M {points[0][0]:.1f} {points[0][1]:.1f}")

    # Line to remaining points
    for pt in points[1:]:
        parts.append(f"L {pt[0]:.1f} {pt[1]:.1f}")

    if closed and len(points) > 2:
        parts.append("Z")

    return " ".join(parts)


def image_to_svg_contours(
    image: np.ndarray,
    output_path: str,
    min_area: float = 30.0,
) -> str:
    """Extract contours from an image and export as SVG.

    Convenience function: runs Canny → findContours → SVG.

    Args:
        image: BGR image.
        output_path: Output SVG path.
        min_area: Minimum contour area.

    Returns:
        Output file path.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    h, w = image.shape[:2]
    return contours_to_svg(contours, w, h, output_path, min_area=min_area)


def image_to_svg_filled(
    image: np.ndarray,
    output_path: str,
    min_area: float = 50.0,
) -> str:
    """Export as SVG with filled color regions.

    Uses color quantization + contour detection to create
    filled SVG paths that approximate the original image.

    Args:
        image: BGR image.
        output_path: Output SVG path.
        min_area: Minimum region area.

    Returns:
        Output file path.
    """
    h, w = image.shape[:2]

    ET.register_namespace("", "http://www.w3.org/2000/svg")
    svg = ET.Element("svg", {
        "xmlns": "http://www.w3.org/2000/svg",
        "viewBox": f"0 0 {w} {h}",
        "width": str(w),
        "height": str(h),
    })

    # Quantize to reduce colors
    pixels = image.reshape(-1, 3).astype(np.float32)
    k = min(16, max(2, len(np.unique(pixels // 32 * 32, axis=0))))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)

    for i in range(k):
        # Create binary mask for this color
        mask = (labels.flatten() == i).reshape(h, w).astype(np.uint8) * 255

        # Find contours in mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        color = centers[i]
        hex_color = f"#{color[2]:02x}{color[1]:02x}{color[0]:02x}"  # BGR→RGB hex

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            epsilon = 0.001 * perimeter
            approx = cv2.approxPolyDP(contour, epsilon, True)
            path_data = _build_path_data(approx, closed=True)

            ET.SubElement(svg, "path", {
                "d": path_data,
                "fill": hex_color,
                "stroke": hex_color,
                "stroke-width": "0.5",
            })

    # Pretty-print
    import xml.dom.minidom as md
    xml_str = ET.tostring(svg, encoding="unicode")
    dom = md.parseString(xml_str)
    pretty = dom.toprettyxml(indent="  ")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(pretty)

    logger.info("SVG filled exported: %s (%d colors)", output_path, k)
    return output_path
