"""
Object classification module.

Classifies enriched contours into specific primitive types,
assigning unique IDs and confidence scores.

Determines the best-matching primitive type for each contour,
avoiding duplicate detections.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np

from .config import (
    PipelineConfig, RectangleConfig, CircleConfig,
    EllipseConfig, PolygonConfig, LineConfig,
)
from .models import (
    PrimitiveBase, PrimitiveType,
    RectanglePrimitive, SquarePrimitive, CirclePrimitive,
    EllipsePrimitive, TrianglePrimitive, PolygonPrimitive,
    LinePrimitive, PolylinePrimitive, BezierCandidatePrimitive,
)
from .primitive_detector import (
    detect_rectangle, detect_square, detect_circle,
    detect_ellipse, detect_triangle, detect_polygon,
    detect_lines_hough, detect_circles_hough, detect_polyline,
)
from .color_analyzer import analyze_primitive_colors, quantize_color
from .utils import logger


def classify_contours(
    enriched_contours: List[Dict[str, Any]],
    image_bgr: np.ndarray,
    gray_image: np.ndarray,
    edges_image: Optional[np.ndarray],
    config: PipelineConfig,
) -> List[PrimitiveBase]:
    """Classify enriched contours into primitive types.

    For each contour, tries to match it to the best primitive type.
    Higher-confidence types (rectangle, circle) are tried first.

    Args:
        enriched_contours: List of enriched contour dicts from contour_detector.
        image_bgr: Original BGR image for color analysis.
        gray_image: Grayscale image for Hough transforms.
        edges_image: Canny edges for line detection (optional).
        config: Full pipeline configuration.

    Returns:
        List of classified primitives with unique IDs.
    """
    primitives: List[PrimitiveBase] = []
    detected_ids: set = set()  # Track contour indices to avoid duplicates

    for contour_data in enriched_contours:
        contour_idx = contour_data["index"]
        contour = contour_data["contour"]
        area = contour_data["area"]
        circularity = contour_data.get("circularity", 0)
        rectangularity = contour_data.get("rectangularity", 0)

        best_primitive: Optional[PrimitiveBase] = None
        best_confidence: float = 0.0

        # ---- Try Circle (high circularity) ----
        if circularity > config.circle.circularity_threshold - 0.1:
            circle = detect_circle(contour_data, config.circle)
            if circle and circle.confidence > best_confidence:
                best_primitive = circle
                best_confidence = circle.confidence

        # ---- Try Ellipse (moderate circularity) ----
        if circularity > 0.3 and circularity < config.circle.circularity_threshold:
            ellipse = detect_ellipse(contour_data, config.ellipse)
            if ellipse and ellipse.confidence > best_confidence:
                best_primitive = ellipse
                best_confidence = ellipse.confidence

        # ---- Try Rectangle (high rectangularity) ----
        if rectangularity > config.rectangle.min_confidence - 0.2:
            rect = detect_rectangle(contour_data, config.rectangle)
            if rect:
                # Check if it's a square
                square = detect_square(rect, config.rectangle)
                if square and square.confidence > best_confidence:
                    best_primitive = square
                    best_confidence = square.confidence
                elif rect.confidence > best_confidence:
                    best_primitive = rect
                    best_confidence = rect.confidence

        # ---- Try Triangle ----
        triangle = detect_triangle(contour_data, config.polygon)
        if triangle and triangle.confidence > best_confidence:
            best_primitive = triangle
            best_confidence = triangle.confidence

        # ---- Try Polygon ----
        polygon = detect_polygon(contour_data, config.polygon)
        if polygon and polygon.confidence > best_confidence:
            best_primitive = polygon
            best_confidence = polygon.confidence

        # ---- Try Polyline (small area, elongated) ----
        if area < config.polygon.min_area * 0.5:
            polyline = detect_polyline(contour_data, config.polygon)
            if polyline and polyline.confidence > best_confidence:
                best_primitive = polyline
                best_confidence = polyline.confidence

        # ---- If nothing detected, classify as POLYGON if area is large enough ----
        if best_primitive is None and area > config.polygon.min_area:
            polygon = detect_polygon(contour_data, config.polygon)
            if polygon:
                best_primitive = polygon
                best_confidence = polygon.confidence

        # Register if found
        if best_primitive is not None:
            # Assign ID
            obj_id = len(primitives) + 1
            best_primitive.id = obj_id

            # Analyze colors
            if config.enable_color_analysis:
                analyze_primitive_colors(image_bgr, best_primitive, contour, config.color)

                # Quantize for cleaner output
                if config.color.quantization_levels > 0 and best_primitive.fill_color:
                    best_primitive.fill_color = quantize_color(
                        best_primitive.fill_color, config.color.quantization_levels
                    )
                if config.color.quantization_levels > 0 and best_primitive.stroke_color:
                    best_primitive.stroke_color = quantize_color(
                        best_primitive.stroke_color, config.color.quantization_levels
                    )

            primitives.append(best_primitive)

    # ---- Additional: Hough Line Detection (from edges, not contours) ----
    if edges_image is not None and config.line.hough_threshold > 0:
        hough_lines = detect_lines_hough(edges_image, config.line)
        for line in hough_lines:
            # Check if this line overlaps with existing primitives
            overlaps = _check_line_overlap(line, primitives)
            if not overlaps:
                line.id = len(primitives) + 1
                primitives.append(line)

    # ---- Additional: Hough Circle Detection (missed by contour approach) ----
    if gray_image is not None and config.circle.hough_param2 > 0:
        hough_circles = detect_circles_hough(gray_image, config.circle)
        for circle in hough_circles:
            if not _check_circle_duplicate(circle, primitives):
                circle.id = len(primitives) + 1
                if config.enable_color_analysis:
                    analyze_primitive_colors(image_bgr, circle, None, config.color)
                    if config.color.quantization_levels > 0 and circle.fill_color:
                        circle.fill_color = quantize_color(
                            circle.fill_color, config.color.quantization_levels
                        )
                primitives.append(circle)

    logger.info(
        "Classification complete: %d primitives detected "
        "(rect=%d, square=%d, circle=%d, ellipse=%d, "
        "triangle=%d, polygon=%d, line=%d, polyline=%d)",
        len(primitives),
        sum(1 for p in primitives if p.primitive_type == PrimitiveType.RECTANGLE),
        sum(1 for p in primitives if p.primitive_type == PrimitiveType.SQUARE),
        sum(1 for p in primitives if p.primitive_type == PrimitiveType.CIRCLE),
        sum(1 for p in primitives if p.primitive_type == PrimitiveType.ELLIPSE),
        sum(1 for p in primitives if p.primitive_type == PrimitiveType.TRIANGLE),
        sum(1 for p in primitives if p.primitive_type == PrimitiveType.POLYGON),
        sum(1 for p in primitives if p.primitive_type == PrimitiveType.LINE),
        sum(1 for p in primitives if p.primitive_type == PrimitiveType.POLYLINE),
    )

    return primitives


def _check_line_overlap(
    line: LinePrimitive,
    primitives: List[PrimitiveBase],
    distance_threshold: float = 15.0,
) -> bool:
    """Check if a detected line overlaps with any existing primitive.

    Args:
        line: Candidate line.
        primitives: Existing primitives.
        distance_threshold: Max distance for overlap detection.

    Returns:
        True if the line overlaps.
    """
    line_center = line.midpoint
    for p in primitives:
        if p.bounding_box is not None:
            bbox = p.bounding_box
            cx, cy = bbox.center.x, bbox.center.y
            import math
            dist = math.hypot(line_center.x - cx, line_center.y - cy)
            if dist < max(bbox.width, bbox.height) * 0.5 + distance_threshold:
                return True
    return False


def _check_circle_duplicate(
    circle: CirclePrimitive,
    primitives: List[PrimitiveBase],
    distance_threshold: float = 30.0,
    radius_threshold: float = 0.3,
) -> bool:
    """Check if a Hough circle duplicates an existing circle/ellipse primitive.

    Args:
        circle: Candidate circle.
        primitives: Existing primitives.
        distance_threshold: Max center distance for duplicate.
        radius_threshold: Max radius difference ratio.

    Returns:
        True if duplicate.
    """
    for p in primitives:
        if isinstance(p, CirclePrimitive):
            center_dist = ((circle.center_x - p.center_x) ** 2 +
                          (circle.center_y - p.center_y) ** 2) ** 0.5
            radius_diff = abs(circle.radius - p.radius) / max(circle.radius, p.radius, 1)
            if center_dist < distance_threshold and radius_diff < radius_threshold:
                return True
        elif isinstance(p, EllipsePrimitive):
            center_dist = ((circle.center_x - p.center_x) ** 2 +
                          (circle.center_y - p.center_y) ** 2) ** 0.5
            avg_ellipse_radius = (p.semi_major + p.semi_minor) / 2
            radius_diff = abs(circle.radius - avg_ellipse_radius) / max(circle.radius, avg_ellipse_radius, 1)
            if center_dist < distance_threshold and radius_diff < radius_threshold:
                return True
    return False


def get_classification_stats(primitives: List[PrimitiveBase]) -> Dict[str, int]:
    """Return classification statistics.

    Args:
        primitives: List of classified primitives.

    Returns:
        Dict mapping type name to count.
    """
    stats: Dict[str, int] = {}
    for p in primitives:
        type_name = p.primitive_type.value
        stats[type_name] = stats.get(type_name, 0) + 1
    return stats
