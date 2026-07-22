"""
Primitive shape detection module.

Detects geometric primitives from contours:
- Rectangles (axis-aligned and rotated)
- Rounded rectangles
- Squares
- Circles
- Ellipses
- Triangles
- Polygons
- Lines (straight and polylines)

Each detection returns a confidence score based on geometric properties.
"""

from __future__ import annotations

from typing import List, Dict, Any, Optional, Tuple

import cv2
import numpy as np

from .config import RectangleConfig, CircleConfig, EllipseConfig, LineConfig, PolygonConfig
from .models import (
    Point, BoundingBox, Color,
    RectanglePrimitive, SquarePrimitive, CirclePrimitive,
    EllipsePrimitive, TrianglePrimitive, PolygonPrimitive,
    LinePrimitive, PolylinePrimitive,
)
from .geometry import (
    simplify_contour, order_vertices_clockwise,
    compute_polygon_angles, detect_corner_radius,
    fit_ellipse_to_contour, fit_line_to_points,
    merge_nearby_lines,
)
from .utils import logger


def detect_rectangle(
    contour_data: Dict[str, Any],
    config: RectangleConfig,
) -> Optional[RectanglePrimitive]:
    """Detect if a contour represents a rectangle.

    Uses approxPolyDP to check for 4 vertices with near-right angles.

    Args:
        contour_data: Enriched contour dictionary from contour_detector.
        config: Rectangle detection configuration.

    Returns:
        RectanglePrimitive or None.
    """
    contour = contour_data["contour"]
    area = contour_data["area"]
    bbox = contour_data["bbox"]

    if area < config.min_area:
        return None

    # Simplify contour
    approx = simplify_contour(contour, config.epsilon_factor)
    num_vertices = len(approx)

    # Rectangle must have ~4 vertices
    if num_vertices < 4:
        # Check if the contour is rectangular despite few vertices
        rectangularity = contour_data.get("rectangularity", 0)
        if rectangularity < config.min_confidence:
            return None
    elif num_vertices > 8:
        # Too many vertices: not a simple rectangle
        return None

    # Use minAreaRect for precise orientation
    center, size, angle = cv2.minAreaRect(contour)
    w, h = size

    # Normalize: ensure width >= height
    if h > w:
        w, h = h, w
        angle = angle + 90

    # Ensure width and height are positive
    w, h = abs(w), abs(h)
    if w < 1 or h < 1:
        return None

    # Normalize angle to [-45, 45]
    if angle > 45:
        angle -= 90
    elif angle < -45:
        angle += 90

    aspect_ratio = w / h if h > 0 else float("inf")

    # Confidence based on multiple factors
    rectangularity = contour_data.get("rectangularity", 0)
    convexity = contour_data.get("convexity", 0)

    # A perfect rectangle has high rectangularity and convexity
    confidence = (rectangularity * 0.6 + convexity * 0.4)

    # Check for rounded corners
    corner_radius = detect_corner_radius(approx, contour)
    is_rounded = corner_radius > config.rounded_corner_max_angle

    if is_rounded:
        # Rounded rectangles may have lower rectangularity; adjust confidence
        confidence = max(confidence, 0.75)

    if confidence < config.min_confidence:
        return None

    x, y = bbox.x, bbox.y

    return RectanglePrimitive(
        id=0,  # Set by caller
        confidence=confidence,
        x=x,
        y=y,
        width=float(w),
        height=float(h),
        rotation=float(angle),
        is_rounded=is_rounded,
        corner_radius=corner_radius if is_rounded else 0.0,
        bounding_box=bbox,
        contour_area=area,
        contour_perimeter=contour_data["perimeter"],
        centroid=contour_data["centroid"],
    )


def detect_square(
    rectangle: RectanglePrimitive,
    config: RectangleConfig,
) -> Optional[SquarePrimitive]:
    """Check if a detected rectangle is actually a square.

    Args:
        rectangle: Already-detected rectangle.
        config: Rectangle configuration.

    Returns:
        SquarePrimitive or None.
    """
    ar = rectangle.aspect_ratio
    tolerance = config.square_aspect_ratio_tolerance

    if abs(ar - 1.0) > tolerance:
        return None

    side = (rectangle.width + rectangle.height) / 2.0
    confidence = rectangle.confidence * (1.0 - abs(ar - 1.0))

    return SquarePrimitive(
        id=0,
        confidence=confidence,
        x=rectangle.x,
        y=rectangle.y,
        side_length=side,
        rotation=rectangle.rotation,
        bounding_box=rectangle.bounding_box,
        contour_area=rectangle.contour_area,
        contour_perimeter=rectangle.contour_perimeter,
        centroid=rectangle.centroid,
    )


def detect_circle(
    contour_data: Dict[str, Any],
    config: CircleConfig,
) -> Optional[CirclePrimitive]:
    """Detect if a contour represents a circle.

    Uses circularity ratio and optionally HoughCircles.

    Args:
        contour_data: Enriched contour dictionary.
        config: Circle detection configuration.

    Returns:
        CirclePrimitive or None.
    """
    contour = contour_data["contour"]
    circularity = contour_data.get("circularity", 0)
    centroid = contour_data["centroid"]

    if circularity < config.circularity_threshold:
        return None

    # Compute radius from area
    area = contour_data["area"]
    if area <= 0:
        return None
    radius = np.sqrt(area / np.pi)

    # Compute radius from minEnclosingCircle
    (cx_enc, cy_enc), r_enc = cv2.minEnclosingCircle(contour)
    (cx_fit, cy_fit), r_fit = cv2.minEnclosingCircle(contour)

    # Confidence from circularity and radius agreement
    radius_agreement = 1.0 - abs(radius - r_enc) / max(radius, r_enc, 1)
    confidence = circularity * 0.6 + radius_agreement * 0.4

    if confidence < config.min_confidence:
        return None

    # Use enclosing circle center
    cx, cy = centroid.x, centroid.y

    return CirclePrimitive(
        id=0,
        confidence=confidence,
        center_x=cx,
        center_y=cy,
        radius=float(radius),
        bounding_box=contour_data["bbox"],
        contour_area=area,
        contour_perimeter=contour_data["perimeter"],
        centroid=centroid,
    )


def detect_circles_hough(
    gray_image: np.ndarray,
    config: CircleConfig,
) -> List[CirclePrimitive]:
    """Detect circles using Hough Transform.

    Args:
        gray_image: Grayscale image.
        config: Circle detection configuration.

    Returns:
        List of CirclePrimitive objects.
    """
    circles = cv2.HoughCircles(
        gray_image,
        cv2.HOUGH_GRADIENT,
        dp=config.hough_dp,
        minDist=config.hough_min_dist,
        param1=config.hough_param1,
        param2=config.hough_param2,
        minRadius=config.hough_min_radius,
        maxRadius=config.hough_max_radius,
    )

    if circles is None:
        return []

    results: List[CirclePrimitive] = []
    for circle in circles[0]:
        cx, cy, r = circle
        confidence = min_confidence = 0.85  # Hough circles are generally reliable
        results.append(CirclePrimitive(
            id=0,
            confidence=confidence,
            center_x=float(cx),
            center_y=float(cy),
            radius=float(r),
        ))

    return results


def detect_ellipse(
    contour_data: Dict[str, Any],
    config: EllipseConfig,
) -> Optional[EllipsePrimitive]:
    """Detect if a contour represents an ellipse.

    Fits an ellipse and checks the quality of the fit.

    Args:
        contour_data: Enriched contour dictionary.
        config: Ellipse detection configuration.

    Returns:
        EllipsePrimitive or None.
    """
    contour = contour_data["contour"]

    if len(contour) < config.min_contour_points:
        return None

    ellipse = fit_ellipse_to_contour(contour)
    if ellipse is None:
        return None

    center, axes, angle = ellipse
    a, b = axes

    # Ensure semi-major >= semi-minor
    if b > a:
        a, b = b, a
        angle += 90

    a, b = a / 2.0, b / 2.0  # Convert from full axes to semi-axes

    if a < 1 or b < 1:
        return None

    # Confidence: how well does the ellipse match the contour?
    # Use ellipse area vs contour area
    ellipse_area = np.pi * a * b
    contour_area = contour_data["area"]
    area_ratio = min(ellipse_area, contour_area) / max(ellipse_area, contour_area, 1)

    circularity = contour_data.get("circularity", 0)
    # Ellipses have moderate circularity; not too high (circle), not too low
    # Penalize if it's almost a circle
    aspect_ratio = b / a if a > 0 else 0
    ellipse_score = 1.0 - abs(0.5 - aspect_ratio) * 0.5  # Peak at aspect_ratio ~ 0.5

    confidence = area_ratio * 0.4 + circularity * 0.3 + ellipse_score * 0.3

    if confidence < config.min_confidence:
        return None

    centroid = contour_data["centroid"]

    return EllipsePrimitive(
        id=0,
        confidence=confidence,
        center_x=float(center[0]),
        center_y=float(center[1]),
        semi_major=float(a),
        semi_minor=float(b),
        rotation=float(angle),
        bounding_box=contour_data["bbox"],
        contour_area=contour_area,
        contour_perimeter=contour_data["perimeter"],
        centroid=centroid,
    )


def detect_triangle(
    contour_data: Dict[str, Any],
    config: PolygonConfig,
) -> Optional[TrianglePrimitive]:
    """Detect if a contour represents a triangle.

    Args:
        contour_data: Enriched contour dictionary.
        config: Polygon detection configuration.

    Returns:
        TrianglePrimitive or None.
    """
    contour = contour_data["contour"]
    area = contour_data["area"]

    if area < config.min_area:
        return None

    approx = simplify_contour(contour, config.epsilon_factor)
    num_vertices = len(approx)

    if num_vertices != 3:
        return None

    # Extract vertices
    vertices = [Point(float(pt[0][0]), float(pt[0][1])) for pt in approx]
    ordered = order_vertices_clockwise(vertices)

    if len(ordered) != 3:
        return None

    triangle = TrianglePrimitive(
        id=0,
        vertex_a=ordered[0],
        vertex_b=ordered[1],
        vertex_c=ordered[2],
        bounding_box=contour_data["bbox"],
        contour_area=area,
        contour_perimeter=contour_data["perimeter"],
        centroid=contour_data["centroid"],
        confidence=config.min_confidence,  # Base confidence
    )

    # Improve confidence: check convexity
    convexity = contour_data.get("convexity", 0)
    triangle.confidence = (convexity + (1.0 if area > config.min_area * 2 else 0.5)) / 2

    return triangle


def detect_polygon(
    contour_data: Dict[str, Any],
    config: PolygonConfig,
) -> Optional[PolygonPrimitive]:
    """Detect a general polygon (more than 3 sides).

    Args:
        contour_data: Enriched contour dictionary.
        config: Polygon detection configuration.

    Returns:
        PolygonPrimitive or None.
    """
    contour = contour_data["contour"]
    area = contour_data["area"]

    if area < config.min_area:
        return None

    approx = simplify_contour(contour, config.epsilon_factor)
    num_vertices = len(approx)

    if num_vertices < config.min_vertices or num_vertices > config.max_vertices:
        return None

    # Extract and order vertices
    vertices = [Point(float(pt[0][0]), float(pt[0][1])) for pt in approx]
    ordered = order_vertices_clockwise(vertices)

    # Confidence based on how well the polygon fits the contour
    perimeter = contour_data["perimeter"]
    approx_perimeter = sum(
        ordered[i].distance_to(ordered[(i + 1) % len(ordered)])
        for i in range(len(ordered))
    )
    perimeter_ratio = min(perimeter, approx_perimeter) / max(perimeter, approx_perimeter, 1)

    convexity = contour_data.get("convexity", 0)
    confidence = perimeter_ratio * 0.5 + convexity * 0.5

    if confidence < config.min_confidence:
        return None

    return PolygonPrimitive(
        id=0,
        vertices=ordered,
        num_sides=num_vertices,
        confidence=confidence,
        bounding_box=contour_data["bbox"],
        contour_area=area,
        contour_perimeter=perimeter,
        centroid=contour_data["centroid"],
    )


def detect_lines_hough(
    edges: np.ndarray,
    config: LineConfig,
) -> List[LinePrimitive]:
    """Detect straight lines using HoughLinesP.

    Args:
        edges: Binary edge image (Canny output).
        config: Line detection configuration.

    Returns:
        List of LinePrimitive objects.
    """
    lines = cv2.HoughLinesP(
        edges,
        rho=config.hough_rho,
        theta=config.hough_theta,
        threshold=config.hough_threshold,
        minLineLength=config.hough_min_line_length,
        maxLineGap=config.hough_max_line_gap,
    )

    if lines is None:
        return []

    # Merge similar lines
    merged = merge_nearby_lines(
        lines,
        angle_threshold_deg=config.merge_angle_threshold,
        distance_threshold=config.merge_distance_threshold,
    )

    results: List[LinePrimitive] = []
    for line in merged:
        if line.ndim == 2:
            x1, y1, x2, y2 = line[0]
        else:
            x1, y1, x2, y2 = line

        length = np.hypot(x2 - x1, y2 - y1)
        # Confidence based on length relative to threshold
        confidence = min(length / (config.hough_min_line_length * 3), 1.0)

        results.append(LinePrimitive(
            id=0,
            confidence=confidence,
            start_x=float(x1),
            start_y=float(y1),
            end_x=float(x2),
            end_y=float(y2),
        ))

    logger.debug("Detected %d lines (after merging)", len(results))
    return results


def detect_polyline(
    contour_data: Dict[str, Any],
    config: PolygonConfig,
) -> Optional[PolylinePrimitive]:
    """Detect a polyline (open contour with multiple segments).

    Polyline has fewer vertices than a polygon and is open.

    Args:
        contour_data: Enriched contour dictionary.
        config: Polygon detection configuration.

    Returns:
        PolylinePrimitive or None.
    """
    contour = contour_data["contour"]
    area = contour_data["area"]

    # Polylines have small area
    if area > config.min_area * 0.5:
        return None

    approx = simplify_contour(contour, config.epsilon_factor * 0.5)
    num_vertices = len(approx)

    if num_vertices < 2:
        return None

    vertices = [Point(float(pt[0][0]), float(pt[0][1])) for pt in approx]

    total_length = sum(
        vertices[i].distance_to(vertices[i + 1])
        for i in range(len(vertices) - 1)
    )

    if total_length < 10:
        return None

    confidence = min(num_vertices / 10.0, 1.0) * 0.7

    return PolylinePrimitive(
        id=0,
        points=vertices,
        confidence=confidence,
        contour_area=area,
        contour_perimeter=total_length,
        bounding_box=contour_data["bbox"],
    )
