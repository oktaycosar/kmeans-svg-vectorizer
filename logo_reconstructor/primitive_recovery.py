"""
Primitive recovery — reconstructs geometric primitives from regions.

Unlike contour approximation, this module recovers the ORIGINAL
geometric parameters: center, radius, width, height, rotation,
corner radius, Bezier control points.

Think: "What geometric primitive created this shape?"
"""

from __future__ import annotations

import cv2
import numpy as np
import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class RecoveredPrimitive:
    """A recovered geometric primitive with parameters."""
    type: str  # rectangle, rounded_rectangle, circle, ellipse, triangle, polygon, bezier, freeform
    confidence: float
    params: Dict[str, Any]
    svg_element: str = ""


def recover_rectangle(contour: np.ndarray, area: float) -> Optional[RecoveredPrimitive]:
    """Recover the original rectangle parameters.

    Uses minAreaRect for axis-aligned or rotated rectangles.

    Returns:
        RecoveredPrimitive or None if not rectangle-like.
    """
    if contour is None or len(contour) < 4:
        return None

    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    w, h = size

    if w < 2 or h < 2:
        return None

    # Normalize
    if h > w:
        w, h = h, w
        angle += 90
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    rect_area = w * h
    if rect_area < 1:
        return None

    fill_ratio = area / rect_area

    # Check corners: sample the 4 corners of the minAreaRect
    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # Count contour points near each corner
    corner_hits = 0
    corner_threshold = min(w, h) * 0.08  # Tighter threshold
    for corner in box:
        distances = np.linalg.norm(
            contour.reshape(-1, 2).astype(np.float32) - corner.astype(np.float32),
            axis=1
        )
        if np.min(distances) < corner_threshold:
            corner_hits += 1

    # Also check: does the contour have roughly 4 dominant corners?
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
    has_4_corners = len(approx) == 4

    confidence = fill_ratio * 0.4 + (corner_hits / 4) * 0.3 + (0.3 if has_4_corners else 0.0)

    if confidence < 0.7:
        return None

    return RecoveredPrimitive(
        type="rectangle",
        confidence=confidence,
        params={
            "x": float(center[0] - w / 2),
            "y": float(center[1] - h / 2),
            "width": float(w),
            "height": float(h),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "rotation": float(angle),
        },
    )


def recover_rounded_rectangle(
    contour: np.ndarray, area: float, perimeter: float,
    image_area: float = 1e9
) -> Optional[RecoveredPrimitive]:
    """Recover rounded rectangle parameters.

    Detects rounded corners by checking if contour deviates from
    the minAreaRect at the corners.
    """
    if contour is None or len(contour) < 8:
        return None

    rect = cv2.minAreaRect(contour)
    center, size, angle = rect
    w, h = size

    if w < 5 or h < 5:
        return None

    # Skip if this is a full-image background (not a real rounded rect)
    if (w * h) > image_area * 0.85:
        return None

    if h > w:
        w, h = h, w
        angle += 90

    box = cv2.boxPoints(rect)
    box = np.intp(box)

    # For each corner, find the max deviation of contour from rect corner
    corner_radii = []
    for corner in box:
        corner_f = corner.astype(np.float32)
        contour_pts = contour.reshape(-1, 2).astype(np.float32)
        distances = np.linalg.norm(contour_pts - corner_f, axis=1)
        nearest_idx = np.argmin(distances)

        # Sample contour points around this corner
        n_pts = len(contour_pts)
        window = min(15, n_pts // 8)
        max_dev = 0.0
        for offset in range(-window, window + 1):
            idx = (nearest_idx + offset) % n_pts
            d = np.linalg.norm(contour_pts[idx] - corner_f)
            max_dev = max(max_dev, d)

        corner_radii.append(max_dev)

    avg_corner_radius = np.mean(corner_radii) if corner_radii else 0
    # Clamp: corner radius can't exceed half the smaller dimension
    min_dim = min(w, h)
    avg_corner_radius = min(avg_corner_radius, min_dim / 2.5)

    # If corners deviate significantly, it's a rounded rectangle
    if avg_corner_radius < min_dim * 0.04:
        return None  # Too sharp, not rounded

    # Confidence based on how uniform the corner radii are
    radius_std = np.std(corner_radii) if len(corner_radii) > 1 else 0
    uniformity = max(0, 1.0 - radius_std / max(avg_corner_radius, 1))
    confidence = 0.7 + uniformity * 0.3

    return RecoveredPrimitive(
        type="rounded_rectangle",
        confidence=confidence,
        params={
            "x": float(center[0] - w / 2),
            "y": float(center[1] - h / 2),
            "width": float(w),
            "height": float(h),
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "rotation": float(angle),
            "corner_radius": float(avg_corner_radius),
        },
    )


def recover_circle(contour: np.ndarray, area: float, perimeter: float) -> Optional[RecoveredPrimitive]:
    """Recover circle parameters: center and radius."""
    if contour is None or len(contour) < 5:
        return None

    # Circularity check
    circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

    if circularity < 0.85:
        return None

    # Fit enclosing circle
    (cx, cy), radius = cv2.minEnclosingCircle(contour)

    # Check that enclosing circle is reasonable
    circle_area = np.pi * radius * radius
    area_ratio = min(area, circle_area) / max(area, circle_area, 1)

    # Also fit using moments for center
    moments = cv2.moments(contour)
    if moments["m00"] > 0:
        cx_m = moments["m10"] / moments["m00"]
        cy_m = moments["m01"] / moments["m00"]
    else:
        cx_m, cy_m = cx, cy

    confidence = circularity * 0.5 + area_ratio * 0.5

    return RecoveredPrimitive(
        type="circle",
        confidence=confidence,
        params={
            "center_x": float(cx),
            "center_y": float(cy),
            "radius": float(radius),
        },
    )


def recover_ellipse(contour: np.ndarray, area: float) -> Optional[RecoveredPrimitive]:
    """Recover ellipse parameters via direct ellipse fitting."""
    if contour is None or len(contour) < 5:
        return None

    try:
        ellipse = cv2.fitEllipse(contour)
    except cv2.error:
        return None

    center, axes, angle = ellipse
    a, b = axes[0] / 2, axes[1] / 2  # Semi-axes

    if a < 1 or b < 1:
        return None

    ellipse_area = np.pi * a * b
    area_ratio = min(area, ellipse_area) / max(area, ellipse_area, 1)

    confidence = area_ratio * 0.7 + 0.3  # Base confidence

    return RecoveredPrimitive(
        type="ellipse",
        confidence=confidence,
        params={
            "center_x": float(center[0]),
            "center_y": float(center[1]),
            "semi_major": float(max(a, b)),
            "semi_minor": float(min(a, b)),
            "rotation": float(angle),
        },
    )


def recover_polygon(contour: np.ndarray, epsilon_factor: float = 0.015) -> Optional[RecoveredPrimitive]:
    """Recover polygon: find the minimum vertices that describe the shape."""
    if contour is None or len(contour) < 3:
        return None

    perimeter = cv2.arcLength(contour, True)
    if perimeter < 10:
        return None

    # Progressive simplification: start coarse, refine until we get 3-12 vertices
    best_approx = None
    for factor in [0.04, 0.03, 0.025, 0.02, 0.015, 0.01, 0.008, 0.005]:
        epsilon = factor * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        nv = len(approx)
        if 3 <= nv <= 12:
            best_approx = approx
            break  # Take the coarsest simplification that gives 3-12 vertices

    # Fallback: take whatever we got with the coarsest epsilon
    if best_approx is None:
        epsilon = 0.04 * perimeter
        best_approx = cv2.approxPolyDP(contour, epsilon, True)

    if best_approx is None or len(best_approx) < 3:
        return None

    vertices = [(float(pt[0][0]), float(pt[0][1])) for pt in best_approx]
    nv = len(vertices)

    polygon_type = "polygon"
    if nv == 3:
        polygon_type = "triangle"
    elif nv == 4:
        polygon_type = "quadrilateral"
    elif nv == 5:
        polygon_type = "pentagon"
    elif nv == 6:
        polygon_type = "hexagon"
    elif nv == 8:
        polygon_type = "octagon"

    confidence = 0.6 + 0.2 * (1.0 / max(nv - 2, 1))  # Simpler = higher confidence

    return RecoveredPrimitive(
        type=polygon_type,
        confidence=confidence,
        params={
            "vertices": vertices,
            "num_sides": nv,
        },
    )


def recover_primitive(region: Dict[str, Any], image_area: float = 1e9) -> Optional[RecoveredPrimitive]:
    """Main entry point: recover the best geometric primitive for a region.

    Tries primitives in order of geometric complexity, from simplest
    (circle/rectangle) to most complex (polygon/bezier).

    Args:
        region: Region dict with contour, area, perimeter, etc.

    Returns:
        Best-matching RecoveredPrimitive.
    """
    contour = region.get("contour")
    area = float(region.get("area", 0))
    perimeter = float(region.get("perimeter", 0))
    circularity = float(region.get("circularity", 0))
    rectangularity = float(region.get("rectangularity", 0))

    candidates: List[RecoveredPrimitive] = []

    # Try circle
    if circularity > 0.80:
        circle = recover_circle(contour, area, perimeter)
        if circle:
            candidates.append(circle)

    # Try ellipse
    if 0.3 < circularity < 0.90:
        ellipse = recover_ellipse(contour, area)
        if ellipse:
            candidates.append(ellipse)

    # Try rectangle
    if rectangularity > 0.65:
        rect = recover_rectangle(contour, area)
        if rect:
            candidates.append(rect)
            # Also try rounded rectangle
            rr = recover_rounded_rectangle(contour, area, perimeter, image_area)
            if rr and rr.confidence > rect.confidence:
                candidates.append(rr)

    # Try polygon
    polygon = recover_polygon(contour)
    if polygon:
        candidates.append(polygon)

    if not candidates:
        return None

    # Return highest confidence
    return max(candidates, key=lambda c: c.confidence)
