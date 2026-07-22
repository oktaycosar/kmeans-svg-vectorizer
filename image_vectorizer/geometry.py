"""
Geometric utility functions for the ImageVectorizer.

Provides low-level geometry helpers: line merging, vertex ordering,
corner detection, angle computation, and shape fitting.
"""

from __future__ import annotations

import math
from typing import List, Tuple, Optional

import numpy as np
import cv2

from .models import Point
from .utils import clamp, normalize_angle_degrees, distance_2d


def merge_nearby_lines(
    lines: np.ndarray,
    angle_threshold_deg: float = 5.0,
    distance_threshold: float = 10.0,
) -> np.ndarray:
    """Merge collinear and nearby line segments detected by HoughLinesP.

    Lines are given as N×1×4 array (x1, y1, x2, y2).
    Returns merged lines in the same format.

    Args:
        lines: Input lines from HoughLinesP.
        angle_threshold_deg: Max angular difference to consider lines collinear.
        distance_threshold: Max perpendicular distance to merge.

    Returns:
        Merged lines array.
    """
    if lines is None or len(lines) == 0:
        return np.array([])

    # Flatten to (N, 4)
    if lines.ndim == 3:
        segments = lines[:, 0, :]  # (N, 4)
    else:
        segments = lines

    if len(segments) <= 1:
        return lines

    n = len(segments)
    used = np.zeros(n, dtype=bool)
    merged: List[List[float]] = []

    for i in range(n):
        if used[i]:
            continue

        used[i] = True
        x1, y1, x2, y2 = segments[i]
        group = [(x1, y1, x2, y2)]

        # Find all lines collinear with this one
        for j in range(i + 1, n):
            if used[j]:
                continue

            x1j, y1j, x2j, y2j = segments[j]

            # Check angular similarity
            angle_i = math.degrees(math.atan2(y2 - y1, x2 - x1))
            angle_j = math.degrees(math.atan2(y2j - y1j, x2j - x1j))
            angle_diff = abs(normalize_angle_degrees(angle_i) - normalize_angle_degrees(angle_j))
            angle_diff = min(angle_diff, 180.0 - angle_diff)

            if angle_diff > angle_threshold_deg:
                continue

            # Check distance: compute minimum distance between endpoints
            d1 = _point_to_line_distance((x1j, y1j), (x1, y1), (x2, y2))
            d2 = _point_to_line_distance((x2j, y2j), (x1, y1), (x2, y2))
            if min(d1, d2) <= distance_threshold:
                used[j] = True
                group.append((x1j, y1j, x2j, y2j))

        # Merge group: find farthest endpoints along the line direction
        if len(group) > 1:
            merged_line = _merge_line_group(group)
            merged.append(merged_line)
        else:
            merged.append([x1, y1, x2, y2])

    result = np.array(merged, dtype=np.int32).reshape(-1, 1, 4)
    return result


def _point_to_line_distance(
    point: Tuple[float, float],
    line_start: Tuple[float, float],
    line_end: Tuple[float, float],
) -> float:
    """Perpendicular distance from a point to an infinite line."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)

    if denom < 1e-9:
        return distance_2d(point, line_start)

    # Cross product magnitude / line length
    return abs(dx * (y1 - py) - (x1 - px) * dy) / denom


def _merge_line_group(
    group: List[Tuple[float, float, float, float]],
) -> List[float]:
    """Merge a group of collinear line segments into a single segment.

    Projects all endpoints onto the best-fit line and takes the farthest two.

    Args:
        group: List of (x1, y1, x2, y2) tuples.

    Returns:
        [x1, y1, x2, y2] for the merged line.
    """
    all_points = []
    for x1, y1, x2, y2 in group:
        all_points.append((x1, y1))
        all_points.append((x2, y2))

    pts = np.array(all_points, dtype=np.float32)

    # Fit line using PCA
    mean = pts.mean(axis=0)
    centered = pts - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    direction = eigenvectors[:, np.argmax(eigenvalues)]

    # Project all points onto the direction vector
    projections = np.dot(centered, direction)
    t_min = projections.min()
    t_max = projections.max()

    start = mean + direction * t_min
    end = mean + direction * t_max

    return [float(start[0]), float(start[1]), float(end[0]), float(end[1])]


def order_vertices_clockwise(vertices: List[Point]) -> List[Point]:
    """Order polygon vertices in clockwise order around their centroid.

    Args:
        vertices: Unordered list of Points.

    Returns:
        Clockwise-ordered list of Points.
    """
    if len(vertices) < 3:
        return vertices

    cx = sum(v.x for v in vertices) / len(vertices)
    cy = sum(v.y for v in vertices) / len(vertices)

    def angle_from_center(v: Point) -> float:
        return math.atan2(v.y - cy, v.x - cx)

    return sorted(vertices, key=angle_from_center, reverse=True)


def compute_polygon_angles(vertices: List[Point]) -> List[float]:
    """Compute interior angles (in degrees) at each vertex of a convex polygon.

    Uses the dot product of adjacent edge vectors.

    Args:
        vertices: Ordered polygon vertices.

    Returns:
        List of interior angles in degrees.
    """
    n = len(vertices)
    if n < 3:
        return []

    angles = []
    for i in range(n):
        a = vertices[(i - 1) % n]
        b = vertices[i]
        c = vertices[(i + 1) % n]

        v1 = (a.x - b.x, a.y - b.y)
        v2 = (c.x - b.x, c.y - b.y)

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)

        if mag1 < 1e-9 or mag2 < 1e-9:
            angles.append(0.0)
            continue

        cos_angle = clamp(dot / (mag1 * mag2), -1.0, 1.0)
        angle = math.degrees(math.acos(cos_angle))
        angles.append(angle)

    return angles


def detect_corner_radius(
    approx_contour: np.ndarray,
    original_contour: np.ndarray,
) -> float:
    """Estimate the corner radius of a rounded rectangle.

    For each approximate vertex, measure the max distance from original
    contour points in the neighborhood. Sharp corners (where contour
    points lie on the vertex) return ~0.

    Args:
        approx_contour: Approximated polygon (e.g., 4 points for a rectangle).
        original_contour: Full original contour.

    Returns:
        Estimated corner radius in pixels (0 if sharp corners).
    """
    if len(approx_contour) < 4:
        return 0.0

    # Get the four vertices
    approx_pts = approx_contour.reshape(-1, 2).astype(np.float32)
    if len(approx_pts) != 4:
        return 0.0

    orig_pts = original_contour.reshape(-1, 2).astype(np.float32)
    n_orig = len(orig_pts)

    # If the contour has very few points (same as approximation),
    # corners must be sharp (no intermediate rounded points exist)
    if n_orig <= len(approx_pts) + 4:
        return 0.0

    # For each vertex, find the max deviation of original contour points
    # within a window around the vertex
    corner_deviations = []
    for vertex in approx_pts:
        # Find closest point in original contour
        distances = np.linalg.norm(orig_pts - vertex, axis=1)
        nearest_idx = np.argmin(distances)

        # Check if the closest point is very near the vertex (sharp corner)
        if distances[nearest_idx] < 1.0:
            corner_deviations.append(0.0)
            continue

        # Sample a window around the nearest point
        window = min(15, n_orig // 4)
        window_indices = [
            (nearest_idx + offset) % n_orig
            for offset in range(-window, window + 1)
        ]

        avg_deviation = 0.0
        count = 0
        for idx in window_indices:
            point = orig_pts[idx]
            d = np.linalg.norm(point - vertex)
            avg_deviation += d
            count += 1

        if count > 0:
            corner_deviations.append(avg_deviation / count)

    if not corner_deviations:
        return 0.0

    avg_corner_dev = np.mean(corner_deviations)
    return float(avg_corner_dev)


def fit_line_to_points(points: List[Point]) -> Tuple[Point, Point, float]:
    """Fit a straight line to a list of points using linear regression.

    Args:
        points: List of Points.

    Returns:
        Tuple of (start_point, end_point, r_squared).
    """
    if len(points) < 2:
        raise ValueError("Need at least 2 points to fit a line.")

    xs = np.array([p.x for p in points])
    ys = np.array([p.y for p in points])

    # Linear regression: y = mx + b
    n = len(xs)
    if n < 2:
        return (Point(0, 0), Point(0, 0), 0.0)

    mean_x = xs.mean()
    mean_y = ys.mean()

    ss_xx = np.sum((xs - mean_x) ** 2)
    ss_yy = np.sum((ys - mean_y) ** 2)
    ss_xy = np.sum((xs - mean_x) * (ys - mean_y))

    if ss_xx < 1e-9:
        # Vertical line
        x_const = mean_x
        start = Point(x_const, ys.min())
        end = Point(x_const, ys.max())
        return (start, end, 1.0)

    slope = ss_xy / ss_xx
    intercept = mean_y - slope * mean_x

    # Project endpoints
    # Find the range of x that contains the points
    x_min, x_max = xs.min(), xs.max()
    y_at_min = slope * x_min + intercept
    y_at_max = slope * x_max + intercept

    start = Point(float(x_min), float(y_at_min))
    end = Point(float(x_max), float(y_at_max))

    # R-squared
    y_pred = slope * xs + intercept
    ss_res = np.sum((ys - y_pred) ** 2)
    r_squared = 1.0 - (ss_res / ss_yy) if ss_yy > 1e-9 else 1.0

    return (start, end, float(r_squared))


def fit_ellipse_to_contour(contour: np.ndarray) -> Optional[Tuple[Tuple[float, float], Tuple[float, float], float]]:
    """Fit a rotated ellipse to a contour.

    Args:
        contour: OpenCV contour array (at least 5 points).

    Returns:
        Tuple of (center (x,y), axes (a,b), angle_degrees), or None.
    """
    if len(contour) < 5:
        return None

    try:
        ellipse = cv2.fitEllipse(contour)
        return ellipse
    except cv2.error:
        return None


def simplify_contour(
    contour: np.ndarray,
    epsilon_factor: float = 0.02,
) -> np.ndarray:
    """Simplify a contour using Douglas-Peucker (approxPolyDP).

    Args:
        contour: OpenCV contour array.
        epsilon_factor: Fraction of perimeter to use as epsilon.

    Returns:
        Simplified contour.
    """
    perimeter = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * perimeter
    return cv2.approxPolyDP(contour, epsilon, True)


def is_convex_contour(contour: np.ndarray) -> bool:
    """Check if a contour is convex.

    Args:
        contour: OpenCV contour array.

    Returns:
        True if convex.
    """
    return bool(cv2.isContourConvex(contour))


def compute_defect_depth(contour: np.ndarray) -> float:
    """Compute the maximum convexity defect depth.

    Useful for detecting shapes with indentations vs convex shapes.

    Args:
        contour: OpenCV contour array.

    Returns:
        Maximum defect depth, or 0 if fails.
    """
    hull = cv2.convexHull(contour, returnPoints=False)
    if hull is None or len(hull) < 3:
        return 0.0

    try:
        defects = cv2.convexityDefects(contour, hull)
        if defects is None:
            return 0.0
        # defects: (N, 1, 4) -> [start, end, farthest, depth]
        depths = defects[:, 0, 3]
        max_depth = depths.max() / 256.0  # OpenCV stores depth in 8-bit fixed-point
        return float(max_depth)
    except cv2.error:
        return 0.0
