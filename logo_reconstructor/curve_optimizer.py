"""
Curve optimizer — fits smooth Bezier curves with minimal control points.

Goal: represent shapes with as few nodes as possible while preserving
corners, tangents, and overall geometry.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Tuple, Optional


def fit_bezier_curve(
    points: np.ndarray,
    max_error: float = 2.0,
    corner_threshold_deg: float = 30.0,
) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Fit cubic Bezier curves to a polyline, minimizing control points.

    Uses a simplified Schneider algorithm: iteratively subdivide at
    points of maximum error, then fit cubic Beziers.

    Args:
        points: Array of (N, 2) contour points.
        max_error: Maximum allowed deviation in pixels.
        corner_threshold_deg: Angle threshold to preserve corners.

    Returns:
        List of SVG path commands: ('M', [...]), ('L', [...]), ('C', [...])
    """
    if len(points) < 2:
        return []

    # Simplify using Douglas-Peucker first
    pts = _simplify_points(points)

    # Detect corners
    corners = _detect_corners(pts, corner_threshold_deg)

    # Build path commands
    commands: List[Tuple[str, List[Tuple[float, float]]]] = []

    # Move to first point
    commands.append(("M", [(float(pts[0][0]), float(pts[0][1]))]))

    segment_start = 0
    for i in range(1, len(pts)):
        is_corner = i in corners
        is_last = (i == len(pts) - 1)

        if is_corner or is_last:
            segment = pts[segment_start:i + 1]

            if len(segment) <= 2:
                # Straight line
                for pt in segment[1:]:
                    commands.append(("L", [(float(pt[0]), float(pt[1]))]))
            elif len(segment) <= 4:
                # Short segment: use lines
                for pt in segment[1:]:
                    commands.append(("L", [(float(pt[0]), float(pt[1]))]))
            else:
                # Fit Bezier
                bezier_cmds = _fit_cubic_bezier_segment(segment, max_error)
                commands.extend(bezier_cmds)

            segment_start = i

    # Close path if start == end
    if len(pts) > 2:
        d = np.hypot(pts[0][0] - pts[-1][0], pts[0][1] - pts[-1][1])
        if d < 3:
            commands.append(("Z", []))

    return commands


def _simplify_points(points: np.ndarray, epsilon: float = 1.5) -> np.ndarray:
    """Simplify polyline using Douglas-Peucker."""
    pts = points.reshape(-1, 2).astype(np.float32)
    # Ensure it's in contour format for approxPolyDP
    contour = pts.reshape(-1, 1, 2)
    perimeter = cv2.arcLength(contour, False)
    eps = epsilon * max(perimeter * 0.001, 0.5)
    approx = cv2.approxPolyDP(contour, eps, False)
    return approx.reshape(-1, 2)


def _detect_corners(pts: np.ndarray, angle_threshold: float) -> set:
    """Detect corner points where direction changes sharply."""
    corners = set()
    n = len(pts)
    if n < 3:
        return corners

    for i in range(n):
        prev_pt = pts[(i - 1) % n]
        curr_pt = pts[i]
        next_pt = pts[(i + 1) % n]

        v1 = prev_pt - curr_pt
        v2 = next_pt - curr_pt

        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 < 1e-6 or mag2 < 1e-6:
            continue

        cos_angle = np.dot(v1, v2) / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        angle_deg = np.degrees(np.arccos(cos_angle))

        # Sharp turn = corner
        if abs(180.0 - angle_deg) > angle_threshold:
            corners.add(i)

    return corners


def _fit_cubic_bezier_segment(
    pts: np.ndarray, max_error: float
) -> List[Tuple[str, List[Tuple[float, float]]]]:
    """Fit cubic Bezier(s) to a segment of points.

    Uses a simple least-squares approach for the control points.
    Falls back to line segments if fit error is too high.
    """
    n = len(pts)
    if n < 3:
        return [("L", [(float(pts[-1][0]), float(pts[-1][1]))])]

    p0 = pts[0]
    p3 = pts[-1]

    # Fit control points using chord-length parameterization
    chords = np.zeros(n)
    for i in range(1, n):
        chords[i] = chords[i - 1] + np.linalg.norm(pts[i] - pts[i - 1])

    if chords[-1] < 1e-6:
        return [("L", [(float(p3[0]), float(p3[1]))])]

    t = chords / chords[-1]  # Normalized [0, 1]

    # Bezier basis matrix for cubic
    # P(t) = (1-t)^3*P0 + 3(1-t)^2*t*P1 + 3(1-t)*t^2*P2 + t^3*P3
    # We know P0 and P3, solve for P1 and P2 using least squares

    A = np.zeros((n, 2))
    b = np.zeros((n, 2))

    for i in range(n):
        ti = t[i]
        A[i, 0] = 3 * (1 - ti) ** 2 * ti
        A[i, 1] = 3 * (1 - ti) * ti ** 2
        b[i] = pts[i] - (1 - ti) ** 3 * p0 - ti ** 3 * p3

    # Solve least squares
    try:
        x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        p1 = x[0]
        p2 = x[1]
    except np.linalg.LinAlgError:
        return [("L", [(float(p3[0]), float(p3[1]))])]

    # Check fit error
    fitted = np.zeros_like(pts)
    for i in range(n):
        ti = t[i]
        fitted[i] = ((1 - ti) ** 3 * p0 + 3 * (1 - ti) ** 2 * ti * p1 +
                      3 * (1 - ti) * ti ** 2 * p2 + ti ** 3 * p3)

    max_dev = np.max(np.linalg.norm(pts - fitted, axis=1))

    if max_dev > max_error * 3:
        # Split and retry
        mid = n // 2
        cmds1 = _fit_cubic_bezier_segment(pts[:mid + 1], max_error)
        cmds2 = _fit_cubic_bezier_segment(pts[mid:], max_error)
        return cmds1 + cmds2

    return [("C", [
        (float(p1[0]), float(p1[1])),
        (float(p2[0]), float(p2[1])),
        (float(p3[0]), float(p3[1])),
    ])]


def optimize_curve(contour: np.ndarray) -> str:
    """Generate optimized SVG path data from a contour.

    Args:
        contour: OpenCV contour array.

    Returns:
        SVG path 'd' attribute string.
    """
    if contour is None or len(contour) < 2:
        return ""

    points = contour.reshape(-1, 2).astype(np.float64)
    commands = fit_bezier_curve(points)

    parts = []
    for cmd, coords in commands:
        if cmd == "M":
            parts.append(f"M {coords[0][0]:.1f} {coords[0][1]:.1f}")
        elif cmd == "L":
            parts.append(f"L {coords[0][0]:.1f} {coords[0][1]:.1f}")
        elif cmd == "C":
            parts.append(
                f"C {coords[0][0]:.1f} {coords[0][1]:.1f} "
                f"{coords[1][0]:.1f} {coords[1][1]:.1f} "
                f"{coords[2][0]:.1f} {coords[2][1]:.1f}"
            )
        elif cmd == "Z":
            parts.append("Z")

    return " ".join(parts)
