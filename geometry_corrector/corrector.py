"""
Geometry Corrector — Stage 2 AI-assisted geometry fixing.

Philosophy: AI DETECTS, geometry FIXES. Never generate coordinates from AI.
Only deterministic math applies corrections.
"""
import math
import numpy as np
from typing import List, Tuple, Optional

# ── Angle utilities ──────────────────────────────────────────────

def angle_between(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """Angle at p2 between p1-p2-p3, in degrees [0, 180]."""
    v1 = p1 - p2
    v2 = p3 - p2
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-9:
        return 180.0
    cos_a = np.clip(dot / norm, -1.0, 1.0)
    return math.degrees(math.acos(cos_a))


def classify_angle(angle_deg: float, tolerance: float = 3.0) -> Optional[str]:
    """Classify an angle as 'right' (≈90°), 'straight' (≈180°), 'acute' (≈45°),
    'diagonal' (≈135°), or None if no clear classification."""
    a = angle_deg % 180
    if a > 180 - tolerance or a < tolerance:
        return "straight"
    if abs(a - 90) <= tolerance:
        return "right"
    if abs(a - 45) <= tolerance:
        return "acute"
    if abs(a - 135) <= tolerance:
        return "diagonal"
    return None


# ── Right-angle snapping ─────────────────────────────────────────

def snap_right_angle(pts: np.ndarray, tolerance: float = 3.0) -> np.ndarray:
    """
    Walk through polygon vertices, find near-90° corners,
    and snap them to exactly 90°.
    
    Geometric principle: For a right angle at B in A-B-C,
    B must lie on the circle with diameter AC (Thales' theorem).
    We project B onto that circle → guaranteed 90° at B.
    
    Returns corrected pts (same shape).
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    if n < 3:
        return pts

    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]

        ang = angle_between(a, b, c)
        if abs(ang - 90.0) > tolerance:
            continue  # Not near 90°

        # Thales circle: center = midpoint of AC, radius = |AC|/2
        mid = (a + c) / 2.0
        ac_half = np.linalg.norm(a - c) / 2.0
        
        if ac_half < 1e-6:
            continue
        
        # Project B onto the circle: B' = mid + ac_half * (B - mid) / |B - mid|
        bm = b - mid
        bm_norm = np.linalg.norm(bm)
        if bm_norm < 1e-6:
            continue
        
        b_corrected = mid + ac_half * bm / bm_norm
        
        # Safety: don't move more than 15% of edge length
        max_move = min(np.linalg.norm(a - b), np.linalg.norm(c - b)) * 0.15
        if np.linalg.norm(b_corrected - b) > max_move:
            # Snap to nearest point on circle within max_move
            direction = b_corrected - b
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-6:
                b_corrected = b + direction / dir_norm * max_move
        
        pts[i] = b_corrected

    return pts


# ── Collinearity straightening ────────────────────────────────────

def straighten_collinear(pts: np.ndarray, tolerance: float = 2.0) -> np.ndarray:
    """
    Find sequences of near-collinear points and snap them to a straight line.
    Uses linear regression on each candidate segment.
    """
    pts = pts.astype(np.float64).copy()
    n = len(pts)
    if n < 3:
        return pts

    i = 0
    while i < n:
        # Find a run of near-straight angles
        run_start = i
        j = i
        while j < n:
            a = pts[j % n]
            b = pts[(j + 1) % n]
            c = pts[(j + 2) % n]
            ang = angle_between(a, b, c)
            if classify_angle(ang, tolerance) == "straight":
                j += 1
            else:
                break

        run_len = (j - run_start + 1)
        if run_len >= 3:
            # At least 3 collinear points → straighten
            indices = [(run_start + k) % n for k in range(run_len + 1)]
            run_pts = pts[indices]
            
            # Linear regression
            x = run_pts[:, 0]
            y = run_pts[:, 1]
            A = np.vstack([x, np.ones_like(x)]).T
            m, b_line = np.linalg.lstsq(A, y, rcond=None)[0]
            
            # Project each point onto the line
            for idx in indices[:-1]:  # Keep endpoints free
                x0 = pts[idx, 0]
                # Line: y = m*x + b, projection:
                # closest point on line: (x0 + m*(y0 - m*x0 - b)/(1+m²), m*that + b)
                y0 = pts[idx, 1]
                t = (x0 + m * (y0 - b_line)) / (1 + m * m)
                proj_y = m * t + b_line
                pts[idx] = np.array([t, proj_y])

        i = j + 1 if j > i else i + 1

    return pts


# ── Main correction pipeline ──────────────────────────────────────

def correct_path(pts: np.ndarray, 
                 right_angle_tol: float = 10.0,
                 collinear_tol: float = 5.0) -> np.ndarray:
    """
    Apply all geometry corrections to a single path's vertices.
    
    Order matters:
    1. Right-angle snapping (most impactful)
    2. Collinearity straightening (secondary)
    
    Returns corrected vertex array.
    """
    original = pts.copy()
    
    # Pass 1: right angles
    pts = snap_right_angle(pts, tolerance=right_angle_tol)
    
    # Pass 2: straight lines  
    pts = straighten_collinear(pts, tolerance=collinear_tol)
    
    # Safety: ensure no point moved more than 5% of bounding box
    diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
    max_move = diag * 0.05
    for i in range(len(pts)):
        if np.linalg.norm(pts[i] - original[i]) > max_move:
            pts[i] = original[i]  # Revert
    
    return pts


def correct_svg_paths(paths_data: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
    """
    Correct all SVG paths. Input: [(area, svg_string), ...]
    Output: [(area, corrected_svg_string), ...]
    """
    corrected = []
    for area, path_str in paths_data:
        # Extract the d="..." part
        if 'd="' not in path_str:
            corrected.append((area, path_str))
            continue
        
        # Parse path data
        d_start = path_str.index('d="') + 3
        d_end = path_str.index('"', d_start)
        d_original = path_str[d_start:d_end]
        
        # Simple parser for "M x y L x y ... Z" format
        tokens = d_original.replace(',', ' ').split()
        if not tokens or tokens[0] != 'M':
            corrected.append((area, path_str))
            continue
        
        coords = []
        i = 1
        while i < len(tokens) - 1:
            if tokens[i] == 'L':
                i += 1
                continue
            if tokens[i] == 'Z':
                break
            try:
                x = float(tokens[i])
                y = float(tokens[i + 1])
                coords.append([x, y])
                i += 2
            except ValueError:
                i += 1
        
        if len(coords) < 3:
            corrected.append((area, path_str))
            continue
        
        pts = np.array(coords)
        corrected_pts = correct_path(pts)
        
        # Rebuild path string
        d_corrected = "M " + " L ".join(f"{p[0]:.2f} {p[1]:.2f}" for p in corrected_pts) + " Z"
        
        new_path_str = path_str[:d_start] + d_corrected + path_str[d_end:]
        corrected.append((area, new_path_str))
    
    return corrected
