"""
AI Shape Detector — "haaaaa bu köşe 90° olmalı!"

Detects geometric features in polygon paths WITHOUT generating coordinates.
Returns DETECTION RESULTS only — the corrector module applies the math.
"""
import math
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field


@dataclass
class CornerDetection:
    """AI says: 'This corner should be a right angle!'"""
    vertex_index: int
    current_angle: float
    target_angle: float  # 90, 180, 45, 135
    confidence: float    # 0-1
    reasoning: str

@dataclass 
class LineDetection:
    """AI says: 'These points should be collinear!'"""
    start_index: int
    end_index: int
    point_count: int
    max_deviation: float  # max distance from best-fit line
    confidence: float
    reasoning: str

@dataclass
class CurveDetection:
    """AI says: 'This segment looks like a circle arc!'"""
    start_index: int
    end_index: int
    radius: float
    center: Tuple[float, float]
    angle_span: float  # degrees
    fit_error: float
    confidence: float
    reasoning: str

@dataclass
class ShapeAnalysis:
    """Complete analysis of a polygon path."""
    vertex_count: int
    corner_detections: List[CornerDetection] = field(default_factory=list)
    line_detections: List[LineDetection] = field(default_factory=list)
    curve_detections: List[CurveDetection] = field(default_factory=list)
    is_axis_aligned: bool = False
    symmetry_score: float = 0.0
    likely_shape: str = "polygon"  # polygon, rectangle, circle, triangle


def angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle ABC in degrees."""
    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    norm = np.linalg.norm(ba) * np.linalg.norm(bc)
    if norm < 1e-9:
        return 180.0
    return float(np.degrees(np.arccos(np.clip(dot / norm, -1.0, 1.0))))


def detect_right_angles(pts: np.ndarray, tolerance: float = 8.0) -> List[CornerDetection]:
    """
    AI walks around the polygon, checks each corner:
    'Is this ≈90°? If yes, it SHOULD be 90° — snap it!'
    
    Higher tolerance catches more candidates.
    """
    detections = []
    n = len(pts)
    
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        
        ang = angle_between(a, b, c)
        deviation = abs(ang - 90.0)
        
        if deviation > tolerance:
            continue
        
        # Confidence: closer to 90° = higher confidence
        confidence = max(0.0, 1.0 - deviation / tolerance)
        
        # Build reasoning
        if deviation < 1.0:
            reasoning = f"Köşe {i}: {ang:.1f}° → zaten neredeyse 90°, eminim"
        elif deviation < 3.0:
            reasoning = f"Köşe {i}: {ang:.1f}° → hafif sapmış, 90° olmalı"
        else:
            reasoning = f"Köşe {i}: {ang:.1f}° → bayağı kaymış ama dik açı niyeti belli"
        
        detections.append(CornerDetection(
            vertex_index=i,
            current_angle=ang,
            target_angle=90.0,
            confidence=confidence,
            reasoning=reasoning
        ))
    
    return detections


def detect_straight_angles(pts: np.ndarray, tolerance: float = 5.0) -> List[CornerDetection]:
    """Detect near-180° angles that should be perfectly straight."""
    detections = []
    n = len(pts)
    
    for i in range(n):
        a = pts[(i - 1) % n]
        b = pts[i]
        c = pts[(i + 1) % n]
        
        ang = angle_between(a, b, c)
        deviation = min(abs(ang - 180.0), abs(ang - 0.0))
        
        if deviation > tolerance:
            continue
        
        confidence = max(0.0, 1.0 - deviation / tolerance)
        
        detections.append(CornerDetection(
            vertex_index=i,
            current_angle=ang,
            target_angle=180.0,
            confidence=confidence,
            reasoning=f"Nokta {i}: {ang:.1f}° → düz olmalı, aradan çıksın"
        ))
    
    return detections


def detect_collinear_runs(pts: np.ndarray, angle_tolerance: float = 3.0,
                          point_tolerance: float = 2.0) -> List[LineDetection]:
    """
    Find sequences of 3+ points that lie on (nearly) the same line.
    These should be simplified to just 2 endpoints.
    """
    n = len(pts)
    if n < 4:
        return []
    
    detections = []
    i = 0
    while i < n:
        # Find run of collinear points
        run_start = i
        j = i
        while j < n:
            a = pts[j % n]
            b = pts[(j + 1) % n]
            c = pts[(j + 2) % n]
            ang = angle_between(a, b, c)
            if abs(ang - 180.0) <= angle_tolerance or abs(ang) <= angle_tolerance:
                j += 1
            else:
                break
        
        run_len = j - run_start + 1
        if run_len >= 3:
            run_pts = pts[[(run_start + k) % n for k in range(run_len + 1)]]
            # Check max deviation from ideal line
            x, y = run_pts[:, 0], run_pts[:, 1]
            A = np.vstack([x, np.ones_like(x)]).T
            m_slope, b_line = np.linalg.lstsq(A, y, rcond=None)[0]
            deviations = np.abs(y - (m_slope * x + b_line))
            max_dev = float(np.max(deviations))
            
            if max_dev <= point_tolerance:
                confidence = 1.0 - max_dev / point_tolerance
                detections.append(LineDetection(
                    start_index=run_start % n,
                    end_index=(run_start + run_len) % n,
                    point_count=run_len + 1,
                    max_deviation=max_dev,
                    confidence=confidence,
                    reasoning=f"{run_len+1} nokta dümdüz — {run_len-1} gereksiz, silinebilir"
                ))
        
        i = j + 1 if j > i else i + 1
    
    return detections


def detect_shape_type(pts: np.ndarray) -> str:
    """Classify the overall shape of a polygon."""
    n = len(pts)
    if n == 3:
        return "triangle"
    if n == 4:
        # Check if it's a rectangle
        angles = []
        for i in range(4):
            a = pts[(i - 1) % 4]
            b = pts[i]
            c = pts[(i + 1) % 4]
            angles.append(angle_between(a, b, c))
        
        if all(abs(a - 90) < 5 for a in angles):
            return "rectangle"
        return "quadrilateral"
    
    # Check for circle-like
    center = pts.mean(axis=0)
    distances = np.linalg.norm(pts - center, axis=1)
    if distances.std() / distances.mean() < 0.05 and n >= 12:
        return "circle"
    
    return "polygon"


def analyze_path(pts: np.ndarray) -> ShapeAnalysis:
    """
    Full AI analysis of a polygon path.
    Returns detections only — NO coordinate modifications.
    """
    analysis = ShapeAnalysis(vertex_count=len(pts))
    
    # Detect corners
    analysis.corner_detections = detect_right_angles(pts)
    analysis.corner_detections += detect_straight_angles(pts)
    
    # Detect collinear runs
    analysis.line_detections = detect_collinear_runs(pts)
    
    # Classify shape
    analysis.likely_shape = detect_shape_type(pts)
    
    # Check axis alignment
    if len(pts) >= 4:
        # Are most edges horizontal/vertical?
        aligned = 0
        for i in range(len(pts)):
            edge = pts[(i + 1) % len(pts)] - pts[i]
            if np.linalg.norm(edge) < 1e-6:
                continue
            angle = abs(np.degrees(np.arctan2(edge[1], edge[0]))) % 90
            if angle < 3 or angle > 87:
                aligned += 1
        analysis.is_axis_aligned = aligned >= len(pts) * 0.7
    
    return analysis


def ai_verdict(analysis: ShapeAnalysis) -> str:
    """
    AI's overall verdict on the path quality.
    Turkish, friendly, like a smart colleague.
    """
    parts = []
    
    if analysis.likely_shape == "rectangle":
        right_count = sum(1 for d in analysis.corner_detections if d.target_angle == 90.0)
        if right_count == 4:
            parts.append("👍 Mükemmel dikdörtgen — köşeler zaten 90°")
        elif right_count >= 2:
            parts.append(f"🔧 {right_count}/4 köşe dik açı — diğerlerini düzelteyim")
        else:
            parts.append("🤔 Dikdörtgen niyeti var ama köşeler bozuk")
    
    elif analysis.likely_shape == "circle":
        parts.append("⭕ Bu bir çember/yay — circle fitting yapılabilir")
    
    if analysis.line_detections:
        total_saved = sum(d.point_count - 2 for d in analysis.line_detections)
        parts.append(f"📏 {total_saved} gereksiz nokta silinebilir")
    
    if analysis.is_axis_aligned:
        parts.append("📐 Eksenlere hizalı — güzel!")
    
    if not parts:
        parts.append("✅ Bu path'te düzeltecek bir şey yok")
    
    return " | ".join(parts)
