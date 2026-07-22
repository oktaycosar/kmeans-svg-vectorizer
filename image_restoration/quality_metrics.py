"""
Vectorization-specific quality metrics.

Measures contour quality, not photo quality.
Evaluates how well the image will perform in OpenCV contour detection.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Dict, Any

from .config import VectorMetricsConfig
from .utils import logger


def _get_contours(gray: np.ndarray) -> list:
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return list(contours)


def count_contours(gray: np.ndarray) -> int:
    return len(_get_contours(gray))


def compute_closed_contour_ratio(gray: np.ndarray) -> float:
    contours = _get_contours(gray)
    if not contours: return 0.0
    closed = sum(1 for c in contours if cv2.contourArea(c) > 5)
    return closed / len(contours)


def compute_fragmentation_index(gray: np.ndarray) -> float:
    contours = _get_contours(gray)
    if not contours: return 1.0
    areas = [cv2.contourArea(c) for c in contours]
    if not areas: return 1.0
    small = sum(1 for a in areas if a < 30)
    return small / len(areas)


def compute_avg_contour_length(gray: np.ndarray) -> float:
    contours = _get_contours(gray)
    if not contours: return 0.0
    perimeters = [cv2.arcLength(c, True) for c in contours]
    return float(np.mean(perimeters)) if perimeters else 0.0


def count_noise_contours(gray: np.ndarray, min_area: int = 20) -> int:
    contours = _get_contours(gray)
    return sum(1 for c in contours if cv2.contourArea(c) < min_area)


def count_color_clusters(image: np.ndarray) -> int:
    if image.ndim != 3: return 1
    q = image // 32 * 32
    combined = (q[:,:,0].astype(np.int32) * 10000 +
                q[:,:,1].astype(np.int32) * 100 +
                q[:,:,2].astype(np.int32))
    return len(np.unique(combined))


def compute_edge_continuity(gray: np.ndarray) -> float:
    edges = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    if np.count_nonzero(edges) == 0: return 0.0
    return 1.0 - (np.count_nonzero(closed) - np.count_nonzero(edges)) / max(np.count_nonzero(edges), 1)


def estimate_primitive_count(gray: np.ndarray) -> dict:
    contours = _get_contours(gray)
    rects = circles = polygons = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 30: continue
        perimeter = cv2.arcLength(c, True)
        if perimeter < 1: continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        nv = len(approx)
        if circularity > 0.85: circles += 1
        elif nv == 4 and area > 50: rects += 1
        elif 3 <= nv <= 12: polygons += 1
    return {"estimated_rectangles": rects, "estimated_circles": circles, "estimated_polygons": polygons}


def compute_vectorization_score(gray: np.ndarray, image: np.ndarray) -> float:
    contours = _get_contours(gray)
    nc = len(contours)
    closed_r = compute_closed_contour_ratio(gray)
    frag = compute_fragmentation_index(gray)
    colors = count_color_clusters(image)
    noise_c = count_noise_contours(gray)
    continuity = compute_edge_continuity(gray)
    primitives = estimate_primitive_count(gray)
    total_prims = sum(primitives.values())
    color_score = max(0, 1.0 - colors / 64.0)
    closed_score = closed_r
    frag_score = 1.0 - frag
    noise_score = max(0, 1.0 - noise_c / max(nc, 1))
    continuity_score = continuity
    prim_score = min(1.0, total_prims / max(nc * 0.5, 1))
    score = (color_score * 0.25 + closed_score * 0.20 +
             frag_score * 0.20 + noise_score * 0.15 +
             continuity_score * 0.10 + prim_score * 0.10) * 100.0
    return round(min(100.0, max(0.0, score)), 1)


def compute_all_vector_metrics(image: np.ndarray, config: VectorMetricsConfig) -> Dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    metrics = {}
    if config.compute_contour_count: metrics["contour_count"] = count_contours(gray)
    if config.compute_closed_ratio: metrics["closed_contour_ratio"] = round(compute_closed_contour_ratio(gray), 4)
    if config.compute_fragmentation: metrics["fragmentation_index"] = round(compute_fragmentation_index(gray), 4)
    if config.compute_avg_contour_length: metrics["avg_contour_length"] = round(compute_avg_contour_length(gray), 1)
    if config.compute_noise_contours: metrics["noise_contour_count"] = count_noise_contours(gray)
    if config.compute_color_clusters: metrics["color_clusters"] = count_color_clusters(image)
    if config.compute_edge_continuity: metrics["edge_continuity"] = round(compute_edge_continuity(gray), 4)
    if config.compute_estimated_primitives: metrics["estimated_primitives"] = estimate_primitive_count(gray)
    metrics["vectorization_score"] = compute_vectorization_score(gray, image)
    return metrics


def compare_vector_metrics(before: dict, after: dict) -> dict:
    comp = {"before": before, "after": after, "improvement": {}}
    for key in set(before) | set(after):
        bv = before.get(key); av = after.get(key)
        if isinstance(bv, (int, float)) and isinstance(av, (int, float)):
            comp["improvement"][key] = round(av - bv, 2)
    return comp
