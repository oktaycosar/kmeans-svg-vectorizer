"""
Region merger — combines fragmented regions into logical objects.

Many adjacent regions of similar color belong to the same object.
This module merges them based on color similarity, shared borders,
and shape continuity.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import List, Dict, Any, Tuple


def color_distance(c1: Tuple[int, ...], c2: Tuple[int, ...]) -> float:
    """Euclidean distance in RGB space."""
    return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))


def are_adjacent(mask1: np.ndarray, mask2: np.ndarray) -> bool:
    """Check if two binary masks share a border (are adjacent)."""
    dilated1 = cv2.dilate(mask1, np.ones((3, 3), np.uint8), iterations=1)
    overlap = cv2.bitwise_and(dilated1, mask2)
    return np.count_nonzero(overlap) > 0


def merge_regions(
    regions: List[Dict[str, Any]],
    color_threshold: float = 30.0,
    min_area: int = 100,
) -> List[Dict[str, Any]]:
    """Merge adjacent regions of similar color into unified objects.

    Args:
        regions: List of region dicts from region_segmenter.
        color_threshold: Max color distance to consider merging.
        min_area: Minimum merged region area.

    Returns:
        Merged region list.
    """
    n = len(regions)
    if n <= 1:
        return regions

    # Union-Find for merging
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # Compare all pairs for merging
    for i in range(n):
        for j in range(i + 1, n):
            ri, rj = regions[i], regions[j]

            # Check color similarity
            d = color_distance(ri["color_bgr"], rj["color_bgr"])
            if d > color_threshold:
                continue

            # Check adjacency
            if not are_adjacent(ri["mask"], rj["mask"]):
                continue

            union(i, j)

    # Group by root
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)

    # Merge each group
    merged = []
    for indices in groups.values():
        if len(indices) == 1:
            r = regions[indices[0]]
            if r["area"] >= min_area:
                merged.append(r)
            continue

        # Combine masks
        combined_mask = np.zeros_like(regions[0]["mask"])
        total_area = 0
        avg_color = np.zeros(3, dtype=np.float64)

        for idx in indices:
            r = regions[idx]
            combined_mask = cv2.bitwise_or(combined_mask, r["mask"])
            total_area += r["area"]
            avg_color += np.array(r["color_bgr"], dtype=np.float64) * r["area"]

        avg_color = tuple(int(v / total_area) for v in avg_color) if total_area > 0 else regions[indices[0]]["color_bgr"]

        # Compute new bbox from combined mask
        ys, xs = np.where(combined_mask > 0)
        if len(xs) > 0:
            x, y = xs.min(), ys.min()
            w, h = xs.max() - x + 1, ys.max() - y + 1
        else:
            x, y, w, h = 0, 0, 1, 1

        # New contour
        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        main_contour = max(contours, key=cv2.contourArea) if contours else None

        cx = x + w / 2
        cy = y + h / 2

        merged.append({
            "mask": combined_mask,
            "bbox": (int(x), int(y), int(w), int(h)),
            "area": int(total_area),
            "color_bgr": avg_color,
            "centroid": (cx, cy),
            "contour": main_contour,
            "pixel_count": int(total_area),
            "merged_from": len(indices),
        })

    merged.sort(key=lambda r: r["area"], reverse=True)
    return merged


def split_nested_regions(
    regions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Detect parent-child relationships: regions fully contained in others.

    Returns regions with 'parent' and 'children' indices added.
    """
    n = len(regions)
    children = {i: [] for i in range(n)}
    parents = {i: -1 for i in range(n)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Check if region j is fully inside region i
            overlap = cv2.bitwise_and(regions[j]["mask"], regions[i]["mask"])
            if np.count_nonzero(overlap) >= regions[j]["area"] * 0.95:
                children[i].append(j)
                parents[j] = i

    for i in range(n):
        regions[i]["children"] = children.get(i, [])
        regions[i]["parent"] = parents.get(i, -1)

    return regions
