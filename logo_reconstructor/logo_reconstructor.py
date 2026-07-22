"""
Logo Reconstructor — main orchestrator.

Pipeline:
1. Segment into color regions
2. Merge adjacent similar regions
3. Build object graph
4. Recover geometric primitives
5. Optimize curves
6. Generate clean SVG (one object = one element)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np

from image_restoration.loader import load_image, split_alpha, resize_if_large, save_image
from image_restoration.utils import logger, ensure_output_dir, get_file_stem, save_json
from image_restoration.color_normalizer import quantize_image
from image_restoration.config import QuantizationConfig

from .region_segmenter import segment_by_color, compute_region_geometry, find_holes_in_region
from .region_merger import merge_regions, split_nested_regions
from .object_graph import ObjectGraph
from .primitive_recovery import recover_primitive, RecoveredPrimitive
from .curve_optimizer import optimize_curve
from .svg_generator import generate_svg


def reconstruct_logo(
    file_path: str,
    output_dir: str = "logo_output",
    quantize_k: int = 16,
    min_region: int = 50,
    merge_colors: bool = True,
) -> Dict[str, Any]:
    """Reconstruct a logo as geometric objects.

    Args:
        file_path: Path to PNG logo.
        output_dir: Output directory.
        quantize_k: K-Means color clusters (fewer = cleaner).
        min_region: Minimum region area.
        merge_colors: Whether to merge adjacent similar regions.

    Returns:
        Dict with keys: svg_path, objects, graph, stats, timing.
    """
    start = time.perf_counter()
    stem = get_file_stem(file_path)
    ensure_output_dir(output_dir)

    logger.info("=" * 50)
    logger.info("LOGO RECONSTRUCTOR: %s", Path(file_path).name)
    logger.info("=" * 50)

    # ═══ 1. Load & Quantize ═══════════════════════════════════
    logger.info("[1/6] Loading + quantizing...")
    image, meta = load_image(file_path)
    image_bgr, alpha = split_alpha(image, meta)
    image_bgr, meta = resize_if_large(image_bgr, meta, 4096)
    h, w = image_bgr.shape[:2]

    # Mandatory quantization
    qconfig = QuantizationConfig(auto_k=False, fallback_k=quantize_k)
    image_q = quantize_image(image_bgr, qconfig, image_bgr, "logo")

    # ═══ 2. Region Segmentation ═══════════════════════════════
    logger.info("[2/6] Segmenting regions...")
    regions = segment_by_color(image_q, tolerance=12, min_region_area=min_region)

    # Compute geometry for each region
    for i in range(len(regions)):
        regions[i] = compute_region_geometry(regions[i])

    logger.info("  %d raw regions found", len(regions))

    # ═══ 3. Region Merging ═════════════════════════════════════
    if merge_colors and len(regions) > 1:
        logger.info("[3/6] Merging adjacent similar regions...")
        regions = merge_regions(regions, color_threshold=35.0, min_area=min_region)
        regions = split_nested_regions(regions)
        logger.info("  %d regions after merging", len(regions))
    else:
        logger.info("[3/6] Skipping region merge")

    # Filter: remove tiny objects (noise)
    min_obj_area = max(30, (h * w) * 0.0002)  # At least 30px or 0.02% of image
    regions = [r for r in regions if r["area"] >= min_obj_area]
    logger.info("  %d regions after size filter (min=%d px)", len(regions), int(min_obj_area))

    # ═══ 4. Build Object Graph ════════════════════════════════
    logger.info("[4/6] Building object graph...")
    graph = ObjectGraph().build_from_regions(regions)
    logger.info("  %d objects in graph", len(graph))

    # ═══ 5. Primitive Recovery ════════════════════════════════
    logger.info("[5/6] Recovering primitives...")
    objects = []
    for node_id, node in graph.nodes.items():
        region = regions[node_id - 1] if node_id <= len(regions) else regions[0]

        primitive = recover_primitive(region, image_area=h * w)

        obj = {
            "id": node.id,
            "bbox": node.bbox,
            "area": node.area,
            "color_hex": node.color_hex,
            "centroid": node.centroid,
            "circularity": node.circularity,
            "rectangularity": node.rectangularity,
            "parent": node.parent,
            "children": node.children,
            "primitive": primitive,
            "contour": region.get("contour"),
            "has_fill": True,
            "z_order": node.id,
        }

        if primitive:
            node.primitive_type = primitive.type
            node.primitive_confidence = primitive.confidence

        objects.append(obj)

    # Count primitives
    prim_counts = {}
    for o in objects:
        if o["primitive"]:
            t = o["primitive"].type
            prim_counts[t] = prim_counts.get(t, 0) + 1
    logger.info("  Primitives: %s", prim_counts)

    # ═══ 6. Generate SVG ═══════════════════════════════════════
    logger.info("[6/6] Generating SVG...")
    svg_path = f"{output_dir}/{stem}_reconstructed.svg"
    generate_svg(objects, w, h, svg_path)

    # Save clean image
    clean_path = f"{output_dir}/{stem}_clean.png"
    save_image(image_q, clean_path)

    # Report
    elapsed = (time.perf_counter() - start) * 1000
    report = {
        "input": str(Path(file_path).absolute()),
        "svg": str(Path(svg_path).absolute()),
        "total_regions": len(regions),
        "total_objects": len(objects),
        "svg_elements": len(objects),
        "primitives_found": prim_counts,
        "processing_time_ms": elapsed,
    }
    report_path = f"{output_dir}/{stem}_reconstruction_report.json"
    save_json(report, report_path)

    logger.info("=" * 50)
    logger.info("DONE in %.0f ms | %d objects → SVG", elapsed, len(objects))
    logger.info("SVG: %s", svg_path)
    logger.info("=" * 50)

    return {
        "svg_path": svg_path,
        "clean_path": clean_path,
        "objects": objects,
        "graph": graph,
        "regions": regions,
        "processing_time_ms": elapsed,
        "report": report,
    }


# CLI entry
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "1.png"
    reconstruct_logo(path, output_dir="logo_output", quantize_k=16)
