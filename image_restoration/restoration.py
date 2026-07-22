"""
Vectorization-Optimized Preprocessing Pipeline.

NEW PHILOSOPHY: Simplify, never complicate.
Pipeline: Load → Analyze → Denoise → Edge-Preserve →
         Quantize (MANDATORY) → Cleanup → Contour Optimize →
         SVG Export → Vector Metrics.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Any

import cv2
import numpy as np

from .config import VectorizationPreprocessConfig, create_logo_config
from .loader import load_image, split_alpha, merge_alpha, resize_if_large, save_image
from .image_analyzer import analyze_image_type
from .denoise import adaptive_denoise
from .edge_preservation import guided_filter
from .color_normalizer import quantize_image
from .background_cleaner import clean_background
from .contour_optimizer import optimize_contours
from .svg_exporter import image_to_svg_contours, image_to_svg_filled
from .quality_metrics import compute_all_vector_metrics, compare_vector_metrics
from .visualization import generate_all_debug_visuals
from .utils import logger, ensure_output_dir, get_file_stem, save_json


def preprocess_for_vectorization(
    file_path: str,
    config: VectorizationPreprocessConfig = None,
    output_dir: str = "vector_ready",
    debug: bool = False,
) -> Dict[str, Any]:
    """Run the vectorization-optimized preprocessing pipeline.

    The output is optimized for contour detection, not visual quality.
    NO sharpening, NO contrast enhancement.
    MANDATORY color quantization.

    Args:
        file_path: Input image path.
        config: Vectorization config.
        output_dir: Output directory.
        debug: Generate debug visuals.

    Returns:
        Dict with: clean_path, svg_path, metrics, analysis, timing.
    """
    if config is None:
        config = create_logo_config()

    start_time = time.perf_counter()
    stem = get_file_stem(file_path)
    ensure_output_dir(output_dir)

    logger.info("=" * 60)
    logger.info("VECTORIZATION PREPROCESS: %s", Path(file_path).name)
    logger.info("=" * 60)

    # ═══ STEP 1: Load ═══════════════════════════════════════════
    logger.info("[1/9] Loading...")
    image, metadata = load_image(file_path)
    image_bgr, alpha = split_alpha(image, metadata)
    original_bgr = image_bgr.copy()
    image_bgr, metadata = resize_if_large(image_bgr, metadata, config.max_dimension)

    # ═══ STEP 2: Analyze Image Type ═════════════════════════════
    logger.info("[2/9] Analyzing image type...")
    analysis = analyze_image_type(image_bgr)
    image_type = analysis["type"]

    # ═══ STEP 3: Adaptive Denoise ══════════════════════════════
    logger.info("[3/9] Adaptive denoising...")
    if config.enable_denoise:
        image_bgr = adaptive_denoise(image_bgr, config.denoise)

    # ═══ STEP 4: Edge-Preserving Smooth ════════════════════════
    logger.info("[4/9] Edge-preserving smoothing...")
    if config.enable_edge_preserve:
        image_bgr = guided_filter(
            image_bgr,
            config.edge_preserve.guided_radius,
            config.edge_preserve.guided_eps,
        )

    # ═══ STEP 5: MANDATORY Color Quantization ══════════════════
    logger.info("[5/9] MANDATORY color quantization...")
    if config.enable_quantization:
        image_bgr = quantize_image(
            image_bgr, config.quantization, original_bgr, image_type
        )

    # ═══ STEP 6: Artifact Cleanup ══════════════════════════════
    logger.info("[6/9] Cleaning artifacts...")
    if config.enable_cleanup:
        image_bgr = clean_background(image_bgr, config.cleanup)

    # ═══ STEP 7: Contour Optimization ══════════════════════════
    logger.info("[7/9] Optimizing contours...")
    if config.enable_contour_optimize:
        image_bgr = optimize_contours(image_bgr, config.contour_optimize)

    # ═══ STEP 8: Vectorization Metrics ═════════════════════════
    logger.info("[8/9] Computing vectorization metrics...")
    after_metrics = compute_all_vector_metrics(image_bgr, config.vector_metrics)

    # Compute before metrics (on original, for comparison)
    before_metrics = compute_all_vector_metrics(original_bgr, config.vector_metrics)
    comparison = compare_vector_metrics(before_metrics, after_metrics)

    # ═══ STEP 9: Save Outputs ══════════════════════════════════
    logger.info("[9/9] Saving outputs...")

    final_image = merge_alpha(image_bgr, alpha)
    clean_path = f"{output_dir}/{stem}_clean.png"
    save_image(final_image, clean_path)

    # SVG Export
    svg_contour_path = ""
    svg_filled_path = ""
    if config.output_svg:
        svg_contour_path = f"{output_dir}/{stem}_contours.svg"
        image_to_svg_contours(image_bgr, svg_contour_path, min_area=30.0)

        svg_filled_path = f"{output_dir}/{stem}_filled.svg"
        image_to_svg_filled(image_bgr, svg_filled_path, min_area=50.0)

    # Debug visuals
    debug_paths = {}
    if debug:
        debug_paths = generate_all_debug_visuals(
            original_bgr, image_bgr, output_dir, stem
        )

    # Vectorization report
    report = {
        "input": str(Path(file_path).absolute()),
        "output": str(Path(clean_path).absolute()),
        "image_type": image_type,
        "image_type_confidence": analysis["confidence"],
        "recommended_colors": analysis["recommended_colors"],
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "comparison": comparison,
        "svg_contour": svg_contour_path,
        "svg_filled": svg_filled_path,
    }
    report_path = f"{output_dir}/{stem}_vectorization_report.json"
    save_json(report, report_path)

    # Timing
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    vs_before = before_metrics.get("vectorization_score", 0)
    vs_after = after_metrics.get("vectorization_score", 0)

    logger.info("=" * 60)
    logger.info("Complete in %.0f ms", elapsed_ms)
    logger.info("Vectorization score: %.1f → %.1f (%+.1f)", vs_before, vs_after, vs_after - vs_before)
    logger.info("Type: %s | Colors: %d", image_type, after_metrics.get("color_clusters", 0))
    logger.info("Output: %s", clean_path)
    logger.info("=" * 60)

    return {
        "clean_path": clean_path,
        "svg_contour_path": svg_contour_path,
        "svg_filled_path": svg_filled_path,
        "image_type": image_type,
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "comparison": comparison,
        "debug_paths": debug_paths,
        "processing_time_ms": elapsed_ms,
        "report": report,
    }
