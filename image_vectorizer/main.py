"""
Main entry point for the ImageVectorizer pipeline.

Orchestrates the full pipeline:
Image → Preprocessing → Contour Detection → Primitive Detection →
Color Analysis → Classification → JSON Export → Visualization

Usage:
    python -m image_vectorizer.main <input_image> [options]
    python -m image_vectorizer.main 1.png --output results/ --debug
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, List

from .config import PipelineConfig, create_fast_config, create_high_quality_config
from .models import VectorizationResult, PrimitiveBase, ImageInfo
from .image_loader import load_image, resize_if_needed
from .preprocess import preprocess_pipeline, preprocess_simple
from .contour_detector import find_contours, filter_contours
from .classifier import classify_contours, get_classification_stats
from .json_exporter import export_to_json
from .svg_exporter import export_svg
from .visualization import generate_debug_image, draw_contours_debug
from .utils import (
    setup_logging, logger, ensure_output_dir,
    get_file_stem, format_timestamp, is_supported_image,
)


def run_pipeline(
    image_path: str,
    config: Optional[PipelineConfig] = None,
    output_dir: str = "output",
) -> VectorizationResult:
    """Run the complete ImageVectorizer pipeline on a single image.

    Args:
        image_path: Path to the input PNG/JPG image.
        config: Pipeline configuration (uses default if None).
        output_dir: Directory for output files.

    Returns:
        VectorizationResult with all detected primitives.

    Raises:
        FileNotFoundError: If image_path doesn't exist.
        ValueError: If the image format is unsupported.
    """
    if config is None:
        config = PipelineConfig()

    start_time = time.perf_counter()

    ensure_output_dir(output_dir)
    stem = get_file_stem(image_path)

    logger.info("=" * 60)
    logger.info("ImageVectorizer Pipeline: %s", Path(image_path).name)
    logger.info("=" * 60)

    # ── STEP 1: Load Image ────────────────────────────────────────
    logger.info("[Step 1/6] Loading image...")
    image, image_info = load_image(image_path)

    # Resize if too large
    image, image_info = resize_if_needed(image, image_info, config.max_dimension)

    # ── STEP 2: Preprocessing ─────────────────────────────────────
    logger.info("[Step 2/6] Preprocessing...")
    if config.enable_preprocessing:
        preprocess_results = preprocess_pipeline(
            image,
            config.preprocess,
            output_binary=True,
            output_edges=True,
        )
        gray = preprocess_results["gray"]
        binary = preprocess_results.get("binary")
        edges = preprocess_results.get("edges")
    else:
        # Minimal preprocessing
        gray = preprocess_simple(image)
        binary = None
        edges = preprocess_simple(image)

    # Save intermediate results for debugging
    if config.visualization.enabled and binary is not None:
        import cv2
        cv2.imwrite(f"{output_dir}/{stem}_binary.png", binary)
    if config.visualization.enabled and edges is not None:
        import cv2
        cv2.imwrite(f"{output_dir}/{stem}_edges.png", edges)

    # ── STEP 3: Contour Detection ─────────────────────────────────
    logger.info("[Step 3/6] Detecting contours...")
    if config.enable_contour_detection and binary is not None:
        contours, hierarchy = find_contours(
            binary,
            retrieval_mode=config.contour.retrieval_mode,
            approximation=config.contour.approximation_method,
        )

        image_area = image_info.width * image_info.height
        enriched = filter_contours(contours, config.contour, image_area)
    else:
        enriched = []
        logger.warning("Contour detection disabled or no binary image available.")

    # ── STEP 4: Primitive Detection + Classification ──────────────
    logger.info("[Step 4/6] Classifying primitives...")
    if config.enable_primitive_detection or config.enable_classification:
        primitives = classify_contours(
            enriched, image, gray, edges, config,
        )
    else:
        primitives = []

    # ── STEP 5: Export JSON ───────────────────────────────────────
    logger.info("[Step 5/6] Exporting results...")
    result = VectorizationResult(
        image_info=image_info,
        objects=primitives,
        total_objects=len(primitives),
    )

    if config.output_json:
        json_path = f"{output_dir}/{stem}_vectorized.json"
        export_to_json(result, json_path, pretty=config.pretty_print_json)

    # ── STEP 6: Debug Visualization ───────────────────────────────
    logger.info("[Step 6/6] Generating visualizations...")
    if config.visualization.enabled:
        viz_config = config.visualization
        viz_config.output_dir = output_dir  # Override with actual output dir
        debug_path = f"{output_dir}/{stem}_debug.png"
        generate_debug_image(image, primitives, viz_config, debug_path)

        # Also draw raw contours
        if enriched:
            contour_path = f"{output_dir}/{stem}_contours.png"
            raw_contours = [c["contour"] for c in enriched]
            draw_contours_debug(image, raw_contours, contour_path)

    # ── Timing ────────────────────────────────────────────────────
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    result.processing_time_ms = elapsed_ms

    # ── Summary ────────────────────────────────────────────────────
    stats = get_classification_stats(primitives)
    logger.info("=" * 60)
    logger.info("Pipeline complete in %.0f ms", elapsed_ms)
    logger.info("Objects detected: %d", len(primitives))
    for type_name, count in sorted(stats.items()):
        logger.info("  %s: %d", type_name, count)
    logger.info("Output directory: %s", output_dir)
    logger.info("=" * 60)

    return result


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="ImageVectorizer — Convert PNG/JPG images to editable vector primitives.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m image_vectorizer.main 1.png
  python -m image_vectorizer.main 1.png --output ./results --debug
  python -m image_vectorizer.main 1.png --quality high
  python -m image_vectorizer.main 1.png --quality fast
        """,
    )
    parser.add_argument("image", help="Path to input PNG or JPG image.")
    parser.add_argument(
        "--output", "-o", default="output",
        help="Output directory for results (default: ./output).",
    )
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a JSON configuration file (overrides defaults).",
    )
    parser.add_argument(
        "--quality", "-q", choices=["default", "fast", "high"],
        default="default",
        help="Predefined quality preset: fast, high, or default.",
    )
    parser.add_argument(
        "--debug", "-d", action="store_true",
        help="Enable debug visualization output.",
    )
    parser.add_argument(
        "--no-json", action="store_true",
        help="Skip JSON export.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO).",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Validate input
    if not Path(args.image).exists():
        logger.error("File not found: %s", args.image)
        sys.exit(1)

    if not is_supported_image(args.image):
        logger.error(
            "Unsupported format: %s. Use PNG, JPG, JPEG, BMP, TIFF, or WebP.",
            args.image,
        )
        sys.exit(1)

    # Select configuration
    if args.quality == "fast":
        config = create_fast_config()
    elif args.quality == "high":
        config = create_high_quality_config()
    else:
        config = PipelineConfig()

    # Override with debug flag
    if args.debug:
        config.visualization.enabled = True

    # Override with no-json flag
    if args.no_json:
        config.output_json = False

    # Run pipeline
    try:
        result = run_pipeline(args.image, config, args.output)
    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        sys.exit(1)

    # Print summary to stdout
    if result.total_objects > 0:
        print(f"\n✅ {result.total_objects} objects detected in {result.processing_time_ms:.0f} ms")
        stats = get_classification_stats(result.objects)
        for name, count in sorted(stats.items()):
            print(f"   • {name}: {count}")
        print(f"\n📁 Results saved to: {args.output}/")
    else:
        print(f"\n⚠️  No objects detected. Try --quality high or adjust parameters.")


if __name__ == "__main__":
    main()
