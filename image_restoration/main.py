"""
Vectorization Preprocessing Engine — CLI Entry Point.

Usage:
    python -m image_restoration.main <image> [options]

Examples:
    python -m image_restoration.main logo.png
    python -m image_restoration.main logo.png --debug --svg
    python -m image_restoration.main ui.png --preset ui
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from .config import (
    VectorizationPreprocessConfig,
    create_logo_config, create_ui_config, create_illustration_config,
)
from .restoration import preprocess_for_vectorization
from .utils import setup_logging, is_supported_image, logger


def main():
    """CLI entry point for the Vectorization Preprocessing Engine."""
    parser = argparse.ArgumentParser(
        description="Vectorization Preprocessing — Optimize images for contour detection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Presets (optimized for image type):
  logo          Aggressive quantization (2-16 colors)
  ui            Moderate quantization (8-32 colors)
  illustration  Balanced quantization (16-64 colors)

Examples:
  python -m image_restoration.main logo.png
  python -m image_restoration.main ui.png --preset ui --debug --svg
        """,
    )
    parser.add_argument("image", help="Path to input PNG/JPG image.")
    parser.add_argument("--output", "-o", default="vector_ready", help="Output directory.")
    parser.add_argument(
        "--preset", "-p", choices=["logo", "ui", "illustration"],
        default="logo", help="Preprocessing preset based on image type.",
    )
    parser.add_argument("--debug", "-d", action="store_true", help="Generate debug visuals.")
    parser.add_argument("--no-svg", action="store_true", help="Skip SVG export.")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG","INFO","WARNING","ERROR"])
    args = parser.parse_args()

    setup_logging(args.log_level)

    if not Path(args.image).exists():
        logger.error("File not found: %s", args.image); sys.exit(1)
    if not is_supported_image(args.image):
        logger.error("Unsupported format."); sys.exit(1)

    preset_map = {"logo": create_logo_config, "ui": create_ui_config,
                   "illustration": create_illustration_config}
    config = preset_map[args.preset]()
    if args.no_svg:
        config.output_svg = False

    try:
        result = preprocess_for_vectorization(args.image, config, args.output, args.debug)
    except Exception as e:
        logger.error("Failed: %s", e, exc_info=True); sys.exit(1)

    vs = result["after_metrics"].get("vectorization_score", 0)
    vs_before = result["before_metrics"].get("vectorization_score", 0)
    print(f"\n✅ Done in {result['processing_time_ms']:.0f} ms")
    print(f"📊 Vectorization Score: {vs_before:.1f} → {vs:.1f} ({vs-vs_before:+.1f})")
    print(f"🏷️  Image type: {result['image_type']}")
    print(f"📁 Clean: {result['clean_path']}")
    if result["svg_contour_path"]:
        print(f"📐 SVG Contour: {result['svg_contour_path']}")
    if result["svg_filled_path"]:
        print(f"🎨 SVG Filled: {result['svg_filled_path']}")
