"""
JSON export module for the ImageVectorizer.

Serializes the vectorization result to a structured JSON format
compatible with downstream consumers (PPTX, SVG, DXF generators).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .models import VectorizationResult, PrimitiveBase
from .utils import logger, ensure_output_dir


def export_to_json(
    result: VectorizationResult,
    output_path: str,
    pretty: bool = True,
    include_image_info: bool = True,
) -> str:
    """Export the vectorization result to a JSON file.

    Args:
        result: Complete vectorization result.
        output_path: Path for the output JSON file.
        pretty: Use indented formatting.
        include_image_info: Include image metadata.

    Returns:
        The output file path.

    Raises:
        IOError: If the file cannot be written.
    """
    output_dir = Path(output_path).parent
    ensure_output_dir(str(output_dir))

    data = result.to_dict()

    # Optionally strip image info for compact output
    if not include_image_info:
        data.pop("image_width", None)
        data.pop("image_height", None)

    indent = 2 if pretty else None

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
    except IOError as e:
        logger.error("Failed to write JSON: %s", e)
        raise

    file_size = Path(output_path).stat().st_size
    logger.info(
        "Exported JSON: %s (%d objects, %.1f KB)",
        output_path, result.total_objects, file_size / 1024.0,
    )

    return output_path


def export_to_json_string(
    result: VectorizationResult,
    pretty: bool = False,
) -> str:
    """Export the vectorization result to a JSON string.

    Args:
        result: Complete vectorization result.
        pretty: Use indented formatting.

    Returns:
        JSON string.
    """
    data = result.to_dict()
    indent = 2 if pretty else None
    return json.dumps(data, indent=indent, ensure_ascii=False)


def export_objects_list(
    primitives: List[PrimitiveBase],
    output_path: str,
    image_width: int = 0,
    image_height: int = 0,
) -> str:
    """Export only the objects list (without image metadata) to JSON.

    Useful for debugging or partial results.

    Args:
        primitives: List of detected primitives.
        output_path: Output file path.
        image_width: Optional image width for context.
        image_height: Optional image height for context.

    Returns:
        Output file path.
    """
    from .models import ImageInfo, VectorizationResult
    info = ImageInfo(width=image_width, height=image_height, channels=0, has_alpha=False)
    result = VectorizationResult(
        image_info=info,
        objects=list(primitives),
        total_objects=len(primitives),
    )
    return export_to_json(result, output_path, pretty=True)


def load_from_json(file_path: str) -> dict:
    """Load a previously exported JSON file.

    Args:
        file_path: Path to the JSON file.

    Returns:
        Parsed dictionary.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
