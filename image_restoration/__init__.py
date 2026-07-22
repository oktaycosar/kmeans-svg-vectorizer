"""
Image Restoration Engine — Optimize images for vectorization.

A deterministic, non-AI preprocessing pipeline that simplifies images
for contour detection: denoising, edge preservation, mandatory color
quantization, artifact removal, and contour optimization.
"""

__version__ = "2.0.0"

from .config import (
    VectorizationPreprocessConfig,
    create_logo_config,
    create_ui_config,
    create_illustration_config,
)
from .restoration import preprocess_for_vectorization

__all__ = [
    "__version__",
    "VectorizationPreprocessConfig",
    "create_logo_config",
    "create_ui_config",
    "create_illustration_config",
    "preprocess_for_vectorization",
]
