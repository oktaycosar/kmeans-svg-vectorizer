"""
Vectorization-optimized preprocessing configuration.

Philosophy: SIMPLIFY the image, never complicate it.
- NO sharpening (creates halos → fragments contours)
- NO contrast enhancement (creates new colors)
- MANDATORY color quantization (fewer colors = cleaner contours)
- Every operation must reduce, not increase, contour complexity.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class DenoiseConfig:
    """Adaptive noise removal — edge-preserving only."""
    nlm_strength: float = 3.0
    nlm_template: int = 7
    nlm_search: int = 21
    bilateral_d: int = 5
    bilateral_sigma_color: float = 30.0
    bilateral_sigma_space: float = 30.0
    median_kernel: int = 0
    adaptive: bool = True


@dataclass
class EdgePreserveConfig:
    """Edge-preserving smoothing — NO sharpening, NO halos."""
    guided_radius: int = 3
    guided_eps: float = 0.01


@dataclass
class QuantizationConfig:
    """Mandatory color quantization — KEY to good vectorization."""
    auto_k: bool = True
    fallback_k: int = 16
    kmeans_iterations: int = 15
    kmeans_epsilon: float = 1.0
    preserve_extremes: bool = True
    merge_delta_e: float = 3.0


@dataclass
class CleanupConfig:
    """Artifact removal + morphological cleanup."""
    min_component_area: int = 20
    morph_close_kernel: Tuple[int, int] = (3, 3)
    morph_open_kernel: Tuple[int, int] = (2, 2)
    remove_jpeg: bool = True
    remove_isolated: bool = True


@dataclass
class ContourOptimizeConfig:
    """Final contour optimization for OpenCV."""
    close_gaps: bool = True
    gap_kernel: Tuple[int, int] = (4, 4)
    merge_edges: bool = True
    min_edge_length: int = 10
    anti_alias_sigma: float = 0.3


@dataclass
class VectorMetricsConfig:
    """Vectorization-specific quality metrics."""
    compute_contour_count: bool = True
    compute_closed_ratio: bool = True
    compute_fragmentation: bool = True
    compute_avg_contour_length: bool = True
    compute_noise_contours: bool = True
    compute_color_clusters: bool = True
    compute_edge_continuity: bool = True
    compute_estimated_primitives: bool = True


@dataclass
class VectorizationPreprocessConfig:
    """Master config — optimized for vectorization, NOT visual quality."""
    denoise: DenoiseConfig = field(default_factory=DenoiseConfig)
    edge_preserve: EdgePreserveConfig = field(default_factory=EdgePreserveConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)
    contour_optimize: ContourOptimizeConfig = field(default_factory=ContourOptimizeConfig)
    vector_metrics: VectorMetricsConfig = field(default_factory=VectorMetricsConfig)

    enable_denoise: bool = True
    enable_edge_preserve: bool = True
    enable_quantization: bool = True
    enable_cleanup: bool = True
    enable_contour_optimize: bool = True

    max_dimension: int = 4096
    output_svg: bool = True
    log_level: str = "INFO"


def create_logo_config() -> VectorizationPreprocessConfig:
    """Optimized for logos: aggressive quantization (2-16 colors)."""
    c = VectorizationPreprocessConfig()
    c.quantization.fallback_k = 8
    c.quantization.merge_delta_e = 5.0
    c.cleanup.min_component_area = 15
    c.contour_optimize.anti_alias_sigma = 0.2
    return c


def create_ui_config() -> VectorizationPreprocessConfig:
    """Optimized for UI screenshots: moderate quantization (8-32 colors)."""
    c = VectorizationPreprocessConfig()
    c.quantization.fallback_k = 24
    c.cleanup.min_component_area = 25
    c.denoise.median_kernel = 3
    return c


def create_illustration_config() -> VectorizationPreprocessConfig:
    """Optimized for illustrations: balanced (16-64 colors)."""
    c = VectorizationPreprocessConfig()
    c.quantization.fallback_k = 48
    c.cleanup.min_component_area = 30
    c.denoise.nlm_strength = 5.0
    c.contour_optimize.anti_alias_sigma = 0.4
    return c
