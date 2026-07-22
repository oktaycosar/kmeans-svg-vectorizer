"""Geometry Corrector — AI-assisted geometry fixing (Stage 2)."""
from .corrector import correct_path, correct_svg_paths, snap_right_angle, straighten_collinear
from .ai_detector import analyze_path, ai_verdict, detect_right_angles, detect_collinear_runs
