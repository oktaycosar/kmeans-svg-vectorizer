"""
Logo Reconstructor — Recover geometric objects from logo images.

Object-level reconstruction engine: segments regions, builds object
graph, recovers geometric primitives, and generates clean SVG with
one SVG element per logical object.
"""

__version__ = "1.0.0"

from .logo_reconstructor import reconstruct_logo

__all__ = ["__version__", "reconstruct_logo"]
