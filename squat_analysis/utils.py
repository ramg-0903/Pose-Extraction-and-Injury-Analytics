"""
Geometry helpers for biomechanical angle computation.

All functions operate on 2-D NumPy arrays (x, y) and return
degrees or unit vectors.  NaN-safe wrappers live in features.py.
"""

import numpy as np


def angle_between(a: np.ndarray, vertex: np.ndarray, b: np.ndarray) -> float:
    """Interior angle at *vertex* formed by rays vertex→a and vertex→b.

    Returns degrees in [0, 180].  Returns 0.0 for degenerate inputs.
    """
    u = a - vertex
    v = b - vertex
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-9 or nv < 1e-9:
        return 0.0
    cos_theta = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_theta)))


def angle_to_vertical(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of segment a→b relative to the Y-up axis.

    0° = perfectly upright, 90° = horizontal.
    Uses abs(cos) so the sign of the vertical component doesn't matter.
    """
    vec = b - a
    norm = np.linalg.norm(vec)
    if norm < 1e-9:
        return 0.0
    cos_theta = np.clip(np.dot(vec / norm, np.array([0.0, 1.0])), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_theta))))


def midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """2-D midpoint between two points."""
    return (a + b) / 2.0


def unit_vec(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Unit vector from *a* to *b*.  Returns zero vector if degenerate."""
    v = b - a
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.zeros(2)