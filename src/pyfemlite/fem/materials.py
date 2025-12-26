from __future__ import annotations
import numpy as np

def D_plane_stress(E: float, nu: float) -> np.ndarray:
    c = E / (1.0 - nu**2)
    return c * np.array([
        [1.0, nu, 0.0],
        [nu, 1.0, 0.0],
        [0.0, 0.0, (1.0 - nu) / 2.0]
    ], dtype=float)

def D_plane_strain(E: float, nu: float) -> np.ndarray:
    c = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return c * np.array([
        [1.0 - nu, nu, 0.0],
        [nu, 1.0 - nu, 0.0],
        [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]
    ], dtype=float)
