from __future__ import annotations
import numpy as np

def gauss_legendre_2():
    a = 1.0 / np.sqrt(3.0)
    xi = np.array([-a, a], dtype=float)
    w = np.array([1.0, 1.0], dtype=float)
    return xi, w

def map_edge_reference_to_physical(xi: float, Xa: np.ndarray, Xb: np.ndarray):
    x = 0.5 * (1.0 - xi) * Xa + 0.5 * (1.0 + xi) * Xb
    L = float(np.linalg.norm(Xb - Xa))
    J = 0.5 * L
    return x, J
