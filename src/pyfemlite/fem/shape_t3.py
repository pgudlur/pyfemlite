from __future__ import annotations
import numpy as np

def t3_area_and_grads(Xe: np.ndarray):
    x1, y1 = Xe[0]
    x2, y2 = Xe[1]
    x3, y3 = Xe[2]
    J = np.array([[x2 - x1, x3 - x1],
                  [y2 - y1, y3 - y1]], dtype=float)
    detJ = np.linalg.det(J)
    A = 0.5 * abs(detJ)
    if A <= 0.0:
        raise ValueError("Degenerate triangle with non-positive area.")
    b1 = y2 - y3
    b2 = y3 - y1
    b3 = y1 - y2
    c1 = x3 - x2
    c2 = x1 - x3
    c3 = x2 - x1
    dNdx = np.array([[b1, c1],
                     [b2, c2],
                     [b3, c3]], dtype=float) / detJ
    return A, dNdx
