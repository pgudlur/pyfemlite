from __future__ import annotations
import numpy as np
from .quadrature import gauss_legendre_2, map_edge_reference_to_physical

def add_elasticity_traction_rhs(f: np.ndarray, X: np.ndarray, edges: np.ndarray, traction_func):
    xi_q, w_q = gauss_legendre_2()

    edges = np.asarray(edges, dtype=int)
    if edges.size == 0:
        return

    for (na, nb) in edges:
        na = int(na); nb = int(nb)
        Xa = X[na]; Xb = X[nb]
        edofs = np.array([2*na, 2*na+1, 2*nb, 2*nb+1], dtype=int)
        fe = np.zeros(4, dtype=float)

        for xi, w in zip(xi_q, w_q):
            x, J = map_edge_reference_to_physical(float(xi), Xa, Xb)
            tx, ty = traction_func(float(x[0]), float(x[1]))
            Na = 0.5 * (1.0 - xi)
            Nb = 0.5 * (1.0 + xi)
            Nmat = np.array([[Na, 0.0],
                             [0.0, Na],
                             [Nb, 0.0],
                             [0.0, Nb]], dtype=float)
            fe += (Nmat @ np.array([tx, ty], dtype=float)) * (w * J)

        f[edofs] += fe
