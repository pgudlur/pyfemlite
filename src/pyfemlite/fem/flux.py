from __future__ import annotations
import numpy as np
from .quadrature import gauss_legendre_2, map_edge_reference_to_physical

def add_poisson_neumann_rhs(f: np.ndarray, X: np.ndarray, edges: np.ndarray, g_func):
    xi_q, w_q = gauss_legendre_2()

    edges = np.asarray(edges, dtype=int)
    if edges.size == 0:
        return

    for (na, nb) in edges:
        na = int(na); nb = int(nb)
        Xa = X[na]; Xb = X[nb]
        edofs = np.array([na, nb], dtype=int)
        fe = np.zeros(2, dtype=float)

        for xi, w in zip(xi_q, w_q):
            x, J = map_edge_reference_to_physical(float(xi), Xa, Xb)
            g = float(g_func(float(x[0]), float(x[1])))
            Na = 0.5 * (1.0 - xi)
            Nb = 0.5 * (1.0 + xi)
            fe += np.array([Na, Nb], dtype=float) * (g * w * J)

        f[edofs] += fe
