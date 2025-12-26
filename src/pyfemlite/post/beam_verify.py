from __future__ import annotations
import numpy as np

def tip_deflection_right_edge(X: np.ndarray, U: np.ndarray, boundary, *, use_abs: bool = True) -> float:
    '''
    Compute cantilever tip deflection from nodal displacements on boundary group "right".

    Parameters
    ----------
    X : (nnode,2) array
    U : (nnode,2) array displacement field (ux,uy)
    boundary : Boundary
    use_abs : if True, return max(|uy|) on right edge; if False, return min(uy) (downward)

    Returns
    -------
    float : tip deflection measure
    '''
    right_nodes = np.asarray(boundary.nodes["right"], dtype=int)
    uy = U[right_nodes, 1]
    if uy.size == 0:
        raise ValueError("Boundary group 'right' has no nodes.")
    if use_abs:
        return float(np.max(np.abs(uy)))
    return float(np.min(uy))

def euler_bernoulli_tip_deflection_end_traction(T0: float, L: float, H: float, E: float) -> float:
    r'''
    Euler–Bernoulli cantilever tip deflection for end traction T0 applied uniformly over height H
    (unit thickness in z).

    Resultant force: P = T0 * H
    I = H^3 / 12
    delta = P L^3 / (3 E I) = 4 T0 L^3 / (E H^2)

    Returns positive magnitude (|delta|).
    '''
    if H <= 0 or L <= 0 or E <= 0:
        raise ValueError("L, H, E must be positive.")
    return float(4.0 * T0 * (L**3) / (E * (H**2)))
