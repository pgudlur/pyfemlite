from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import spsolve

from .shape_t3 import t3_area_and_grads
from .assembly import assemble_global, add_local_to_triplets
from .bc import apply_dirichlet
from .traction import add_elasticity_traction_rhs
from pyfemlite.mesh.boundary import Boundary

def solve_elasticity_t3(
    X, T, D,
    body_force,
    *,
    boundary: Boundary | None = None,
    dirichlet: dict[str, callable] | None = None,   # group -> (ux,uy)
    traction: dict[str, callable] | None = None,    # group -> (tx,ty)
    validate_boundary: bool = False,
    # legacy explicit
    dbc_dofs: dict[int, float] | None = None,
    traction_edges=None,
    traction_func=None,
):
    if validate_boundary and boundary is not None:
        boundary.validate(X, T, strict=True)

    nnode = X.shape[0]
    ndof = 2 * nnode
    I, J, V = [], [], []
    f = np.zeros(ndof, dtype=float)

    for e in range(T.shape[0]):
        conn = T[e]
        Xe = X[conn, :]
        A, dNdx = t3_area_and_grads(Xe)

        B = np.zeros((3, 6), dtype=float)
        for a in range(3):
            dN_dx, dN_dy = dNdx[a, 0], dNdx[a, 1]
            B[0, 2*a + 0] = dN_dx
            B[1, 2*a + 1] = dN_dy
            B[2, 2*a + 0] = dN_dy
            B[2, 2*a + 1] = dN_dx

        Ke = (B.T @ D @ B) * A

        xc = Xe.mean(axis=0)
        bx, by = body_force(xc[0], xc[1])

        fe = np.zeros(6, dtype=float)
        for a in range(3):
            fe[2*a + 0] = bx * A / 3.0
            fe[2*a + 1] = by * A / 3.0

        edofs = np.array([2*conn[0], 2*conn[0]+1,
                          2*conn[1], 2*conn[1]+1,
                          2*conn[2], 2*conn[2]+1], dtype=int)

        add_local_to_triplets(edofs, Ke, I, J, V)
        f[edofs] += fe

    if boundary is not None and traction:
        for grp, tr_fn in traction.items():
            add_elasticity_traction_rhs(f, X, boundary.edges[grp], tr_fn)

    if traction_edges is not None and traction_func is not None:
        add_elasticity_traction_rhs(f, X, traction_edges, traction_func)

    K = assemble_global(ndof, I, J, V)

    dbc: dict[int, float] = {}

    if boundary is not None and dirichlet:
        for grp, disp_fn in dirichlet.items():
            for n in boundary.nodes[grp]:
                x, y = X[int(n)]
                ux, uy = disp_fn(float(x), float(y))
                dbc[2*int(n)] = float(ux)
                dbc[2*int(n)+1] = float(uy)

    if dbc_dofs is not None:
        dbc.update(dbc_dofs)

    Kb, fb = apply_dirichlet(K, f, dbc)
    u = spsolve(Kb, fb)
    return u
