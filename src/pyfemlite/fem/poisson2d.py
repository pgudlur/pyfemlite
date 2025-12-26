from __future__ import annotations
import numpy as np
from scipy.sparse.linalg import spsolve

from .shape_t3 import t3_area_and_grads
from .assembly import assemble_global, add_local_to_triplets
from .bc import apply_dirichlet
from .flux import add_poisson_neumann_rhs
from pyfemlite.mesh.boundary import Boundary

def solve_poisson_t3(
    X, T,
    kappa,
    rhs_func,
    *,
    boundary: Boundary | None = None,
    dirichlet: dict[str, callable] | None = None,
    neumann: dict[str, callable] | None = None,
    validate_boundary: bool = False,
    # legacy explicit
    dirichlet_nodes=None,
    dirichlet_value_func=None,
    neumann_edges=None,
    neumann_g=None,
):
    if validate_boundary and boundary is not None:
        boundary.validate(X, T, strict=True)

    nnode = X.shape[0]
    ndof = nnode
    I, J, V = [], [], []
    f = np.zeros(ndof, dtype=float)

    for e in range(T.shape[0]):
        conn = T[e]
        Xe = X[conn, :]
        A, dNdx = t3_area_and_grads(Xe)
        Ke = kappa * (dNdx @ dNdx.T) * A

        xc = Xe.mean(axis=0)
        fe = np.full(3, rhs_func(xc[0], xc[1]) * A / 3.0)

        add_local_to_triplets(conn, Ke, I, J, V)
        f[conn] += fe

    if boundary is not None and neumann:
        for grp, g_fn in neumann.items():
            add_poisson_neumann_rhs(f, X, boundary.edges[grp], g_fn)

    if neumann_edges is not None and neumann_g is not None:
        add_poisson_neumann_rhs(f, X, neumann_edges, neumann_g)

    K = assemble_global(ndof, I, J, V)

    dbc: dict[int, float] = {}

    if boundary is not None and dirichlet:
        for grp, fn in dirichlet.items():
            for n in boundary.nodes[grp]:
                x, y = X[int(n)]
                dbc[int(n)] = float(fn(float(x), float(y)))

    if dirichlet_nodes is not None and dirichlet_value_func is not None:
        for n in dirichlet_nodes:
            x, y = X[int(n)]
            dbc[int(n)] = float(dirichlet_value_func(float(x), float(y)))

    Kb, fb = apply_dirichlet(K, f, dbc)
    u = spsolve(Kb, fb)
    return u
