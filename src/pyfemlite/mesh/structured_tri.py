from __future__ import annotations
import numpy as np
from .boundary import Boundary, edges_from_node_chain

def structured_unit_square_tri(nx: int, ny: int):
    xs = np.linspace(0.0, 1.0, nx + 1)
    ys = np.linspace(0.0, 1.0, ny + 1)
    X = np.array([(x, y) for y in ys for x in xs], dtype=float)

    def nid(i, j):
        return j * (nx + 1) + i

    elems = []
    for j in range(ny):
        for i in range(nx):
            n00 = nid(i, j)
            n10 = nid(i + 1, j)
            n01 = nid(i, j + 1)
            n11 = nid(i + 1, j + 1)
            elems.append((n00, n10, n11))
            elems.append((n00, n11, n01))
    T = np.array(elems, dtype=int)

    left   = np.array([nid(0,  j) for j in range(ny + 1)], dtype=int)
    right  = np.array([nid(nx, j) for j in range(ny + 1)], dtype=int)
    bottom = np.array([nid(i, 0) for i in range(nx + 1)], dtype=int)
    top    = np.array([nid(i, ny) for i in range(nx + 1)], dtype=int)

    nodes = {"left": left, "right": right, "bottom": bottom, "top": top}
    edges = {
        "left":   edges_from_node_chain(left),
        "right":  edges_from_node_chain(right),
        "bottom": edges_from_node_chain(bottom),
        "top":    edges_from_node_chain(top),
    }
    boundary = Boundary(nodes=nodes, edges=edges)
    boundary.validate(X, T, strict=True)
    return X, T, boundary
