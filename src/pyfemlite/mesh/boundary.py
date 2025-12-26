from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
from typing import Callable

@dataclass(frozen=True)
class Boundary:
    nodes: dict[str, np.ndarray]
    edges: dict[str, np.ndarray]

    def validate(self, X: np.ndarray, T: np.ndarray, *, strict: bool = True) -> None:
        if X.ndim != 2 or X.shape[1] != 2:
            raise ValueError("X must have shape (nnode,2).")
        if T.ndim != 2 or T.shape[1] != 3:
            raise ValueError("T must have shape (nelem,3).")

        nnode = X.shape[0]
        all_bnd_edges = extract_boundary_edges(T)
        all_bnd_set = _edge_set(all_bnd_edges)

        for name, arr in self.nodes.items():
            a = np.asarray(arr, dtype=int).ravel()
            if a.size == 0:
                continue
            if np.any(a < 0) or np.any(a >= nnode):
                bad = a[(a < 0) | (a >= nnode)]
                raise ValueError(f"Boundary.nodes['{name}'] has out-of-range node ids: {bad[:10]}")
            if np.unique(a).size != a.size:
                raise ValueError(f"Boundary.nodes['{name}'] contains duplicate node ids.")

        for name, E in self.edges.items():
            e = np.asarray(E, dtype=int)
            if e.size == 0:
                continue
            if e.ndim != 2 or e.shape[1] != 2:
                raise ValueError(f"Boundary.edges['{name}'] must have shape (nedges,2).")
            i = np.minimum(e[:, 0], e[:, 1])
            j = np.maximum(e[:, 0], e[:, 1])
            e2 = np.column_stack([i, j]).astype(int)

            if np.any(e2[:, 0] == e2[:, 1]):
                raise ValueError(f"Boundary.edges['{name}'] contains a zero-length edge (i==j).")

            if np.any(e2 < 0) or np.any(e2 >= nnode):
                bad_rows = np.where((e2 < 0) | (e2 >= nnode))[0]
                raise ValueError(f"Boundary.edges['{name}'] has out-of-range node ids in rows: {bad_rows[:10]}")

            if np.unique(e2, axis=0).shape[0] != e2.shape[0]:
                raise ValueError(f"Boundary.edges['{name}'] contains duplicate edges.")

            grp_set = _edge_set(e2)
            extra = grp_set.difference(all_bnd_set)
            if extra:
                ex = np.array(list(extra), dtype=int)
                raise ValueError(
                    f"Boundary.edges['{name}'] contains edges not on the mesh boundary. Example: {ex[0].tolist()}"
                )

            if strict:
                node_set = set(np.asarray(self.nodes.get(name, np.array([], dtype=int)), dtype=int).ravel().tolist())
                if not node_set:
                    raise ValueError(
                        f"Boundary.edges['{name}'] is non-empty but Boundary.nodes['{name}'] is empty/missing."
                    )
                for (a, b) in grp_set:
                    if a not in node_set or b not in node_set:
                        raise ValueError(
                            f"Boundary group '{name}' edge endpoints must be included in nodes['{name}']. "
                            f"Offending edge: {(a, b)}"
                        )

    def summary(self) -> str:
        lines = ["Boundary groups:"]
        for name in sorted(set(self.nodes.keys()) | set(self.edges.keys())):
            n_nodes = int(np.asarray(self.nodes.get(name, np.array([], dtype=int))).size)
            n_edges = int(np.asarray(self.edges.get(name, np.zeros((0, 2), dtype=int))).shape[0])
            lines.append(f"  - {name:10s}: nodes={n_nodes:6d}, edges={n_edges:6d}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()

def _edge_set(edges: np.ndarray) -> set[tuple[int, int]]:
    edges = np.asarray(edges, dtype=int)
    if edges.size == 0:
        return set()
    return set((int(i), int(j)) for i, j in edges)

def extract_boundary_edges(T: np.ndarray) -> np.ndarray:
    edge_count = defaultdict(int)
    for tri in T:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for i, j in [(a, b), (b, c), (c, a)]:
            if i > j:
                i, j = j, i
            edge_count[(i, j)] += 1
    bnd = [e for e, cnt in edge_count.items() if cnt == 1]
    return np.array(bnd, dtype=int)

def edges_from_node_chain(node_ids: np.ndarray) -> np.ndarray:
    node_ids = np.asarray(node_ids, dtype=int).ravel()
    if node_ids.size < 2:
        return np.zeros((0, 2), dtype=int)
    e = np.column_stack([node_ids[:-1], node_ids[1:]])
    a = np.minimum(e[:, 0], e[:, 1])
    b = np.maximum(e[:, 0], e[:, 1])
    return np.column_stack([a, b]).astype(int)

def build_boundary_from_predicates(
    X: np.ndarray,
    T: np.ndarray,
    predicates: dict[str, Callable[[float, float], bool]],
) -> Boundary:
    all_bnd_edges = extract_boundary_edges(T)

    nodes: dict[str, np.ndarray] = {}
    edges: dict[str, np.ndarray] = {}

    for name, pred in predicates.items():
        mask_nodes = np.array([bool(pred(float(x), float(y))) for x, y in X], dtype=bool)
        group_nodes = np.where(mask_nodes)[0].astype(int)
        nodes[name] = group_nodes

        mask_edges = []
        for i, j in all_bnd_edges:
            xi, yi = X[int(i)]
            xj, yj = X[int(j)]
            mask_edges.append(bool(pred(float(xi), float(yi))) and bool(pred(float(xj), float(yj))))
        group_edges = all_bnd_edges[np.array(mask_edges, dtype=bool)]
        edges[name] = group_edges.astype(int)

    return Boundary(nodes=nodes, edges=edges)
