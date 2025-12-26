from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix

def apply_dirichlet(K: csr_matrix, f: np.ndarray, dbc: dict[int, float]):
    K = K.tolil()
    for dof, val in dbc.items():
        K.rows[dof] = [dof]
        K.data[dof] = [1.0]
        f[dof] = val
    return K.tocsr(), f
