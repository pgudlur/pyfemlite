from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

def assemble_global(n_dof: int, triplets_i, triplets_j, triplets_v) -> csr_matrix:
    K = coo_matrix((triplets_v, (triplets_i, triplets_j)), shape=(n_dof, n_dof))
    return K.tocsr()

def add_local_to_triplets(edofs: np.ndarray, Ke: np.ndarray, I, J, V):
    ii, jj = np.meshgrid(edofs, edofs, indexing="ij")
    I.extend(ii.ravel().tolist())
    J.extend(jj.ravel().tolist())
    V.extend(Ke.ravel().tolist())
