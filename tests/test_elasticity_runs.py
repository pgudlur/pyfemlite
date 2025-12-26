import numpy as np
from pyfemlite.mesh.structured_tri import structured_unit_square_tri
from pyfemlite.fem.elasticity2d import solve_elasticity_t3
from pyfemlite.fem.materials import D_plane_stress

def test_elasticity_runs_with_traction():
    X, T, boundary = structured_unit_square_tri(8, 4)
    E, nu = 200e9, 0.3
    D = D_plane_stress(E, nu)

    body = lambda x,y: (0.0, 0.0)
    clamp = lambda x,y: (0.0, 0.0)
    tr = lambda x,y: (0.0, -1.0)

    u = solve_elasticity_t3(X, T, D, body_force=body, boundary=boundary,
                            dirichlet={"left": clamp}, traction={"right": tr})
    assert u.shape[0] == 2*X.shape[0]
