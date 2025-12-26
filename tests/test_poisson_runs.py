import numpy as np
from pyfemlite.mesh.structured_tri import structured_unit_square_tri
from pyfemlite.fem.poisson2d import solve_poisson_t3

def test_poisson_mms_runs():
    pi = np.pi
    u_exact = lambda x,y: np.sin(pi*x)*np.sin(pi*y)
    f_rhs = lambda x,y: 2*(pi**2)*np.sin(pi*x)*np.sin(pi*y)

    X, T, boundary = structured_unit_square_tri(10, 10)
    u = solve_poisson_t3(X, T, 1.0, f_rhs, boundary=boundary,
                         dirichlet={"left": u_exact, "right": u_exact, "bottom": u_exact, "top": u_exact})
    assert u.shape[0] == X.shape[0]
