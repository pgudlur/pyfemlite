import numpy as np
from pyfemlite.mesh.structured_tri import structured_unit_square_tri
from pyfemlite.fem.poisson2d import solve_poisson_t3
from pyfemlite.io.vtk import write_vtk_unstructured_tri

pi = np.pi
u_exact = lambda x, y: np.sin(pi * x) * np.sin(pi * y)
f_rhs   = lambda x, y: 2 * (pi**2) * np.sin(pi * x) * np.sin(pi * y)

nx, ny = 40, 40
X, T, boundary = structured_unit_square_tri(nx, ny)

boundary.validate(X, T, strict=True)
print(boundary.summary())

u = solve_poisson_t3(
    X, T,
    kappa=1.0,
    rhs_func=f_rhs,
    boundary=boundary,
    dirichlet={"left": u_exact, "right": u_exact, "bottom": u_exact, "top": u_exact},
    validate_boundary=True,
)

write_vtk_unstructured_tri("poisson_mms_named_bc.vtk", X, T, {"u": u})
print("Wrote poisson_mms_named_bc.vtk (open in ParaView).")
