import numpy as np
from pyfemlite.mesh.structured_tri import structured_unit_square_tri
from pyfemlite.fem.elasticity2d import solve_elasticity_t3
from pyfemlite.fem.materials import D_plane_stress
from pyfemlite.io.vtk import write_vtk_unstructured_tri
from pyfemlite.post.beam_verify import tip_deflection_right_edge, euler_bernoulli_tip_deflection_end_traction

L, H = 10.0, 1.0
nx, ny = 120, 12

X, T, boundary = structured_unit_square_tri(nx, ny)
X[:, 0] *= L
X[:, 1] *= H

boundary.validate(X, T, strict=True)
print(boundary.summary())

E, nu = 210e9, 0.30
D = D_plane_stress(E, nu)

body = lambda x, y: (0.0, 0.0)

clamp = lambda x, y: (0.0, 0.0)

# Uniform downward traction on right edge
T0 = 1e6
tr_right = lambda x, y: (0.0, -T0)

u = solve_elasticity_t3(
    X, T, D,
    body_force=body,
    boundary=boundary,
    dirichlet={"left": clamp},
    traction={"right": tr_right},
    validate_boundary=True,
)

U = u.reshape(-1, 2)
write_vtk_unstructured_tri("cantilever_named_bc.vtk", X, T, {"U": U})
print("Wrote cantilever_named_bc.vtk (open in ParaView).")

# Verification: tip deflection vs Euler–Bernoulli beam theory
delta_fem = tip_deflection_right_edge(X, U, boundary, use_abs=True)
delta_eb = euler_bernoulli_tip_deflection_end_traction(T0=T0, L=L, H=H, E=E)
rel_err = abs(delta_fem - delta_eb) / delta_eb

print("\nTip deflection verification (magnitude):")
print(f"  FEM  delta_tip = {delta_fem:.6e}")
print(f"  EB   delta_tip = {delta_eb:.6e}")
print(f"  Rel. error     = {rel_err:.6%}")

# Save verification numbers for easy copy/paste
import csv
with open('cantilever_tip_deflection.csv', 'w', newline='', encoding='utf-8') as f:
    wtr = csv.writer(f)
    wtr.writerow(['L', 'H', 'E', 'nu', 'T0', 'nx', 'ny', 'delta_fem', 'delta_eb', 'rel_error'])
    wtr.writerow([L, H, E, nu, T0, nx, ny, delta_fem, delta_eb, rel_err])
print("Wrote cantilever_tip_deflection.csv")
