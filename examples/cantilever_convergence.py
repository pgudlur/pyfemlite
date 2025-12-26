import numpy as np
import matplotlib.pyplot as plt

from pyfemlite.mesh.structured_tri import structured_unit_square_tri
from pyfemlite.fem.elasticity2d import solve_elasticity_t3
from pyfemlite.fem.materials import D_plane_stress
from pyfemlite.post.beam_verify import tip_deflection_right_edge, euler_bernoulli_tip_deflection_end_traction

def run_case(nx: int, ny: int, L: float, H: float, E: float, nu: float, T0: float):
    X, T, boundary = structured_unit_square_tri(nx, ny)
    X[:, 0] *= L
    X[:, 1] *= H

    D = D_plane_stress(E, nu)
    body = lambda x, y: (0.0, 0.0)
    clamp = lambda x, y: (0.0, 0.0)
    tr_right = lambda x, y: (0.0, -T0)

    u = solve_elasticity_t3(
        X, T, D,
        body_force=body,
        boundary=boundary,
        dirichlet={"left": clamp},
        traction={"right": tr_right},
    )
    U = u.reshape(-1, 2)
    delta_fem = tip_deflection_right_edge(X, U, boundary, use_abs=True)
    return delta_fem

def main():
    # Beam parameters (match cantilever example)
    L, H = 10.0, 1.0
    E, nu = 210e9, 0.30
    T0 = 1e6

    delta_eb = euler_bernoulli_tip_deflection_end_traction(T0=T0, L=L, H=H, E=E)

    # Mesh refinement levels
    nxs = [20, 40, 80, 120, 160]
    # Keep aspect ratio roughly constant
    nys = [max(2, nx // 10) for nx in nxs]

    hs = []
    errs = []
    deltas = []

    for nx, ny in zip(nxs, nys):
        delta_fem = run_case(nx, ny, L, H, E, nu, T0)
        err = abs(delta_fem - delta_eb) / delta_eb
        h = L / nx  # characteristic mesh size along length
        hs.append(h)
        errs.append(err)
        deltas.append(delta_fem)
        print(f"nx={nx:4d}, ny={ny:4d}, h={h:.4e}, delta_fem={delta_fem:.6e}, rel_err={err:.6%}")

    # Save results table
    import csv
    with open("cantilever_convergence.csv", "w", newline="", encoding="utf-8") as f:
        wtr = csv.writer(f)
        wtr.writerow(["nx", "ny", "h", "delta_fem", "delta_eb", "rel_error"])
        for nx, ny, h, d, e in zip(nxs, nys, hs, deltas, errs):
            wtr.writerow([nx, ny, h, d, delta_eb, e])
    print("Wrote cantilever_convergence.csv")

    # Convergence plot (log-log): relative error vs h
    plt.figure()
    plt.loglog(hs, errs, marker="o")
    plt.gca().invert_xaxis()
    plt.xlabel("h = L/nx")
    plt.ylabel("Relative error in tip deflection")
    plt.title("Cantilever tip deflection convergence (FEM vs Euler–Bernoulli)")
    plt.grid(True, which="both")

    outdir = "docs/img"
    import os
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "cantilever_tip_deflection_convergence.png")
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    print(f"Wrote {outpath}")

if __name__ == "__main__":
    main()
