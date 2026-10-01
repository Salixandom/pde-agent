"""
Verification & Validation study: reproduces the base paper's
Section 5 methodology (three closed-form benchmark cases, mesh
refinement N in {8,16,32,64}, L2 error via quadrature, convergence
rate via log-log linear regression) against THIS project's
from-scratch FEM solver + custom linear solvers, instead of
validating DOLFINx.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .mesh import rectangle_mesh, Mesh2D
from .assembly import assemble_KM, apply_dirichlet, build_dirichlet_operator, apply_dirichlet_rhs
from .linear_solvers import solve_dense
from scipy.sparse.linalg import spsolve, splu

from .quadrature import TRI_QUAD  # shared 6-point degree-4 triangle quadrature


def l2_error(mesh: Mesh2D, T_nodal: np.ndarray, exact_fn) -> float:
    """Integrated L2 error ||T_h - T_exact||_2 over the whole mesh,
    using P1 interpolation inside each element (barycentric = shape
    functions) evaluated at quadrature points -- avoids the
    spurious nodal-exactness artifact the base paper also flags."""
    total = 0.0
    for tri in mesh.triangles:
        p1, p2, p3 = mesh.nodes[tri]
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        area2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
        A = 0.5 * abs(area2)
        Tn = T_nodal[tri]
        acc = 0.0
        for l1, l2, l3, w in TRI_QUAD:
            x = l1 * x1 + l2 * x2 + l3 * x3
            y = l1 * y1 + l2 * y2 + l3 * y3
            Th = l1 * Tn[0] + l2 * Tn[1] + l3 * Tn[2]
            Te = exact_fn(x, y)
            acc += w * (Th - Te) ** 2
        total += A * acc
    return float(np.sqrt(total))


def _steady_solve(mesh, K, F, dirichlet, method="scipy"):
    A, b = apply_dirichlet(K, F, dirichlet)
    if method == "scipy":
        return spsolve(A.tocsr(), b)
    return solve_dense(A.toarray(), b, method=method)


# ---------------------------------------------------------------- #
# Case 1: 2D steady-state linear profile, T(x,y) = x
# ---------------------------------------------------------------- #
def case1_linear_profile(N: int, method="scipy"):
    mesh = rectangle_mesh(1.0, 1.0, N, N)
    K, M, F = assemble_KM(mesh, k=1.0, rho=1.0, cp=1.0, Q=0.0)
    dirichlet = {int(i): 0.0 for i in mesh.boundary_nodes("left")}
    dirichlet.update({int(i): 1.0 for i in mesh.boundary_nodes("right")})
    T = _steady_solve(mesh, K, F, dirichlet, method)
    err = l2_error(mesh, T, lambda x, y: x)
    return mesh.n_nodes, err


# ---------------------------------------------------------------- #
# Case 2: 2D transient Fourier-mode decay
#   T0 = sin(pi x) sin(pi y), homogeneous Dirichlet, alpha = k/(rho*cp)
#   exact: T(x,y,t) = exp(-2*alpha*pi^2*t) * sin(pi x) sin(pi y)
# ---------------------------------------------------------------- #
def case2_fourier_decay(N: int, method="scipy", dt: float = 1e-4, t_end: float = 1e-2):
    k, rho, cp = 1.0, 1.0, 1.0
    alpha = k / (rho * cp)
    mesh = rectangle_mesh(1.0, 1.0, N, N)
    K, M, F0 = assemble_KM(mesh, k=k, rho=rho, cp=cp, Q=0.0)

    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    T = np.sin(np.pi * x) * np.sin(np.pi * y)

    dirichlet = {}
    for side in ("left", "right", "top", "bottom"):
        for i in mesh.boundary_nodes(side):
            dirichlet[int(i)] = 0.0

    n_steps = max(1, int(round(t_end / dt)))
    A_const = (M / dt) + K
    A_elim, correction, didx, dvals = build_dirichlet_operator(A_const, dirichlet)
    factorized = splu(A_elim.tocsc()) if method == "scipy" else None
    for _ in range(n_steps):
        b = (M @ (T / dt)) + F0
        b = apply_dirichlet_rhs(b, correction, didx, dvals)
        if factorized is not None:
            T = factorized.solve(b)
        else:
            T = solve_dense(A_elim.toarray(), b, method=method)

    t_final = n_steps * dt
    decay = np.exp(-2 * alpha * np.pi ** 2 * t_final)

    def exact(xx, yy):
        return decay * np.sin(np.pi * xx) * np.sin(np.pi * yy)

    err = l2_error(mesh, T, exact)
    return mesh.n_nodes, err


# ---------------------------------------------------------------- #
# Case 3: 2D steady-state Poisson, constant source
#   -k lap(T) = f ; T=0 left/right (Dirichlet); zero-flux top/bottom
#   exact: T(x) = f*(x - x^2) / (2k)   [1D-like, "1D Poisson"]
# ---------------------------------------------------------------- #
def case3_constant_source(N: int, method="scipy"):
    k, f = 1.0, 1000.0
    mesh = rectangle_mesh(1.0, 1.0, N, N)
    K, M, F = assemble_KM(mesh, k=k, rho=1.0, cp=1.0, Q=f)
    dirichlet = {int(i): 0.0 for i in mesh.boundary_nodes("left")}
    dirichlet.update({int(i): 0.0 for i in mesh.boundary_nodes("right")})
    T = _steady_solve(mesh, K, F, dirichlet, method)
    err = l2_error(mesh, T, lambda x, y: f * (x - x ** 2) / (2 * k))
    return mesh.n_nodes, err


@dataclass
class ConvergenceResult:
    case: str
    Ns: list
    dofs: list
    errors: list
    rate: float


def _fit_rate(Ns, errors):
    h = 1.0 / np.array(Ns, dtype=float)
    e = np.array(errors, dtype=float)
    mask = e > 0
    if mask.sum() < 2:
        return float("nan")
    slope, _ = np.polyfit(np.log(h[mask]), np.log(e[mask]), 1)
    return float(slope)


def run_convergence_study(Ns=(8, 16, 32, 64), method="scipy") -> list:
    cases = {
        "Steady Linear Profile": case1_linear_profile,
        "Transient Fourier Decay": case2_fourier_decay,
        "Steady Poisson (const. source)": case3_constant_source,
    }
    results = []
    for name, fn in cases.items():
        dofs, errs = [], []
        for N in Ns:
            n_dof, err = fn(N, method=method)
            dofs.append(n_dof)
            errs.append(err)
        rate = _fit_rate(Ns, errs)
        results.append(ConvergenceResult(case=name, Ns=list(Ns), dofs=dofs,
                                          errors=errs, rate=rate))
    return results
