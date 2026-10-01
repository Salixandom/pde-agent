"""
P1 (linear triangle) finite-element assembly for the 2D heat
equation:  rho*cp * dT/dt = k * div(grad T) + Q

Weak form / semi-discrete system:  M dT/dt + K T = F
  M : consistent mass matrix   (rho*cp * int phi_i phi_j dOmega)
  K : stiffness matrix         (k * int grad(phi_i).grad(phi_j) dOmega)
  F : load vector              (source term + Neumann/Robin boundary terms)

This mirrors DOLFINx's role in the base paper (PDE-Agents), but is
implemented directly instead of delegated to an external FEM
library, so mesh assembly and the resulting linear system are
fully inspectable / extendable with the CSE401 solvers in
linear_solvers.py.
"""
from __future__ import annotations
import numpy as np
from scipy import sparse

from .mesh import Mesh2D
from .quadrature import TRI_QUAD


def _element_matrices(coords: np.ndarray, k: float, rho: float, cp: float):
    """coords: (3,2) triangle vertex coordinates."""
    (x1, y1), (x2, y2), (x3, y3) = coords
    area2 = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    A = 0.5 * abs(area2)
    if A < 1e-14:
        raise ValueError("degenerate triangle")

    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    Ke = (k / (4.0 * A)) * (np.outer(b, b) + np.outer(c, c))
    Me = (rho * cp * A / 12.0) * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype=float)
    return Ke, Me, A


def assemble_KM(mesh: Mesh2D, k: float, rho: float, cp: float, Q: float = 0.0):
    """Assemble global sparse stiffness K, mass M, and constant
    volumetric-source load vector F."""
    n = mesh.n_nodes
    K = sparse.lil_matrix((n, n))
    M = sparse.lil_matrix((n, n))
    F = np.zeros(n)

    for tri in mesh.triangles:
        coords = mesh.nodes[tri]
        Ke, Me, A = _element_matrices(coords, k, rho, cp)
        if callable(Q):
            # 6-point quadrature for spatially varying source Q(x, y)
            (x1, y1), (x2, y2), (x3, y3) = coords
            Fe = np.zeros(3)
            for l1, l2, l3, w in TRI_QUAD:
                qx = l1 * x1 + l2 * x2 + l3 * x3
                qy = l1 * y1 + l2 * y2 + l3 * y3
                q_val = Q(qx, qy)
                Fe[0] += w * q_val * l1
                Fe[1] += w * q_val * l2
                Fe[2] += w * q_val * l3
            Fe *= A
        else:
            Fe = np.full(3, Q * A / 3.0)
        for a in range(3):
            F[tri[a]] += Fe[a]
            for bidx in range(3):
                K[tri[a], tri[bidx]] += Ke[a, bidx]
                M[tri[a], tri[bidx]] += Me[a, bidx]

    return K.tocsr(), M.tocsr(), F


def apply_neumann(mesh: Mesh2D, F: np.ndarray, side: str, flux: float):
    """Add a constant inward heat flux [W/m^2] on a boundary side."""
    F = F.copy()
    edges = mesh.boundary_edges(side)
    for n0, n1 in edges:
        L = np.linalg.norm(mesh.nodes[n1] - mesh.nodes[n0])
        F[n0] += flux * L / 2.0
        F[n1] += flux * L / 2.0
    return F


def apply_robin(mesh: Mesh2D, K, F: np.ndarray, side: str, h: float, T_inf: float):
    """Add convective boundary term h*(T_inf - T) on a boundary side."""
    K = K.tolil()
    F = F.copy()
    edges = mesh.boundary_edges(side)
    local = (h / 6.0) * np.array([[2, 1], [1, 2]], dtype=float)
    for n0, n1 in edges:
        L = np.linalg.norm(mesh.nodes[n1] - mesh.nodes[n0])
        idx = [n0, n1]
        for a in range(2):
            F[idx[a]] += h * T_inf * L / 2.0
            for bidx in range(2):
                K[idx[a], idx[bidx]] += local[a, bidx] * L
    return K.tocsr(), F


def apply_dirichlet(A, b: np.ndarray, dof_values: dict):
    """Symmetric elimination of Dirichlet dofs (one-shot, steady-state
    convenience wrapper): for each fixed dof i with value v, subtract
    A[:,i]*v from b, then zero row/col i and set A[i,i]=1, b[i]=v.

    Implemented with sparse diagonal masking (no per-entry Python
    loops), so it stays fast even for large boundary sets."""
    A_elim, correction, idx, vals = build_dirichlet_operator(A, dof_values)
    b_adj = apply_dirichlet_rhs(b, correction, idx, vals)
    return A_elim, b_adj


def build_dirichlet_operator(A, dof_values: dict):
    """Precompute the Dirichlet-eliminated matrix once, plus the RHS
    correction vector, so a transient time-stepping loop only needs
    the O(n) `apply_dirichlet_rhs` call per step instead of
    re-eliminating the (unchanged) LHS matrix every step."""
    n = A.shape[0]
    idx = np.array(list(dof_values.keys()), dtype=int)
    vals = np.array(list(dof_values.values()), dtype=float)

    mask = np.ones(n)
    mask[idx] = 0.0
    Dk = sparse.diags(mask)
    Ddir = sparse.diags(1.0 - mask)

    A_csr = A.tocsr()
    correction = np.asarray(A_csr[:, idx] @ vals).ravel()
    A_elim = (Dk @ A_csr @ Dk + Ddir).tocsr()
    return A_elim, correction, idx, vals


def apply_dirichlet_rhs(b: np.ndarray, correction: np.ndarray, idx: np.ndarray,
                         vals: np.ndarray) -> np.ndarray:
    """Cheap per-step RHS-only Dirichlet application, paired with a
    matrix already eliminated by `build_dirichlet_operator`."""
    b_adj = b - correction
    b_adj[idx] = vals
    return b_adj
