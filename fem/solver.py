"""
HeatConfig / HeatSolver: the numerical kernel the Simulation Agent
calls (mirrors the base paper's `run_simulation` tool + DOLFINx
runner, but implemented from scratch here).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional
import time
import numpy as np
from scipy.sparse.linalg import spsolve, splu
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.fields import AliasChoices

from .mesh import Mesh2D, rectangle_mesh
from .assembly import (assemble_KM, apply_neumann, apply_robin, apply_dirichlet,
                        build_dirichlet_operator, apply_dirichlet_rhs)
from .linear_solvers import solve_dense

BCType = Literal["dirichlet", "neumann", "robin", "insulated"]
_BC_FIELDS = {"side", "type", "value", "h", "T_inf"}


@dataclass
class BoundaryCondition:
    """Kept as a dataclass so positional construction BoundaryCondition(side, type, value)
    works throughout the codebase unchanged."""
    side: str                  # 'left' | 'right' | 'top' | 'bottom'
    type: str                  # BCType
    value: float = 0.0         # Dirichlet temp or Neumann flux
    h: float = 0.0             # Robin convection coefficient
    T_inf: float = 0.0         # Robin ambient temperature


class HeatConfig(BaseModel):
    """Pydantic model — unknown LLM fields are silently dropped; 'bcs' accepts
    common aliases; BC dicts are converted to BoundaryCondition dataclasses."""
    model_config = ConfigDict(
        extra="ignore",
        populate_by_name=True,
        arbitrary_types_allowed=True,   # allows Q_func callable and BC dataclasses
    )

    Lx: float = 1.0
    Ly: float = 1.0
    Nx: int = 20
    Ny: int = 20
    k: float = 50.0
    rho: float = 7850.0
    cp: float = 490.0
    Q: float = 0.0
    bcs: List[Any] = Field(
        default_factory=list,
        validation_alias=AliasChoices(
            "bcs", "boundary_conditions", "boundaries",
            "bc_list", "bcs_list", "boundary_conds",
        ),
    )
    T_init: float = 293.15
    dt: float = 1.0
    t_end: float = 0.0
    linear_solver: str = "scipy"
    theta: float = 1.0
    Q_func: Any = Field(default=None, exclude=True)

    @field_validator("bcs", mode="before")
    @classmethod
    def _coerce_bcs(cls, v):
        """Convert each BC dict (from LLM JSON) into a BoundaryCondition dataclass.
        Applies comprehensive key aliases so Gemini/OpenAI naming variants are accepted;
        skips any BC entry that still lacks 'side' or 'type' after normalisation."""
        _ALIASES: dict = {
            # side
            "boundary": "side", "wall": "side", "face": "side",
            "location": "side", "position": "side", "direction": "side",
            "boundary_side": "side", "side_name": "side",
            # type
            "bc_type": "type", "condition": "type", "condition_type": "type",
            "boundary_type": "type", "kind": "type", "bc_kind": "type",
            "constraint_type": "type",
            # value
            "temperature": "value", "temp": "value", "T": "value",
            "flux": "value", "heat_flux": "value", "T_value": "value",
            "val": "value", "magnitude": "value",
            # h
            "conv_coeff": "h", "h_conv": "h", "convection": "h",
            "heat_transfer_coeff": "h", "htc": "h",
            # T_inf
            "ambient": "T_inf", "T_ambient": "T_inf", "T_infinity": "T_inf",
            "ambient_temp": "T_inf", "T_env": "T_inf", "T_far": "T_inf",
        }
        result = []
        for bc in (v or []):
            if isinstance(bc, BoundaryCondition):
                result.append(bc)
                continue
            if not isinstance(bc, dict):
                continue
            # Normalise key names
            norm: dict = {}
            for k, val in bc.items():
                norm[_ALIASES.get(k, k)] = val
            # Only keep recognised fields
            filtered = {k: norm[k] for k in _BC_FIELDS if k in norm}
            if "side" not in filtered or "type" not in filtered:
                continue   # skip malformed BC rather than crash
            try:
                result.append(BoundaryCondition(**filtered))
            except TypeError:
                continue
        return result


@dataclass
class HeatResult:
    mesh: Mesh2D
    T: np.ndarray
    T_history: list
    solve_time_s: float
    dofs: int
    max_stable_dt: Optional[float] = None
    warnings: list = field(default_factory=list)


def _build_system(mesh: Mesh2D, cfg: HeatConfig):
    Q_source = cfg.Q_func if cfg.Q_func is not None else cfg.Q
    K, M, F = assemble_KM(mesh, cfg.k, cfg.rho, cfg.cp, Q_source)
    for bc in cfg.bcs:
        if bc.type == "neumann":
            F = apply_neumann(mesh, F, bc.side, bc.value)
        elif bc.type == "robin":
            K, F = apply_robin(mesh, K, F, bc.side, bc.h, bc.T_inf)
        # 'insulated' => zero-flux => no contribution needed
    return K, M, F


def _dirichlet_map(mesh: Mesh2D, cfg: HeatConfig) -> dict:
    dof_values = {}
    for bc in cfg.bcs:
        if bc.type == "dirichlet":
            for i in mesh.boundary_nodes(bc.side):
                dof_values[int(i)] = bc.value
    return dof_values


def _solve_linear(A, b, method: str):
    if method == "scipy":
        return spsolve(A.tocsr(), b)
    else:
        return solve_dense(A.toarray(), b, method=method)


def solve_heat(cfg: HeatConfig) -> HeatResult:
    t0 = time.time()
    mesh = rectangle_mesh(cfg.Lx, cfg.Ly, cfg.Nx, cfg.Ny)
    K, M, F = _build_system(mesh, cfg)
    dirichlet = _dirichlet_map(mesh, cfg)

    warnings = []
    max_dt = None

    if cfg.t_end <= 0.0:
        # Steady state: K T = F, with Dirichlet BCs applied.
        A, b = apply_dirichlet(K, F, dirichlet)
        T = _solve_linear(A, b, cfg.linear_solver)
        history = [(0.0, T.copy())]
    else:
        # Backward Euler: (M/dt + K) T^{n+1} = M/dt T^n + F
        T = np.full(mesh.n_nodes, cfg.T_init)
        history = [(0.0, T.copy())]

        # Spectral diagnostic: explicit-Euler stability bound 2/lambda_max,
        # using the correct generalized eigenproblem K_ff v = lam M_ff v
        # restricted to free (non-Dirichlet) DOFs.  Informational only;
        # Backward Euler is unconditionally stable for this problem.
        try:
            from scipy.sparse.linalg import eigsh as _eigsh
            free = np.array([i for i in range(mesh.n_nodes) if i not in dirichlet],
                            dtype=int)
            if len(free) >= 2:
                K_ff = K[np.ix_(free, free)]
                M_ff = M[np.ix_(free, free)]
                lam_vals = _eigsh(K_ff, k=1, M=M_ff, which="LM",
                                  return_eigenvectors=False, tol=1e-6)
                lam_max_spectral = float(lam_vals[0])
                if lam_max_spectral > 0:
                    max_dt = 2.0 / lam_max_spectral
        except Exception:
            pass

        # General theta-method:
        #   (M/dt + θK) T^{n+1} = (M/dt - (1-θ)K) T^n + F
        # θ=1.0 → Backward Euler (unconditionally stable, default)
        # θ=0.5 → Crank-Nicolson (2nd-order in time)
        # θ=0.0 → Forward Euler (conditionally stable: dt ≤ 2/λ_max)
        theta = cfg.theta

        def _step_matrices(dt_step):
            A_lhs = (M / dt_step) + theta * K
            A_rhs = (M / dt_step) - (1.0 - theta) * K  # applied to T^n
            A_elim, corr, d_idx, d_vals = build_dirichlet_operator(A_lhs, dirichlet)
            return A_rhs, A_elim, corr, d_idx, d_vals

        def _advance(T_in, A_rhs, A_elim, corr, d_idx, d_vals, factorized):
            b = A_rhs @ T_in + F
            b = apply_dirichlet_rhs(b, corr, d_idx, d_vals)
            return (factorized.solve(b) if factorized is not None
                    else solve_dense(A_elim.toarray(), b, method=cfg.linear_solver))

        # Use floor-division + a shortened final step so the simulation
        # always reaches exactly t_end regardless of how dt divides t_end.
        n_full = int(cfg.t_end / cfg.dt)
        dt_rem = cfg.t_end - n_full * cfg.dt

        if n_full > 0:
            A_rhs, A_elim, corr, d_idx, d_vals = _step_matrices(cfg.dt)
            factorized = splu(A_elim.tocsc()) if cfg.linear_solver == "scipy" else None
            for step in range(n_full):
                T = _advance(T, A_rhs, A_elim, corr, d_idx, d_vals, factorized)
                history.append(((step + 1) * cfg.dt, T.copy()))

        # Shortened final step to cover any remaining time.
        if dt_rem > 1e-12 * cfg.dt:
            A_rhs_f, A_elim_f, corr_f, d_idx_f, d_vals_f = _step_matrices(dt_rem)
            fact_f = splu(A_elim_f.tocsc()) if cfg.linear_solver == "scipy" else None
            T = _advance(T, A_rhs_f, A_elim_f, corr_f, d_idx_f, d_vals_f, fact_f)
            history.append((cfg.t_end, T.copy()))

        # Fallback: if nothing ran (extremely small t_end), do one full step.
        if len(history) == 1:
            A_rhs, A_elim, corr, d_idx, d_vals = _step_matrices(cfg.dt)
            fact = splu(A_elim.tocsc()) if cfg.linear_solver == "scipy" else None
            T = _advance(T, A_rhs, A_elim, corr, d_idx, d_vals, fact)
            history.append((cfg.dt, T.copy()))

    if cfg.Nx < 10 or cfg.Ny < 10:
        warnings.append("COARSE_MESH_2D: nx or ny < 10")
    if not dirichlet and all(bc.type in ("neumann", "insulated") for bc in cfg.bcs):
        warnings.append("NO_BOUNDARY_CONDITIONS or pure-Neumann system (no reference point)")

    return HeatResult(
        mesh=mesh, T=T, T_history=history,
        solve_time_s=time.time() - t0,
        dofs=mesh.n_nodes, max_stable_dt=max_dt, warnings=warnings,
    )
