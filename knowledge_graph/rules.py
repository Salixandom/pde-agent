"""
Pure-Python pre-run rule engine, mirroring the base paper's
Table (Section 4.2) of nine analytical checks fired before every
simulation attempt. Returns a list of (code, message) warnings;
the Simulation Agent's `check_config_warnings` tool wraps this.
"""
from __future__ import annotations


def check_config(cfg) -> list[tuple[str, str]]:
    warnings = []

    # Check material properties first to avoid ZeroDivisionError downstream
    if cfg.k <= 0 or cfg.rho <= 0 or cfg.cp <= 0:
        warnings.append(("INVALID_MATERIAL_PROPS", "k, rho, or cp is <= 0."))
    elif cfg.t_end > 0:
        alpha = cfg.k / (cfg.rho * cfg.cp)
        h = min(cfg.Lx / cfg.Nx, cfg.Ly / cfg.Ny)
        explicit_limit = h ** 2 / (2 * alpha)
        if cfg.dt > 10 * explicit_limit:
            warnings.append((
                "LARGE_DT_RELATIVE_DIFFUSION",
                f"dt={cfg.dt:.3g}s is >10x the explicit diffusion limit "
                f"({explicit_limit:.3g}s) -- fine for Backward Euler "
                f"(unconditionally stable) but may blur transient features.",
            ))

    if cfg.Nx < 10 or cfg.Ny < 10:
        warnings.append(("COARSE_MESH_2D", f"Nx={cfg.Nx}, Ny={cfg.Ny} below the "
                                            f"recommended 10x10 resolution."))

    if cfg.t_end > 0 and cfg.t_end < 5 * cfg.dt:
        warnings.append(("SHORT_SIMULATION", "t_end is less than 5 timesteps; "
                                              "transient behaviour may be under-resolved."))

    # INCONSISTENT_IC: initial temperature far from Dirichlet boundary temperatures
    # (mirrors paper's Table §4 rule: |u_init - T_bc_min| > 100 K)
    if cfg.t_end > 0:
        bc_dirichlet_temps = [bc.value for bc in cfg.bcs if bc.type == "dirichlet"]
        if bc_dirichlet_temps:
            min_bc = min(bc_dirichlet_temps)
            if abs(cfg.T_init - min_bc) > 100:
                warnings.append((
                    "INCONSISTENT_IC",
                    f"T_init={cfg.T_init:.1f} K differs from the minimum Dirichlet BC "
                    f"({min_bc:.1f} K) by {abs(cfg.T_init - min_bc):.1f} K (>100 K) -- "
                    f"the transient solution will start with a large thermal jump.",
                ))

    if not cfg.bcs:
        warnings.append(("NO_BOUNDARY_CONDITIONS", "No boundary conditions specified."))
    elif all(bc.type in ("neumann", "insulated") for bc in cfg.bcs):
        warnings.append(("NO_BOUNDARY_CONDITIONS", "Pure Neumann/insulated system has "
                                                     "no reference temperature -- solution "
                                                     "is only defined up to a constant."))

    return warnings
