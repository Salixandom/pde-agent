"""
Generates the report-ready artifacts for the course project:
  outputs/convergence_study.png      -- V&V spatial convergence (paper Fig. 3 style)
  outputs/solver_comparison.png      -- scipy vs Gauss vs LU accuracy + timing
  outputs/simulation_gallery.png     -- 6-panel temperature field gallery (paper Fig. 4 style)
  outputs/kg_structure.png           -- NetworkX KG graph (paper Fig. 2 style)
  outputs/solver_validation.txt      -- numeric V&V table
  outputs/agent_demo_transcript.txt  -- LangGraph supervisor pipeline demo
"""
import os
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load .env so GOOGLE_API_KEY / ANTHROPIC_API_KEY reach get_llm()
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from fem.verification import run_convergence_study, case3_constant_source, case2_fourier_decay
from fem.solver import HeatConfig, BoundaryCondition, solve_heat
from agents.context import AppContext
from agents.llm_factory import get_llm
from agents.supervisor import run_request, stream_request

OUT = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT, exist_ok=True)


def convergence_plot():
    """Reproduce paper Fig. 3: log-log L2 error vs mesh size for all 3 benchmark cases,
    with all three linear solvers shown for the Poisson case (our extension)."""
    results_scipy = run_convergence_study(Ns=(8, 16, 32, 64, 128), method="scipy")

    fig, ax = plt.subplots(figsize=(7, 5))
    # Linear profile errors are at machine precision — label as exact, not a fitted slope
    markers = {
        "Steady Linear Profile":          ("^-", "C2", "exact"),
        "Transient Fourier Decay":        ("o-", "C0", None),
        "Steady Poisson (const. source)": ("s-", "C1", None),
    }
    for r in results_scipy:
        if r.case not in markers:
            continue
        h = 1.0 / np.array(r.Ns, dtype=float)
        mk, col, override = markers[r.case]
        rate_label = override if override else f"rate={r.rate:.2f}"
        ax.loglog(h, r.errors, mk, color=col, label=f"{r.case} (~ε_mach)" if override else f"{r.case} (rate={r.rate:.2f})")

    h_ref = 1.0 / np.array([8, 64], dtype=float)
    ax.loglog(h_ref, 0.5 * h_ref ** 2, "k--", label=r"$\mathcal{O}(h^2)$ reference")
    ax.set_xlabel("Mesh size h = 1/N")
    ax.set_ylabel(r"$\|e\|_{L^2}$")
    ax.set_title("Spatial convergence study — custom P1 FEM (paper §5 style)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "convergence_study.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "solver_validation.txt"), "w") as f:
        f.write("Verification & Validation summary\n")
        f.write("=" * 60 + "\n")
        for r in results_scipy:
            is_exact = max(r.errors) < 1e-10
            rate_str = "exact (machine precision)" if is_exact else f"{r.rate:.3f}"
            f.write(f"\n{r.case} : rate = {rate_str}\n")
            for N, dof, err in zip(r.Ns, r.dofs, r.errors):
                f.write(f"   N={N:3d}  dofs={dof:5d}  L2_err={err:.4e}\n")

        f.write("\n\nCustom linear solver cross-validation (N=16 Poisson case)\n")
        f.write("-" * 60 + "\n")
        for method in ("scipy", "gauss", "lu"):
            dofs, err = case3_constant_source(N=16, method=method)
            f.write(f"   {method:6s}  dofs={dofs}  L2_err={err:.6e}\n")
    print("Wrote convergence_study.png and solver_validation.txt")


def solver_comparison_plot():
    """New graph: compare scipy / Gauss / LU across mesh sizes on the Poisson case.
    Shows both L2 error (accuracy) and wall-clock time (cost) -- our custom-solver
    extension contribution."""
    Ns = [8, 16, 32]  # 64 is slow for dense custom solvers
    methods = ["scipy", "gauss", "lu"]
    labels = {"scipy": "SciPy (sparse)", "gauss": "Custom Gauss", "lu": "Custom LU"}
    colors = {"scipy": "C0", "gauss": "C1", "lu": "C2"}

    errors = {m: [] for m in methods}
    times = {m: [] for m in methods}

    for N in Ns:
        for m in methods:
            t0 = time.perf_counter()
            _, err = case3_constant_source(N=N, method=m)
            elapsed = time.perf_counter() - t0
            errors[m].append(err)
            times[m].append(elapsed)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    h = 1.0 / np.array(Ns, dtype=float)

    for m in methods:
        ax1.loglog(h, errors[m], "o-", color=colors[m], label=labels[m])
    h_ref = np.array([h[0], h[-1]])
    ax1.loglog(h_ref, 0.4 * h_ref ** 2, "k--", label=r"$\mathcal{O}(h^2)$")
    ax1.set_xlabel("h = 1/N")
    ax1.set_ylabel(r"$\|e\|_{L^2}$")
    ax1.set_title("L2 Error: scipy vs custom solvers")
    ax1.legend(fontsize=9)
    ax1.grid(True, which="both", alpha=0.3)

    x = np.arange(len(Ns))
    width = 0.25
    for i, m in enumerate(methods):
        ax2.bar(x + i * width, times[m], width, label=labels[m], color=colors[m], alpha=0.8)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels([f"N={N}" for N in Ns])
    ax2.set_ylabel("Wall time (s)")
    ax2.set_title("Solve time: scipy vs custom solvers")
    ax2.legend(fontsize=9)

    fig.suptitle("Custom Gauss/LU vs SciPy sparse — Poisson benchmark", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "solver_comparison.png"), dpi=150)
    plt.close(fig)
    print("Wrote solver_comparison.png")


def simulation_gallery():
    """Reproduce paper Fig. 4: 6-panel temperature field gallery covering
    Dirichlet, mixed BCs, volumetric source, Robin, and transient cases."""
    cases = [
        ("(a) Steady Dirichlet — copper\n373 K left, 273 K right", HeatConfig(
            k=385.0, rho=8960.0, cp=385.0, Nx=30, Ny=30,
            bcs=[BoundaryCondition("left", "dirichlet", 373.15),
                 BoundaryCondition("right", "dirichlet", 273.15)])),
        ("(b) Mixed BCs — steel\nDirichlet+Neumann+Robin+Q", HeatConfig(
            k=50.0, rho=7850.0, cp=490.0, Nx=30, Ny=30, Q=5000.0,
            bcs=[BoundaryCondition("left", "dirichlet", 300.0),
                 BoundaryCondition("top", "neumann", 2000.0),
                 BoundaryCondition("right", "robin", h=25.0, T_inf=293.0),
                 BoundaryCondition("bottom", "insulated")])),
        ("(c) Robin convection — titanium\nLeft 573 K, right convective h=50", HeatConfig(
            k=6.7, rho=4430.0, cp=526.0, Nx=30, Ny=30,
            bcs=[BoundaryCondition("left", "dirichlet", 573.15),
                 BoundaryCondition("right", "robin", h=50.0, T_inf=293.15),
                 BoundaryCondition("top", "insulated"),
                 BoundaryCondition("bottom", "insulated")])),
        ("(d) Volumetric source — aluminium\nQ=10 kW/m³, insulated sides", HeatConfig(
            k=205.0, rho=2700.0, cp=900.0, Nx=30, Ny=30, Q=10000.0,
            bcs=[BoundaryCondition("left", "dirichlet", 293.15),
                 BoundaryCondition("right", "dirichlet", 293.15),
                 BoundaryCondition("top", "insulated"),
                 BoundaryCondition("bottom", "insulated")])),
        ("(e) Transient t=20s — aluminium\nSudden 400 K left boundary", HeatConfig(
            k=205.0, rho=2700.0, cp=900.0, Nx=30, Ny=30,
            bcs=[BoundaryCondition("left", "dirichlet", 400.0),
                 BoundaryCondition("right", "dirichlet", 293.0)],
            T_init=293.0, dt=0.5, t_end=20.0)),
        ("(f) Transient t=5s — stainless steel\nTop Neumann flux 3 kW/m²", HeatConfig(
            k=16.3, rho=8000.0, cp=500.0, Nx=30, Ny=30,
            bcs=[BoundaryCondition("left", "dirichlet", 293.15),
                 BoundaryCondition("right", "dirichlet", 293.15),
                 BoundaryCondition("top", "neumann", 3000.0),
                 BoundaryCondition("bottom", "insulated")],
            T_init=293.15, dt=0.1, t_end=5.0)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (title, cfg) in zip(axes.flat, cases):
        res = solve_heat(cfg)
        mesh = res.mesh
        tpc = ax.tricontourf(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles, res.T,
                              levels=20, cmap="inferno")
        fig.colorbar(tpc, ax=ax, label="T (K)")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
    fig.suptitle("Representative temperature fields — P1 FEM (paper Fig. 4 style)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "simulation_gallery.png"), dpi=150)
    plt.close(fig)
    print("Wrote simulation_gallery.png (6 panels)")


def kg_structure_plot():
    """Visualize the populated NetworkX KG (paper Fig. 2 style): material nodes,
    run nodes, and KnownIssue nodes with their relationship edges."""
    import networkx as nx
    from agents.context import AppContext
    from agents.tools import make_simulation_tools

    # Build a small populated KG for illustration
    ctx = AppContext.build(db_path=":memory:")
    tools = {t.name: t for t in make_simulation_tools(ctx)}

    # Run a few simulations to create Run nodes
    for mat, left_t, right_t in [("copper", 373.15, 273.15),
                                   ("steel", 400.0, 300.0),
                                   ("aluminium", 350.0, 293.15)]:
        mat_data = ctx.kg.get_material(mat)
        cfg = json.dumps({
            "Lx": 1.0, "Ly": 1.0, "Nx": 8, "Ny": 8,
            "k": mat_data.k, "rho": mat_data.rho, "cp": mat_data.cp, "Q": 0.0,
            "bcs": [{"side": "left", "type": "dirichlet", "value": left_t,
                      "h": 0.0, "T_inf": 0.0},
                     {"side": "right", "type": "dirichlet", "value": right_t,
                      "h": 0.0, "T_inf": 0.0}],
            "T_init": 293.15, "dt": 1.0, "t_end": 0.0, "linear_solver": "scipy",
        })
        tools["run_simulation"].func(cfg, f"{mat} plate simulation", mat)

    G = ctx.kg.g
    node_colors, node_sizes, labels = [], [], {}
    for node, data in G.nodes(data=True):
        kind = data.get("kind", "")
        short = node.split(":", 1)[1] if ":" in node else node
        labels[node] = short[:14]
        if kind == "Material":
            node_colors.append("#4C9BE8")
            node_sizes.append(800)
        elif kind == "Run":
            node_colors.append("#5CB85C")
            node_sizes.append(600)
        elif kind == "KnownIssue":
            node_colors.append("#E8844C")
            node_sizes.append(700)
        else:
            node_colors.append("#AAAAAA")
            node_sizes.append(400)

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                            ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.4, arrows=True,
                            arrowsize=15, edge_color="#888888")
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    from matplotlib.patches import Patch
    legend = [Patch(color="#4C9BE8", label="Material"),
               Patch(color="#5CB85C", label="Run"),
               Patch(color="#E8844C", label="KnownIssue")]
    ax.legend(handles=legend, loc="upper left")
    ax.set_title("Knowledge Graph structure (paper Fig. 2 style)\n"
                  "NetworkX + TF-IDF warm-start adaptation of Neo4j + HNSW", fontsize=10)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kg_structure.png"), dpi=150)
    plt.close(fig)
    print("Wrote kg_structure.png")


def stability_comparison_plot():
    """Show Forward Euler (theta=0) blowing up above the CFL limit vs
    Backward Euler (theta=1) staying stable -- directly validates the
    spectral eigenvalue diagnostic (max_stable_dt = 2/lambda_max)."""
    from fem.mesh import rectangle_mesh

    # Use small mesh so FE stability limit is large enough to visualise
    N, t_end = 6, 0.05
    bcs = [BoundaryCondition("left",   "dirichlet", 1.0),
           BoundaryCondition("right",  "dirichlet", 0.0),
           BoundaryCondition("top",    "insulated"),
           BoundaryCondition("bottom", "insulated")]

    # Get spectral stability limit from BE run
    cfg_probe = HeatConfig(Lx=1.0, Ly=1.0, Nx=N, Ny=N, k=1.0, rho=1.0, cp=1.0,
                            bcs=bcs, T_init=0.0, dt=1e-4, t_end=t_end, theta=1.0)
    res_probe = solve_heat(cfg_probe)
    max_dt = res_probe.max_stable_dt or 0.002

    dt_safe     = 0.5 * max_dt
    dt_unstable = 2.2 * max_dt

    mesh = rectangle_mesh(1.0, 1.0, N, N)
    center_idx = int(np.argmin(
        (mesh.nodes[:, 0] - 0.5)**2 + (mesh.nodes[:, 1] - 0.5)**2))

    configs = [
        ("BE θ=1, dt=2.2×limit (stable)",  1.0, dt_unstable, "C0", "-"),
        ("FE θ=0, dt=0.5×limit (stable)",  0.0, dt_safe,     "C2", "--"),
        ("FE θ=0, dt=2.2×limit (unstable)", 0.0, dt_unstable, "C3", "-"),
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for label, theta, dt, col, ls in configs:
        cfg = HeatConfig(Lx=1.0, Ly=1.0, Nx=N, Ny=N, k=1.0, rho=1.0, cp=1.0,
                          bcs=bcs, T_init=0.0, dt=dt, t_end=t_end, theta=theta)
        res = solve_heat(cfg)
        t_vals = [s[0] for s in res.T_history]
        T_c    = [np.clip(s[1][center_idx], -5, 5) for s in res.T_history]
        ax1.plot(t_vals, T_c, ls, color=col, label=label)

    ax1.axhline(0, color="black", linewidth=0.5, alpha=0.4)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("T at center (K)")
    ax1.set_title(f"Center-point temperature\n(max_stable_dt = {max_dt:.4f} s)")
    ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

    # Panel 2: FE final field with unstable dt (show spatial blowup)
    cfg_fe = HeatConfig(Lx=1.0, Ly=1.0, Nx=N, Ny=N, k=1.0, rho=1.0, cp=1.0,
                         bcs=bcs, T_init=0.0, dt=dt_unstable, t_end=t_end, theta=0.0)
    res_fe = solve_heat(cfg_fe)
    T_clipped = np.clip(res_fe.T, -5, 5)
    tpc = ax2.tricontourf(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles,
                           T_clipped, levels=20, cmap="RdBu_r")
    fig.colorbar(tpc, ax=ax2, label="T (K, clipped to ±5)")
    ax2.set_title(f"FE θ=0 final field (dt={dt_unstable:.4f} s > limit)\nSpatial oscillations = CFL violation")
    ax2.set_xlabel("x"); ax2.set_ylabel("y"); ax2.set_aspect("equal")

    fig.suptitle("Stability comparison: Forward Euler vs Backward Euler\n"
                 "Validates spectral eigenvalue diagnostic (max_stable_dt = 2/λ_max)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "stability_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote stability_comparison.png  (max_stable_dt={max_dt:.5f} s)")


def gaussian_source_demo():
    """Demonstrate spatially varying Q(x,y): Gaussian heat source (laser-spot model).
    Shows the radially symmetric temperature hotspot the constant-Q formulation cannot produce."""
    import math

    Q_peak = 5e6   # W/m³
    x0, y0 = 0.3, 0.7
    sigma  = 0.08

    def Q_gaussian(x, y):
        return Q_peak * math.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))

    bcs = [BoundaryCondition(side, "dirichlet", 293.15)
           for side in ("left", "right", "top", "bottom")]

    cfg_gauss = HeatConfig(
        Lx=1.0, Ly=1.0, Nx=40, Ny=40, k=205.0, rho=2700.0, cp=900.0,
        Q=0.0, Q_func=Q_gaussian,
        bcs=bcs,
    )
    cfg_const = HeatConfig(
        Lx=1.0, Ly=1.0, Nx=40, Ny=40, k=205.0, rho=2700.0, cp=900.0,
        Q=Q_peak * 2 * math.pi * sigma**2,  # same total power
        bcs=bcs,
    )

    res_g = solve_heat(cfg_gauss)
    res_c = solve_heat(cfg_const)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, res, title in [
        (axes[0], res_g, f"Gaussian source Q(x,y)\nPeak={Q_peak:.0e} W/m³ at ({x0},{y0}), σ={sigma}"),
        (axes[1], res_c, "Equivalent constant source\n(same total power, Q uniform)"),
    ]:
        mesh = res.mesh
        tpc = ax.tricontourf(mesh.nodes[:, 0], mesh.nodes[:, 1], mesh.triangles,
                              res.T, levels=20, cmap="inferno")
        fig.colorbar(tpc, ax=ax, label="T (K)")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_aspect("equal")

    fig.suptitle("Spatially varying Q(x,y) — Gaussian laser-spot heat source\n"
                 "Paper Fig. 4(d) style — aluminium, Dirichlet 293 K on all sides",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "gaussian_source.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote gaussian_source.png  (T_max={res_g.T.max():.1f} K)")


def temporal_convergence_plot():
    """Temporal convergence study: verify Backward Euler is O(Δt).
    Fixes N=32 (spatial error small), varies dt, measures L2 error against the
    analytical Fourier-decay solution -- completes the V&V picture beyond the
    spatial study."""
    from fem.verification import l2_error
    from fem.mesh import rectangle_mesh
    from fem.assembly import assemble_KM, build_dirichlet_operator, apply_dirichlet_rhs
    from scipy.sparse.linalg import splu as _splu

    k, rho, cp = 0.1, 1.0, 1.0        # slow decay: alpha=0.1
    alpha = k / (rho * cp)
    t_end, N = 0.5, 32

    mesh = rectangle_mesh(1.0, 1.0, N, N)
    K, M, F0 = assemble_KM(mesh, k=k, rho=rho, cp=cp, Q=0.0)
    x, y = mesh.nodes[:, 0], mesh.nodes[:, 1]
    dirichlet = {}
    for side in ("left", "right", "top", "bottom"):
        for i in mesh.boundary_nodes(side):
            dirichlet[int(i)] = 0.0

    decay = np.exp(-2 * alpha * np.pi ** 2 * t_end)
    exact = lambda xx, yy: decay * np.sin(np.pi * xx) * np.sin(np.pi * yy)

    dts = [0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002]
    errors = []
    for dt in dts:
        T = np.sin(np.pi * x) * np.sin(np.pi * y)
        n_full = int(t_end / dt)
        dt_rem = t_end - n_full * dt
        if n_full > 0:
            A = (M / dt) + K
            A_el, corr, didx, dvals = build_dirichlet_operator(A, dirichlet)
            fac = _splu(A_el.tocsc())
            for _ in range(n_full):
                b = apply_dirichlet_rhs((M @ (T / dt)) + F0, corr, didx, dvals)
                T = fac.solve(b)
        if dt_rem > 1e-12 * dt:
            A = (M / dt_rem) + K
            A_el, corr, didx, dvals = build_dirichlet_operator(A, dirichlet)
            b = apply_dirichlet_rhs((M @ (T / dt_rem)) + F0, corr, didx, dvals)
            T = _splu(A_el.tocsc()).solve(b)
        errors.append(l2_error(mesh, T, exact))

    slope, _ = np.polyfit(np.log(dts[:5]), np.log(errors[:5]), 1)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(dts, errors, "o-", color="C2", label=f"Backward Euler (rate≈{slope:.2f})")
    dt_ref = np.array([dts[0], dts[-1]])
    ax.loglog(dt_ref, 0.3 * dt_ref, "k--", label=r"$\mathcal{O}(\Delta t)$ reference")
    ax.set_xlabel(r"$\Delta t$ (s)")
    ax.set_ylabel(r"$\|e\|_{L^2}$")
    ax.set_title(f"Temporal convergence — Backward Euler (N={N}, t_end={t_end} s)\n"
                 rf"Fourier decay: k={k}, $\alpha$={alpha}")
    ax.legend(); ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "temporal_convergence.png"), dpi=150)
    plt.close(fig)
    print(f"Wrote temporal_convergence.png  (fitted rate = {slope:.2f})")


def transient_timeseries_plot():
    """Center-point temperature vs time for 5 engineering materials under identical BCs.
    Demonstrates how thermal diffusivity α = k/(ρcₚ) controls transient response speed --
    the physical motivation for correct KG material retrieval."""
    materials = [
        ("Copper",          385.0, 8960.0, 385.0),
        ("Aluminium",       205.0, 2700.0, 900.0),
        ("Steel",            50.0, 7850.0, 490.0),
        ("Stainless steel",  16.3, 8000.0, 500.0),
        ("Titanium",          6.7, 4430.0, 526.0),
    ]
    from fem.mesh import rectangle_mesh
    mesh = rectangle_mesh(1.0, 1.0, 20, 20)
    center_idx = int(np.argmin(
        (mesh.nodes[:, 0] - 0.5) ** 2 + (mesh.nodes[:, 1] - 0.5) ** 2))

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, k, rho, cp in materials:
        alpha = k / (rho * cp)
        cfg = HeatConfig(
            Lx=1.0, Ly=1.0, Nx=20, Ny=20, k=k, rho=rho, cp=cp,
            bcs=[BoundaryCondition("left",   "dirichlet", 400.0),
                 BoundaryCondition("right",  "dirichlet", 293.15),
                 BoundaryCondition("top",    "insulated"),
                 BoundaryCondition("bottom", "insulated")],
            T_init=293.15, dt=1.0, t_end=300.0,
        )
        res = solve_heat(cfg)
        t_vals = [s[0] for s in res.T_history]
        T_c    = [s[1][center_idx] for s in res.T_history]
        ax.plot(t_vals, T_c, label=f"{name} (α={alpha:.2e} m²/s)")

    T_ss = 293.15 + 0.5 * (400.0 - 293.15)
    ax.axhline(T_ss, color="black", linestyle=":", alpha=0.5,
                label=f"Steady-state ≈ {T_ss:.1f} K")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("T at center node (K)")
    ax.set_title("Transient center-point temperature — 5 materials, same BCs\n"
                 "Left=400 K, Right=293 K, insulated top/bottom")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "transient_timeseries.png"), dpi=150)
    plt.close(fig)
    print("Wrote transient_timeseries.png")


def material_sensitivity_plot():
    """T_max, T_mean, and thermal diffusivity for all 5 materials under identical BCs.
    Lightweight version of paper §6.4 error-propagation analysis: shows why correct
    material retrieval (KG) matters -- wrong k causes wrong peak temperatures."""
    from knowledge_graph.kg_store import KnowledgeGraph, seed_default_materials
    kg = KnowledgeGraph(); seed_default_materials(kg)

    names, T_maxs, T_means, alphas = [], [], [], []
    for mat_name in kg.list_materials():
        mat = kg.get_material(mat_name)
        cfg = HeatConfig(
            Lx=1.0, Ly=1.0, Nx=20, Ny=20,
            k=mat.k, rho=mat.rho, cp=mat.cp, Q=5000.0,
            bcs=[BoundaryCondition("left",   "dirichlet", 373.15),
                 BoundaryCondition("right",  "dirichlet", 293.15),
                 BoundaryCondition("top",    "neumann",   1000.0),
                 BoundaryCondition("bottom", "insulated")],
        )
        res = solve_heat(cfg)
        names.append(mat_name.replace("_", "\n"))
        T_maxs.append(float(res.T.max()))
        T_means.append(float(res.T.mean()))
        alphas.append(mat.k / (mat.rho * mat.cp) * 1e6)   # ×10⁻⁶ m²/s

    x = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, vals, col, ylabel, title in [
        (axes[0], T_maxs,  "C3", "T_max (K)",       "Peak temperature"),
        (axes[1], T_means, "C0", "T_mean (K)",      "Mean temperature"),
        (axes[2], alphas,  "C2", "α (×10⁻⁶ m²/s)", "Thermal diffusivity"),
    ]:
        ax.bar(x, vals, color=col, alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Material sensitivity: identical geometry/BCs, different materials\n"
                 "(motivates KG retrieval — wrong k → wrong T_max, per paper §6.4)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "material_sensitivity.png"), dpi=150)
    plt.close(fig)
    print("Wrote material_sensitivity.png")


def agent_demo():
    ctx = AppContext.build(db_path=os.path.join(OUT, "demo_runs.db"))
    llm = get_llm()
    provider = type(llm).__name__ if llm else "offline stub"
    mode_line = f"Mode: LLM-backed ({provider}) with KG Smart warm-start" if llm else "Mode: offline stub (no API key)"

    # Structured sections so the transcript reads as a coherent demo
    sections = [
        ("=== SECTION 1: Steady-State Simulations (Material Retrieval + Warm-Start) ===", [
            "Simulate steady-state heat conduction in a copper plate. Left boundary is 373 K, right boundary is 273 K.",
            "Now do the same problem but with aluminium instead of copper.",
            "Repeat for stainless steel with the same boundary conditions.",
            "Run a steady-state problem in titanium: 500 K on the left, 293 K on the right.",
        ]),
        ("=== SECTION 2: Transient Simulations (Time-Dependent Heat Equation) ===", [
            "Simulate transient heat conduction in a copper plate. Left=400 K, right=293 K, run for 30 seconds.",
            "Transient problem in aluminium: left boundary 450 K, right boundary 300 K, simulate for 60 seconds.",
            "Transient stainless steel plate: 373 K left, 273 K right, 120 seconds total time.",
        ]),
        ("=== SECTION 3: Mixed Boundary Conditions ===", [
            "Steady-state steel plate: left wall fixed at 300 K (Dirichlet), right wall has convective cooling with h=25 W/m²K and ambient temperature 293 K (Robin BC).",
            "Copper plate with a heat flux of 5000 W/m² applied at the top boundary (Neumann), left wall at 350 K, right wall at 300 K. Steady-state.",
        ]),
        ("=== SECTION 4: Knowledge Graph & Material Queries ===", [
            "What material in the knowledge graph has the highest thermal conductivity? What about the highest thermal diffusivity?",
            "Query the knowledge graph for information about titanium and compare its thermal properties to copper.",
            "Find past runs similar to a steady-state copper problem at 373 K and 273 K boundary conditions.",
        ]),
        ("=== SECTION 5: Analytics & Run Comparison ===", [
            "Compare the most recent two simulation runs and explain the differences in their results.",
            "Which material produced the highest peak temperature across all runs so far?",
            "Analyse the solve times across all recent runs. Is there a pattern related to material or simulation type?",
        ]),
        ("=== SECTION 6: Database & Run History ===", [
            "List all completed simulation runs with their temperature ranges and solve times.",
            "What is the overall success rate of the system? How many runs have been completed?",
            "Get the status and full details of the most recent simulation run.",
        ]),
        ("=== SECTION 7: Advanced / Stress Tests ===", [
            "Simulate a high-temperature copper plate: 1000 K on the left, 300 K on the right. Steady-state. Check for any configuration warnings first.",
            "Run a transient problem in steel with a very short timestep: left=400 K, right=293 K, dt=0.1s, total time=10 seconds.",
            "What would happen if I tried to simulate with k=0? Can the system detect this as invalid?",
        ]),
    ]

    lines = [mode_line, ""]
    total = sum(len(qs) for _, qs in sections)
    done = 0
    for section_title, requests in sections:
        lines.append(section_title)
        lines.append("")
        for req in requests:
            done += 1
            print(f"\n[agent_demo {done}/{total}] > {req[:80]}...")
            response = run_request(ctx, llm, req)
            print(f"[agent_demo] done ({len(response)} chars)")
            lines.append(f"> {req}")
            lines.append(response)
            lines.append("")

    with open(os.path.join(OUT, "agent_demo_transcript.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote agent_demo_transcript.txt ({done} queries, {sum(len(l) for l in lines)} chars)")


if __name__ == "__main__":
    convergence_plot()
    temporal_convergence_plot()
    solver_comparison_plot()
    stability_comparison_plot()
    simulation_gallery()
    gaussian_source_demo()
    transient_timeseries_plot()
    material_sensitivity_plot()
    kg_structure_plot()
    agent_demo()
    print("\nAll report artifacts written to", OUT)
