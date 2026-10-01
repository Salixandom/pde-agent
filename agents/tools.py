"""
Tool registry for the three specialist agents, mirroring the base
paper's Section 3.2 tool lists:
  Simulation Agent : check_config_warnings, query_knowledge_graph,
                      validate_config, run_simulation, warm_start
  Analytics Agent   : compare_runs
  Database Agent    : list_recent_runs, get_run_status, success_rate

Each factory function takes the shared AppContext and returns a
list of @tool-decorated callables, so the same tool logic can be
bound to real langchain agents (agents/*_agent.py) or invoked
directly for the offline demo (demo_offline.py).
"""
from __future__ import annotations
import json
import numpy as np
from langchain_core.tools import tool

from agents.context import AppContext
from fem.solver import HeatConfig, BoundaryCondition, solve_heat
from knowledge_graph.rules import check_config

_VALID_BC_TYPES = {"dirichlet", "neumann", "robin", "insulated"}
_VALID_SOLVERS = {"scipy", "gauss", "lu"}


def _safe_parse(value):
    """Recursively unwrap double-encoded JSON strings.
    Some LLMs (e.g. Gemini) serialise nested objects as JSON strings
    inside the outer JSON, so we may receive '{"type":"dirichlet"}'
    (a string) where we expect {"type": "dirichlet"} (a dict)."""
    if isinstance(value, str):
        try:
            return _safe_parse(json.loads(value))
        except (json.JSONDecodeError, ValueError):
            return value
    if isinstance(value, list):
        return [_safe_parse(v) for v in value]
    if isinstance(value, dict):
        return {k: _safe_parse(v) for k, v in value.items()}
    return value


def _cfg_from_json(config_json) -> HeatConfig:
    """Parse a HeatConfig from a JSON string or dict.
    _safe_parse unwraps any double-encoded strings (Gemini quirk);
    Pydantic then handles aliasing, type coercion, and unknown-field stripping."""
    if isinstance(config_json, str):
        d = json.loads(config_json)
    else:
        d = dict(config_json)
    return HeatConfig.model_validate(_safe_parse(d))


def _collect_errors(cfg: HeatConfig) -> list[str]:
    """Return validation error strings; empty list means config is valid."""
    errors = []
    if not isinstance(cfg.Nx, int) or cfg.Nx <= 0:
        errors.append(f"Nx must be a positive integer, got {cfg.Nx!r}")
    if not isinstance(cfg.Ny, int) or cfg.Ny <= 0:
        errors.append(f"Ny must be a positive integer, got {cfg.Ny!r}")
    if cfg.Lx <= 0:
        errors.append(f"Lx must be > 0, got {cfg.Lx}")
    if cfg.Ly <= 0:
        errors.append(f"Ly must be > 0, got {cfg.Ly}")
    if cfg.k <= 0 or cfg.rho <= 0 or cfg.cp <= 0:
        errors.append("k, rho, and cp must each be > 0")
    if cfg.t_end > 0 and cfg.dt <= 0:
        errors.append(f"dt must be > 0 for transient problems, got {cfg.dt}")
    for bc in cfg.bcs:
        if bc.type not in _VALID_BC_TYPES:
            errors.append(f"unknown BC type {bc.type!r}; must be one of {sorted(_VALID_BC_TYPES)}")
    if cfg.linear_solver not in _VALID_SOLVERS:
        errors.append(f"unknown linear_solver {cfg.linear_solver!r}; must be one of {sorted(_VALID_SOLVERS)}")
    if not (0.0 <= cfg.theta <= 1.0):
        errors.append(f"theta must be in [0, 1], got {cfg.theta} "
                      f"(0=Forward Euler, 0.5=Crank-Nicolson, 1=Backward Euler)")
    return errors


def make_simulation_tools(ctx: AppContext):
    @tool
    def check_config_warnings(config_json: str) -> str:
        """Run the pre-simulation rule engine on a HeatConfig JSON
        string and return any warnings as JSON."""
        try:
            cfg = _cfg_from_json(config_json)
            warns = check_config(cfg)
            return json.dumps(warns)
        except Exception as e:  # noqa: BLE001
            return json.dumps([("PARSE_ERROR", str(e))])

    @tool
    def query_knowledge_graph(material_or_query: str) -> str:
        """Look up a material's properties (k, rho, cp) in the
        knowledge graph, or run a general text query."""
        return json.dumps(ctx.kg.query(material_or_query))

    @tool
    def warm_start(task_description: str) -> str:
        """Retrieve the top-3 most similar past runs for few-shot
        warm-start context (KG Smart pattern)."""
        return json.dumps(ctx.kg.warm_start(task_description, top_k=3))

    @tool
    def validate_config(config_json: str) -> str:
        """Sanity-check a HeatConfig JSON string: parses cleanly,
        mesh dimensions are positive integers, material properties
        are positive, dt > 0 for transient, BC types and solver
        name are recognized."""
        try:
            cfg = _cfg_from_json(config_json)
            errors = _collect_errors(cfg)
            if errors:
                return json.dumps({"valid": False, "errors": errors})
            return json.dumps({"valid": True})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"valid": False, "error": str(e)})

    @tool
    def run_simulation(config_json: str, description: str, material: str = "") -> str:
        """Run the FEM heat-equation solver for a HeatConfig JSON
        string and persist the result (run lineage + KG warm-start
        index + SQLite record).  Config is validated before the solve;
        non-finite results are rejected as failures."""
        # Validate before touching the database
        try:
            cfg = _cfg_from_json(config_json)
        except Exception as e:  # noqa: BLE001
            return json.dumps({"status": "failed", "error": f"config parse error: {e}"})
        errors = _collect_errors(cfg)
        if errors:
            return json.dumps({"status": "failed",
                                "error": "invalid config: " + "; ".join(errors)})

        run_id = ctx.db.create_run(description, json.loads(config_json))
        try:
            res = solve_heat(cfg)
            # Reject non-finite results (NaN / Inf) as numerical failures
            if not np.all(np.isfinite(res.T)):
                raise ValueError("solver produced non-finite temperatures (NaN or Inf)")
            ctx.db.complete_run(
                run_id, dofs=res.dofs, solve_time_s=res.solve_time_s,
                t_min=float(res.T.min()), t_max=float(res.T.max()),
                t_mean=float(res.T.mean()), warnings=res.warnings,
            )
            t_min_val = float(res.T.min())
            t_max_val = float(res.T.max())
            # T_score: physics plausibility check (paper §6.4).
            # Above absolute zero and below a numerically insane bound.
            t_score = 1.0 if (t_min_val >= 0.0 and t_max_val <= 10_000.0) else 0.5
            # KG indexing is best-effort: an indexing failure must not
            # retroactively mark a successful simulation as failed.
            try:
                ctx.kg.record_run(run_id, description, json.loads(config_json),
                                   material=material or None)
            except Exception:  # noqa: BLE001
                pass
            return json.dumps({
                "run_id": run_id, "status": "completed", "dofs": res.dofs,
                "solve_time_s": round(res.solve_time_s, 4),
                "T_min": t_min_val, "T_max": t_max_val,
                "T_mean": float(res.T.mean()),
                "max_stable_dt": res.max_stable_dt,
                "t_score": t_score,
                "warnings": res.warnings,
            })
        except Exception as e:  # noqa: BLE001
            ctx.db.fail_run(run_id, str(e))
            # Record failure pattern as a KnownIssue node for future diagnostics
            try:
                e_str = str(e).lower()
                if "non-finite" in e_str or "nan" in e_str or "inf" in e_str:
                    issue_code = "NUMERICAL_FAILURE"
                elif "singular" in e_str:
                    issue_code = "SINGULAR_SYSTEM"
                else:
                    issue_code = "SIMULATION_ERROR"
                ctx.kg.add_known_issue(issue_code, str(e)[:200])
                ctx.kg.g.add_edge(f"run:{run_id}", f"issue:{issue_code}",
                                   kind="TRIGGERED")
            except Exception:  # noqa: BLE001
                pass
            return json.dumps({"run_id": run_id, "status": "failed", "error": str(e)})

    @tool
    def run_parametric_sweep(base_config_json: str, parameter: str,
                              values_json: str, description: str = "") -> str:
        """Sweep one HeatConfig field across a list of values and return a
        results table.  parameter: HeatConfig field name (e.g. 'k', 'Nx',
        'dt', 'theta').  values_json: JSON array, e.g. '[1.0, 10.0, 100.0]'.
        Each entry runs a full validated simulation; results include T_min,
        T_max, T_mean, t_score, and any warnings for each value."""
        try:
            base = json.loads(base_config_json)
            values = json.loads(values_json)
            if not isinstance(values, list) or len(values) == 0:
                return json.dumps({"status": "failed",
                                    "error": "values_json must be a non-empty JSON array"})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"status": "failed", "error": f"parse error: {e}"})

        rows = []
        for v in values:
            sweep_cfg = {**base, parameter: v}
            desc = description or f"sweep {parameter}={v}"
            row = json.loads(run_simulation.func(json.dumps(sweep_cfg), desc))
            row["sweep_value"] = v
            rows.append(row)

        return json.dumps({
            "parameter": parameter,
            "values": values,
            "results": rows,
            "summary": {
                "n_completed": sum(1 for r in rows if r.get("status") == "completed"),
                "n_failed":    sum(1 for r in rows if r.get("status") != "completed"),
            },
        })

    @tool
    def modify_config(config_json: str, changes_json: str) -> str:
        """Apply field-level overrides to a HeatConfig JSON string and
        return the updated config.  changes_json is a JSON dict of
        field overrides, e.g. '{"dt": 0.01, "Nx": 20}'.  Use after
        check_config_warnings to correct flagged parameters before
        re-running, mirroring the paper's self-correction loop."""
        try:
            d = json.loads(config_json)
            overrides = json.loads(changes_json)
            d.update(overrides)
            updated_json = json.dumps(d)
            cfg = _cfg_from_json(updated_json)
            errors = _collect_errors(cfg)
            if errors:
                return json.dumps({"success": False, "errors": errors,
                                    "config_json": updated_json})
            return json.dumps({"success": True, "config_json": updated_json})
        except Exception as e:  # noqa: BLE001
            return json.dumps({"success": False, "error": str(e)})

    return [check_config_warnings, query_knowledge_graph, warm_start,
            validate_config, modify_config, run_parametric_sweep, run_simulation]


def make_analytics_tools(ctx: AppContext):
    @tool
    def compare_runs(run_id_a: str, run_id_b: str) -> str:
        """Compare two completed runs' summary statistics."""
        a, b = ctx.db.get_run(run_id_a), ctx.db.get_run(run_id_b)
        if a is None or b is None:
            return json.dumps({"error": "one or both run_ids not found"})
        return json.dumps({
            "run_a": {"T_max": a["t_max"], "T_mean": a["t_mean"], "dofs": a["dofs"]},
            "run_b": {"T_max": b["t_max"], "T_mean": b["t_mean"], "dofs": b["dofs"]},
            "delta_T_max": (a["t_max"] or 0) - (b["t_max"] or 0),
        })

    return [compare_runs]


def make_database_tools(ctx: AppContext):
    @tool
    def list_recent_runs(limit: int = 5) -> str:
        """List the most recent simulation runs."""
        return json.dumps(ctx.db.list_recent(limit))

    @tool
    def get_run_status(run_id: str) -> str:
        """Get the full stored record for one run_id."""
        rec = ctx.db.get_run(run_id)
        return json.dumps(rec if rec else {"error": "not found"})

    @tool
    def success_rate() -> str:
        """Overall success rate across all recorded runs."""
        return json.dumps({"success_rate": ctx.db.success_rate()})

    return [list_recent_runs, get_run_status, success_rate]
