"""
Builds the three specialist agent nodes (Simulation, Analytics,
Database), mirroring the base paper's agent stack (Table 1 / Section
3.2). Each `build_*_agent` returns a callable node function usable
directly inside a LangGraph StateGraph.

When an LLM is configured (agents/llm_factory.get_llm() != None),
each specialist is a genuine ReAct agent (langgraph.prebuilt.
create_react_agent) with the tools from agents/tools.py bound --
this is the real system, architecturally identical to the paper's
per-agent ReAct loops.

When no LLM is configured, a deterministic offline "stub" walks the
same tool sequence the paper's system-prompted agents are
instructed to follow (check_config_warnings -> validate_config ->
run_simulation for the Simulation Agent, etc.), so the LangGraph
wiring and the full FEM/KG/DB pipeline underneath it can be
exercised end-to-end without network access.
"""
from __future__ import annotations
import json
import re
from langchain_core.messages import AIMessage, HumanMessage

from agents.context import AppContext
from agents.tools import make_simulation_tools, make_analytics_tools, make_database_tools
from fem.solver import HeatConfig, BoundaryCondition

SIM_SYSTEM_PROMPT = (
    "You are the Simulation Agent for a finite-element heat-equation "
    "assistant. Given a natural-language task, extract material and "
    "boundary-condition parameters, call check_config_warnings and "
    "validate_config before run_simulation, and report the resulting "
    "temperature range, run_id, and any warnings."
)
ANALYTICS_SYSTEM_PROMPT = (
    "You are the Analytics Agent. Compare simulation runs and "
    "summarize differences in plain language."
)
DB_SYSTEM_PROMPT = (
    "You are the Database Agent. Answer questions about run history, "
    "run status, and overall system reliability using your tools."
)


def _last_human_text(state) -> str:
    for m in reversed(state["messages"]):
        if isinstance(m, HumanMessage):
            return m.content
    return state["messages"][-1].content


# --------------------------------------------------------------- #
# Offline heuristic parsing (used only when no LLM is configured)
# --------------------------------------------------------------- #
def _guess_config(ctx: AppContext, text: str) -> tuple[HeatConfig, str]:
    text_l = text.lower().replace("_", " ")
    # Sort longest-first so 'stainless steel' matches before 'steel'
    material = next(
        (m for m in sorted(ctx.kg.list_materials(), key=len, reverse=True)
         if m.replace("_", " ") in text_l),
        "steel",
    )
    mat = ctx.kg.get_material(material)

    temps = [float(t) for t in re.findall(r"(\d+(?:\.\d+)?)\s*k\b", text_l)]
    t_hot, t_cold = (temps[0], temps[1]) if len(temps) >= 2 else (373.15, 273.15)

    transient = "transient" in text_l or "time" in text_l
    cfg = HeatConfig(
        Lx=1.0, Ly=1.0, Nx=20, Ny=20,
        k=mat.k, rho=mat.rho, cp=mat.cp, Q=0.0,
        bcs=[
            BoundaryCondition(side="left", type="dirichlet", value=t_hot),
            BoundaryCondition(side="right", type="dirichlet", value=t_cold),
        ],
        T_init=t_cold, dt=1.0, t_end=(50.0 if transient else 0.0),
    )
    return cfg, material


def _make_warm_start_prompt(ctx: AppContext, task_text: str) -> str:
    """Inject top-3 similar past runs into the system prompt BEFORE the
    agent loop -- the paper's KG Smart warm-start pattern (§6.3 Algorithm 1).
    Front-loading context eliminates budget-exhaustion from mandatory pre-run
    KG tool calls, which the paper identifies as the dominant reliability gain."""
    prior_runs = ctx.kg.warm_start(task_text, top_k=3)
    if not prior_runs:
        return SIM_SYSTEM_PROMPT
    examples = "\n".join(
        f"  Run {r['run_id']} (sim={r['similarity']:.2f}): "
        f"{r['description']} → config={json.dumps(r['config'])}"
        for r in prior_runs
    )
    return (
        SIM_SYSTEM_PROMPT
        + "\n\nKG Smart warm-start — top similar past runs (use as few-shot reference):\n"
        + examples
        + "\n\nUse these configurations as starting points; query_knowledge_graph "
          "and warm_start tools are available for failure recovery if needed."
    )


def build_simulation_agent(ctx: AppContext, llm):
    tools = make_simulation_tools(ctx)
    if llm is not None:
        from langgraph.prebuilt import create_react_agent

        def node(state):
            # Warm-start injection: embed task, retrieve similar runs, inject
            # into system prompt before the ReAct loop starts (KG Smart §6.3).
            task_text = _last_human_text(state)
            system_prompt = _make_warm_start_prompt(ctx, task_text)
            agent = create_react_agent(llm, tools, prompt=system_prompt)
            result = agent.invoke({"messages": state["messages"]})
            return {"messages": result["messages"][len(state["messages"]):]}
        return node

    by_name = {t.name: t for t in tools}

    def offline_node(state):
        text = _last_human_text(state)
        cfg, material = _guess_config(ctx, text)
        config_json = json.dumps({
            "Lx": cfg.Lx, "Ly": cfg.Ly, "Nx": cfg.Nx, "Ny": cfg.Ny,
            "k": cfg.k, "rho": cfg.rho, "cp": cfg.cp, "Q": cfg.Q,
            "bcs": [bc.__dict__ for bc in cfg.bcs],
            "T_init": cfg.T_init,
            "dt": cfg.dt, "t_end": cfg.t_end, "linear_solver": cfg.linear_solver,
        })
        prior = json.loads(by_name["warm_start"].func(text))
        warns = json.loads(by_name["check_config_warnings"].func(config_json))
        valid = json.loads(by_name["validate_config"].func(config_json))
        sim = json.loads(by_name["run_simulation"].func(config_json, text, material))

        prior_line = (f"Warm-start: found {len(prior)} similar prior run(s) "
                       f"(top similarity {prior[0]['similarity']:.2f})"
                       if prior else "Warm-start: no similar prior runs yet")
        reply = (
            f"[Simulation Agent | offline stub -- no LLM configured]\n"
            f"{prior_line}\n"
            f"Material guessed: {material} (k={cfg.k}, rho={cfg.rho}, cp={cfg.cp})\n"
            f"Pre-run warnings: {warns if warns else 'none'}\n"
            f"Config valid: {valid['valid']}\n"
            f"Run {sim.get('run_id')}: status={sim.get('status')}, "
            f"dofs={sim.get('dofs')}, T range=[{sim.get('T_min'):.2f}, "
            f"{sim.get('T_max'):.2f}] K (mean {sim.get('T_mean'):.2f} K), "
            f"solve_time={sim.get('solve_time_s')}s"
            if sim.get("status") == "completed" else
            f"[Simulation Agent | offline stub] run failed: {sim.get('error')}"
        )
        return {"messages": [AIMessage(content=reply)]}

    return offline_node


def build_analytics_agent(ctx: AppContext, llm):
    tools = make_analytics_tools(ctx)
    if llm is not None:
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(llm, tools, prompt=ANALYTICS_SYSTEM_PROMPT)

        def node(state):
            result = agent.invoke({"messages": state["messages"]})
            return {"messages": result["messages"][len(state["messages"]):]}
        return node

    by_name = {t.name: t for t in tools}

    def offline_node(state):
        text = _last_human_text(state)
        ids = re.findall(r"\b[0-9a-f]{8}\b", text)
        if len(ids) >= 2:
            out = json.loads(by_name["compare_runs"].func(ids[0], ids[1]))
            reply = f"[Analytics Agent | offline stub] comparison: {out}"
        else:
            recent = ctx.db.list_recent(5)
            reply = (f"[Analytics Agent | offline stub] no two run_ids found in the "
                      f"request; most recent runs: "
                      f"{[r['run_id'] for r in recent]}")
        return {"messages": [AIMessage(content=reply)]}

    return offline_node


def build_database_agent(ctx: AppContext, llm):
    tools = make_database_tools(ctx)
    if llm is not None:
        from langgraph.prebuilt import create_react_agent
        agent = create_react_agent(llm, tools, prompt=DB_SYSTEM_PROMPT)

        def node(state):
            result = agent.invoke({"messages": state["messages"]})
            return {"messages": result["messages"][len(state["messages"]):]}
        return node

    by_name = {t.name: t for t in tools}

    def offline_node(state):
        recent = json.loads(by_name["list_recent_runs"].func(5))
        rate = json.loads(by_name["success_rate"].func())
        reply = (f"[Database Agent | offline stub] recent runs: "
                 f"{[r['run_id'] for r in recent]}; "
                 f"overall success_rate={rate['success_rate']:.2%}")
        return {"messages": [AIMessage(content=reply)]}

    return offline_node
