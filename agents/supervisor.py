"""
Supervisor StateGraph: routes a natural-language request to one of
three specialist agents, mirroring the base paper's Fig. 1
architecture ("LangGraph Supervisor -- router + synthesiser").

Graph shape:  START -> supervisor -> {simulation|analytics|database} -> supervisor -> END

The supervisor is asked (once via LLM classification if configured,
else via keyword heuristic) which specialist should handle the
request; after that specialist reports back, the supervisor routes
to END rather than looping, which matches the paper's single-turn
CLI-style example dialogues in Appendix D.
"""
from __future__ import annotations
from typing import Literal, TypedDict
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, MessagesState

from agents.context import AppContext
from agents.specialists import build_simulation_agent, build_analytics_agent, build_database_agent


class SupervisorState(MessagesState):
    next: str


ROUTES = ("simulation", "analytics", "database")


def _keyword_route(text: str) -> str:
    t = text.lower()
    if any(w in t for w in ("compare", "vs ", "versus", "difference between")):
        return "analytics"
    if any(w in t for w in ("history", "recent run", "run status", "success rate",
                             "how many runs", "reliab")):
        return "database"
    return "simulation"


def _extract_text(content) -> str:
    """Safely extract a plain string from an LLM response content field.
    Handles both plain strings (Anthropic/OpenAI) and lists of content
    parts (Gemini returns [{type: text, text: ...}, ...])."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict):
                parts.append(part.get("text", ""))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)


def _llm_route(llm, text: str) -> str:
    prompt = (
        "Classify this request into exactly one word: SIMULATION, ANALYTICS, "
        "or DATABASE.\n"
        "- SIMULATION: run or configure a new FEM heat-equation simulation.\n"
        "- ANALYTICS: compare or analyze existing simulation results.\n"
        "- DATABASE: questions about run history, run status, or reliability.\n\n"
        f"Request: {text}\nAnswer with one word only."
    )
    out = _extract_text(llm.invoke(prompt).content).strip().upper()
    for r in ROUTES:
        if r.upper() in out:
            return r
    return "simulation"


def build_graph(ctx: AppContext, llm=None):
    sim_node = build_simulation_agent(ctx, llm)
    ana_node = build_analytics_agent(ctx, llm)
    db_node = build_database_agent(ctx, llm)

    def supervisor(state: SupervisorState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage):
            return {"next": "FINISH"}
        text = last.content
        route = _llm_route(llm, text) if llm is not None else _keyword_route(text)
        return {"next": route}

    def route_after_supervisor(state: SupervisorState) -> str:
        return state["next"] if state["next"] != "FINISH" else END

    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("simulation", sim_node)
    graph.add_node("analytics", ana_node)
    graph.add_node("database", db_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_after_supervisor,
                                 {"simulation": "simulation", "analytics": "analytics",
                                  "database": "database", END: END})
    for node in ROUTES:
        graph.add_edge(node, "supervisor")

    return graph.compile()


def run_request(ctx: AppContext, llm, text: str) -> str:
    """Blocking invocation — returns the full response as a string."""
    app = build_graph(ctx, llm)
    result = app.invoke({"messages": [HumanMessage(content=text)]})
    return _extract_text(result["messages"][-1].content)


def stream_request(ctx: AppContext, llm, text: str) -> None:
    """Show a live spinner (with tool-name updates), invoke the agent,
    then typewriter-print the response.

    The spinner writes to stderr so it stays visible even while stdout is
    temporarily redirected to /dev/null — this swallows any print() calls
    from third-party SDKs (e.g. Gemini's AFC advisory) without affecting
    the user-visible output."""
    import sys
    import io
    import contextlib
    import itertools
    import threading
    import time
    from langchain_core.callbacks import BaseCallbackHandler

    app = build_graph(ctx, llm)

    # ── Spinner on stderr ─────────────────────────────────────────────
    _FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    _phase: list[str] = ["Thinking"]
    _stop = threading.Event()

    def _spin() -> None:
        for frame in itertools.cycle(_FRAMES):
            if _stop.is_set():
                break
            sys.stderr.write(f"\r{frame} {_phase[0]}   ")
            sys.stderr.flush()
            time.sleep(0.08)
        # Clear spinner line
        sys.stderr.write("\r" + " " * 60 + "\r")
        sys.stderr.flush()

    spinner = threading.Thread(target=_spin, daemon=True)
    spinner.start()
    # ─────────────────────────────────────────────────────────────────

    class _SpinnerCallback(BaseCallbackHandler):
        def on_tool_start(self, serialized, input_str, **kwargs):  # noqa: ANN001
            name = serialized.get("name", "")
            if name:
                _phase[0] = f"→ {name}"

    # Redirect stdout → StringIO to swallow any SDK print() noise.
    # The spinner uses stderr so it remains visible during this block.
    with contextlib.redirect_stdout(io.StringIO()):
        result = app.invoke(
            {"messages": [HumanMessage(content=text)]},
            config={"callbacks": [_SpinnerCallback()]},
        )

    _stop.set()
    spinner.join(timeout=0.3)

    response = _extract_text(result["messages"][-1].content)
    try:
        from rich.console import Console
        from rich.markdown import Markdown
        # force_terminal=True ensures colours and markup work inside Docker
        Console(force_terminal=True).print(Markdown(response))
    except ImportError:
        print(response)
