"""
CLI entrypoint.

Interactive mode:
    python3 main.py
    > Solve a steady-state heat problem in a copper plate, 373K left, 273K right
    > What's the recent run history?

One-shot mode:
    python3 main.py "Solve a transient heat problem in aluminium"

Set ANTHROPIC_API_KEY (or OPENAI_API_KEY) in the environment to use
a real LLM for agent reasoning; otherwise the deterministic offline
stub handles requests (see agents/llm_factory.py).
"""
import sys
import os
import warnings

# Suppress all warnings (Gemini AFC advisories, deprecation notices, etc.).
# These are cosmetic and do not affect correctness; PYTHONWARNINGS=ignore
# is also set in docker-compose.yml for warnings that bypass Python's filter.
warnings.filterwarnings("ignore")

# Load .env before anything else so all env vars are available to sub-modules
try:
    from dotenv import load_dotenv
    load_dotenv(override=False)  # don't override vars already set in the shell
except ImportError:
    pass  # python-dotenv optional; fall back to environment variables only

from agents.context import AppContext
from agents.llm_factory import get_llm
from agents.supervisor import run_request, stream_request


def _respond(ctx, llm, text: str) -> None:
    """Stream when an LLM is configured; fall back to blocking for the offline stub."""
    if llm is not None:
        stream_request(ctx, llm, text)
    else:
        print(run_request(ctx, llm, text))


def main():
    db_path = os.environ.get("PDE_AGENTS_DB", "runs.db")
    ctx = AppContext.build(db_path=db_path)
    llm = get_llm()
    mode = "LLM-backed (streaming)" if llm is not None else "offline stub (no API key set)"
    print(f"PDE-Agents-lite -- {mode}\n")

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
        print(f"> {text}")
        _respond(ctx, llm, text)
        return

    print("Type a request (or 'quit'):")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not text or text.lower() in ("quit", "exit"):
            break
        _respond(ctx, llm, text)


if __name__ == "__main__":
    main()
