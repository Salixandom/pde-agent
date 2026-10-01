"""
Pluggable LLM backend.

The base paper runs two locally-hosted open-source models
(Qwen3-Coder-Next, Llama 4 Scout) via Ollama on dual GPUs. This
sandbox has neither GPUs, Ollama, nor a stored API key, so:

  * `get_llm()` returns a real `ChatAnthropic` model IF an
    `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`, via langchain-openai)
    environment variable is present -- this is the path to use for
    an actual course submission / real run, exactly matching the
    paper's tool-calling ReAct architecture.
  * If no key is present, `get_llm()` returns None, and the
    specialist agents (agents/specialists.py) fall back to a
    deterministic offline "stub" reasoning loop, so the full
    LangGraph wiring, tools, KG, and solver can still be
    demonstrated end-to-end without network access.

Swap models by setting PDE_AGENTS_MODEL, e.g.:
    export ANTHROPIC_API_KEY=sk-...
    export PDE_AGENTS_MODEL=claude-sonnet-4-6

For Gemini:
    export GOOGLE_API_KEY=your-key
    export PDE_AGENTS_MODEL=gemini-2.0-flash   # default when GOOGLE_API_KEY is set
    pip install langchain-google-genai
"""
from __future__ import annotations
import os


def get_llm():
    model_name = os.environ.get("PDE_AGENTS_MODEL")
    temperature = float(os.environ.get("PDE_AGENTS_TEMPERATURE", "0"))

    if os.environ.get("ANTHROPIC_API_KEY"):
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model_name or "claude-sonnet-4-6", temperature=temperature)

    if os.environ.get("GOOGLE_API_KEY"):
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "GOOGLE_API_KEY set but langchain-google-genai is not installed; "
                "pip install langchain-google-genai"
            ) from e
        return ChatGoogleGenerativeAI(model=model_name or "gemini-2.0-flash",
                                       temperature=temperature)

    if os.environ.get("OPENAI_API_KEY"):
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "OPENAI_API_KEY set but langchain-openai is not installed; "
                "pip install langchain-openai"
            ) from e
        return ChatOpenAI(model=model_name or "gpt-4o", temperature=temperature)

    return None  # triggers the offline stub reasoning path
