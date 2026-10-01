# PDE-Agents-lite

A lightweight, from-scratch implementation of the multi-agent FEM simulation framework described in:

> **"PDE-Agents: An LLM-Orchestrated Multi-Agent Framework for Automated Finite Element Simulations with Knowledge Graph-Augmented Reasoning"**  
> Adhikari, Noorsumar & Jensen — arXiv:2606.07850

The system takes a natural-language request ("solve a steady-state heat problem in a copper plate"), routes it through a LangGraph supervisor to specialist agents, runs a hand-written P1 finite element solver, and returns results with full run lineage. No GPU, no database server, no mandatory API key — it runs entirely offline out of the box.

---

## How it works

A natural-language request enters a **LangGraph supervisor** that routes to one of three specialist agents:

```
natural-language request
         │
         ▼
  ┌─────────────┐
  │  Supervisor  │  StateGraph router + response synthesizer
  └──────┬──────┘
         │
   ┌─────┼─────┐
   ▼     ▼     ▼
Sim   Analytics  DB
Agent   Agent   Agent
  │
  ├── query_knowledge_graph   material property lookup
  ├── warm_start              top-3 similar past runs (TF-IDF cosine)
  ├── check_config_warnings   rule-based pre-flight checks
  ├── validate_config
  └── run_simulation ──► HeatSolver
                            ├── fem/mesh.py          structured P1 mesh
                            ├── fem/assembly.py      K, M, F matrices + BCs
                            └── fem/linear_solvers.py Gauss / LU / power method
                         ──► SQLite run record + knowledge graph node
```

**Simulation Agent** handles FEM setup and execution. **Analytics Agent** answers questions about results and convergence. **Database Agent** queries run history.

---

## Implementation

The paper's stack requires a GPU cluster, Neo4j, and Postgres. This implementation swaps those for zero-dependency equivalents while keeping the architecture and numerical methods intact.

| Component | Paper | This project |
|---|---|---|
| FEM solver | DOLFINx / PETSc (external) | Hand-written P1 assembly (`fem/assembly.py`) |
| Linear solve | PETSc | Custom Gauss elimination + LU decomp (`fem/linear_solvers.py`), cross-validated against scipy |
| Stability estimate | Static CFL formula check | Power method on the discrete diffusion operator — computes spectral radius at runtime |
| Knowledge graph | Neo4j + nomic-embed 768-d + HNSW | networkx + TF-IDF cosine (`knowledge_graph/kg_store.py`) |
| Run persistence | PostgreSQL + MinIO | SQLite (`database/run_store.py`) |
| LLM | Qwen3 / Llama 4 via Ollama (2× GPU) | Any LangChain chat model via API key, or a deterministic offline stub |
| Geometry | Gmsh | Structured triangular mesh generator, rectangle domains (`fem/mesh.py`) |

---

## Verification & Validation

`fem/verification.py` runs three closed-form benchmarks across four mesh refinements (N ∈ {8, 16, 32, 64}), computing L2 error via 6-point triangle quadrature and fitting convergence rates by log-log regression.

| Benchmark | Convergence rate | Expected (P1) |
|---|---|---|
| Steady linear profile | machine precision (~1e-14) | exact in P1 space |
| Transient Fourier decay | **2.13** | 2.0 |
| Steady Poisson (const. source) | **2.00** | 2.0 |

Both convergent cases hit the theoretical O(h²) rate for P1 elements. The custom Gauss and LU solvers agree with `scipy.sparse.linalg.spsolve` to 6 significant figures on the same systems.

Full numbers: `outputs/solver_validation.txt` and `outputs/convergence_study.png`.

---

## Quick start

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). No API key needed to run.

```bash
git clone <this-repo> && cd pde-agents-lite
cp .env.example .env
uv sync
uv run python main.py "Solve a steady-state heat problem in a copper plate, 373K left, 273K right"
```

Or with plain pip:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py "Solve a steady-state heat problem in a copper plate, 373K left, 273K right"
```

**Makefile shortcuts:** `make install` · `make run` · `make demo` · `make test` · `make report`

---

## Walkthrough

### Prerequisites

| Tool | Version | Install |
|---|---|---|
| Python | 3.12+ | `python3 --version` |
| uv | any | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| git | any | — |

An API key is optional. Without one the offline stub runs the same tool sequence a real ReAct agent would, exercising the full graph and FEM solver.

---

### 1. Clone and install

```bash
git clone <this-repo> && cd pde-agents-lite
cp .env.example .env
uv sync                  # ~30 s first run; reproduces the locked environment exactly
```

No uv:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

### 2. Run the tests

```bash
uv run python tests/test_core.py
# make test
```

Expected output:

```
[PASS] Gauss elimination matches scipy (rel err < 1e-6)
[PASS] LU decomposition matches scipy (rel err < 1e-6)
[PASS] Power method eigenvalue within 1% of numpy.linalg.eig
[PASS] Steady linear profile — L2 error at machine precision
[PASS] Transient Fourier decay — convergence rate 2.13 (expected ~2.0)
[PASS] Steady Poisson (const. source) — convergence rate 2.00
```

All six pass with no API key.

---

### 3. Generate report artifacts

```bash
uv run python demo_report.py
# make demo
```

Runs two offline agent conversations and writes into `outputs/`:

- `convergence_study.png` — log-log convergence plot
- `solver_validation.txt` — solver cross-validation numbers
- `simulation_gallery.png` — temperature field plots across mesh refinements
- `agent_transcript.txt` — full tool-call sequence

---

### 4. Run a simulation

**Offline (no API key):**

```bash
uv run python main.py "Solve a steady-state heat problem in a copper plate, 373K left, 273K right"
```

The offline stub follows the same tool sequence as a real ReAct agent: knowledge-graph warm-start → config warnings → validation → FEM solve → report. Output is a Rich-formatted panel with mesh info, solver stats, and temperature summary.

**With a real LLM:**

Edit `.env`:

```
ANTHROPIC_API_KEY=sk-ant-...    # claude-sonnet-4-6 (default)
# GOOGLE_API_KEY=...            # gemini-2.0-flash (default)
# OPENAI_API_KEY=sk-...         # gpt-4o (default)
```

The supervisor now reasons over requests and routes via LLM inference. Numerical output is identical.

**Interactive mode:**

```bash
uv run python main.py
# make run
```

Example session:

```
> Solve a steady-state heat problem in a copper plate, 373K left, 273K right
> Now run the same case transiently for 10 seconds
> Show me the last 5 simulation runs
> What is the L2 error for the last run?
```

---

### 5. Inspect run history

Every simulation is persisted to `runs.db`. Query it directly:

```bash
sqlite3 runs.db "SELECT id, material, solver_type, max_temp, timestamp FROM runs ORDER BY timestamp DESC LIMIT 5;"
```

Or ask the Database Agent in the interactive REPL: `"Show me the last 5 runs"`.

---

### 6. Docker (optional)

```bash
make docker-build    # builds the uv-based image (~2 min first time)
make docker-run      # interactive CLI
make docker-demo     # single copper-plate demo, then exit
```

Run history persists across sessions via a named Docker volume.

---

### Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langchain_anthropic` | env not activated | `source .venv/bin/activate` or use `uv run` prefix |
| `ANTHROPIC_API_KEY not set` warning | no key in `.env` | add a key, or ignore — offline stub runs fine |
| `runs.db` permission error in Docker | volume ownership mismatch | `docker compose down -v && make docker-build` |
| Convergence rate outside expected range | dependency version drift | `uv sync` to restore the locked environment |

---

## Project structure

```
fem/                  mesh, assembly, linear solvers, heat solver, V&V
knowledge_graph/      graph store, material seed data, rule engine
database/             SQLite run persistence
agents/               LLM factory, tools, specialist agents, LangGraph supervisor
tests/test_core.py    correctness tests
demo_report.py        generates outputs/ artifacts
generate_report.py    scope-compliance audit → FULL_REPORT.md
main.py               CLI entrypoint
outputs/              generated artifacts (not committed)
```

`runs.db` and `outputs/` are generated — they are rebuilt by the commands above, not shipped. `2606.07850v2.pdf` is the reference paper.

---

## Extending the project

The paper's `EXPLICIT_CFL_VIOLATION` rule is a static formula check. This project computes a **dynamic stability estimate** via the power method on the discrete diffusion operator (`fem/solver.py`) for every transient run. A natural extension is to have the Simulation Agent act on that estimate — auto-shrinking `dt` when the spectral radius implies instability — and benchmark it against the paper's static-rule baseline using the paper's 50-task ablation methodology (Section 6).
