# PDE-Agents-lite

A from-scratch implementation of the base paper **"PDE-Agents: An
LLM-Orchestrated Multi-Agent Framework for Automated Finite Element
Simulations with Knowledge Graph-Augmented Reasoning"** (Adhikari,
Noorsumar & Jensen, MatPro/IFE, arXiv:2606.07850), built for a
CSE401 (Numerical Analysis, Simulation and Modeling) course project.

It reproduces the paper's **architecture** (LangGraph supervisor +
three specialist agents, knowledge-graph warm-start, rule-based
pre-run checks, run lineage) and its **numerical rigor** (a formal
V&V convergence study), while replacing every piece of the paper's
stack that needs a GPU cluster, a database server, or a downloadable
model with a from-scratch equivalent that runs anywhere.

## What's genuinely "yours" vs. what the paper contributed

| Paper's PDE-Agents | This implementation | Why |
|---|---|---|
| DOLFINx/PETSc FEM solver (external, black-box) | **Hand-written P1 FEM assembly** (`fem/assembly.py`) | Makes the linear system the agent solves fully inspectable/extendable |
| PETSc's linear solve | **Custom Gauss elimination + LU decomposition** (`fem/linear_solvers.py`), cross-validated against `numpy`/`scipy` | Direct tie-in to CSE401's systems-of-equations unit — this is the actual numerical-methods contribution of the project |
| — (not in paper) | **Power method** for the discrete diffusion operator's dominant eigenvalue (`fem/linear_solvers.py::power_method`), used for an automatic stability estimate | Ties in CSE401's eigenvalue unit; extends the paper's static CFL rule into a computed one |
| Neo4j + nomic-embed (768-d) + HNSW | `networkx` graph + TF-IDF/cosine similarity (`knowledge_graph/kg_store.py`) | No graph-DB server or downloadable embedding model available in a plain sandbox; same warm-start *pattern* (embed task → retrieve top-3 similar runs → inject as context) |
| PostgreSQL + MinIO | SQLite (`database/run_store.py`) | Same run-lineage guarantee, zero-ops |
| Qwen3-Coder-Next / Llama 4 Scout via Ollama on 2x GPUs | `ChatAnthropic` via LangChain (or any LangChain chat model) when an API key is set, else a **deterministic offline stub** | No local GPU/Ollama in a plain sandbox; the graph topology, tools, and downstream pipeline are identical either way |
| Gmsh geometry | Structured triangular mesh generator (`fem/mesh.py`), rectangle domains only | Sufficient for the paper's Fig. 4(a)/(b)/(e) cases; extending to Gmsh-style boolean/hole geometries is listed under Future Work |

Everything in the *left* column that this project could not
faithfully reproduce (GPU-hosted open models, Neo4j, Postgres) is a
**resource substitution**, not a shortcut on the numerical methods —
the FEM assembly, linear solves, and convergence analysis are all
implemented directly, nothing is delegated to an opaque library.

## Architecture

```
        natural-language request
                 |
                 v
        +-----------------+
        |   Supervisor     |   LangGraph StateGraph, routes to one
        | (router/synth.)  |   of three specialists (paper Fig. 1)
        +--------+---------+
                 |
   +-------------+-------------+
   v             v             v
Simulation   Analytics     Database
 Agent         Agent         Agent
   |
   +-- check_config_warnings  (rule engine, knowledge_graph/rules.py)
   +-- query_knowledge_graph  (material lookup)
   +-- warm_start             (top-3 similar past runs, TF-IDF cosine)
   +-- validate_config
   +-- run_simulation  ------> fem/solver.py (HeatSolver)
                                 |-- fem/mesh.py       (structured P1 mesh)
                                 |-- fem/assembly.py   (K, M, F assembly + BCs)
                                 |-- fem/linear_solvers.py (Gauss / LU / power method)
                               --> SQLite run record + KG run node
```

## Verification & Validation

`fem/verification.py` reproduces the paper's Section 5 methodology:
three closed-form benchmarks (steady linear profile, transient
Fourier-mode decay, steady Poisson with constant source), mesh
refinement N ∈ {8,16,32,64}, L2 error via 6-point triangle
quadrature, convergence rate via log-log regression.

Results (`outputs/solver_validation.txt`, `outputs/convergence_study.png`):

| Case | Fitted rate | Paper's rate |
|---|---|---|
| Steady Linear Profile | machine precision (~1e-14) | machine precision (paper: exact in P1 space) |
| Transient Fourier Decay | **2.13** | 2.04 |
| Steady Poisson (const. source) | **2.00** | 2.00 |

Both convergent cases hit the theoretical O(h²) rate for P1
elements, matching the paper's Table 2 to within normal
quadrature/discretization variance.

The custom Gauss elimination and LU decomposition solvers are
cross-validated against `scipy.sparse.linalg.spsolve` on the same
system and agree to 6 significant figures (see
`outputs/solver_validation.txt`).

## Running it

The project is managed with [uv](https://docs.astral.sh/uv/) — the lockfile
(`uv.lock`) plus the pinned `requirements.txt` give an exact, reproducible
environment (Python >= 3.12).

```bash
# with uv (recommended)
uv sync                                                       # create .venv from uv.lock

uv run python tests/test_core.py        # 1. correctness tests (custom solvers vs numpy/scipy, convergence rates)
uv run python demo_report.py            # 2. report artifacts (convergence plot, simulation gallery, agent transcript)
uv run python generate_report.py        # 3. full scope-compliance audit -> FULL_REPORT.md

uv run python main.py "Solve a steady-state heat problem in a copper plate, 373K left, 273K right"
uv run python main.py                   # interactive mode
```

Without uv, use the pinned requirements file (same resolved versions):

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_core.py && python demo_report.py && python generate_report.py
```

Shortcuts: `make install run demo test report` (see Makefile), or
`make docker-build && make docker-run` for the containerized CLI.

By default there's no LLM configured, so the agents run a
**deterministic offline stub** that follows the same tool sequence
a real ReAct agent would (parse request -> check warnings -> validate
-> run -> report), which is enough to exercise the full LangGraph
graph, the knowledge-graph warm-start, and the FEM solver end to
end without any API key.

To use a real LLM for actual agent *reasoning* (recommended for the
final submission/demo, since it's the closest match to the paper's
architecture):

```bash
export ANTHROPIC_API_KEY=sk-...
python3 main.py "Solve a steady-state heat problem in a copper plate, 373K left, 273K right"
```

## Suggested course-project extension (beyond reproduction)

Per the assignment's "Methodological Extension" track: the paper's
`EXPLICIT_CFL_VIOLATION` rule is a static formula check. This
project already wires in a **power-method-based dynamic stability
estimate** (`fem/solver.py`, computed automatically for every
transient run) — a natural next step is to have the Simulation
Agent actually *act* on that estimate (auto-shrink `dt` when the
estimated spectral radius implies instability) rather than only
reporting it, and to benchmark accuracy/wall-time against the
paper's static-rule baseline across the same 50-task-style ablation
methodology (Section 6 of the paper).

## File map

```
fem/               mesh, assembly, custom linear solvers, solver, V&V
knowledge_graph/   graph + embedding store, seed materials, rule engine
database/          SQLite run persistence
agents/            LLM factory, tools, specialist agents, LangGraph supervisor
tests/test_core.py correctness tests
demo_report.py     generates outputs/ (plots + transcripts)
generate_report.py full scope-compliance audit -> FULL_REPORT.md
main.py            CLI entrypoint
outputs/           generated artifacts (git-ignored recommended)
```

## Recreating this project on another machine

Everything needed is inside the repo:

| Need | File | Notes |
|---|---|---|
| Dependency spec (loose) | `pyproject.toml` | uv's project definition, Python >= 3.12 |
| Exact locked environment | `uv.lock` | `uv sync` reproduces the tested versions byte-for-byte |
| pip fallback (pinned) | `requirements.txt` | autogenerated from the lock via `uv export`; `pip install -r requirements.txt` |
| API keys | `.env.example` | `cp .env.example .env` and set one provider key — or leave blank for the offline stub |
| Container | `Dockerfile`, `docker-compose.yml` | uv-based image; DB persisted in a named volume |

Steps on a fresh machine (needs only git + uv, or git + Python 3.12+):

```bash
git clone <this-repo> && cd pde-agents-lite
cp .env.example .env          # optionally add an API key
uv sync                       # or: pip install -r requirements.txt
uv run python tests/test_core.py
uv run python demo_report.py  # regenerates all plots/transcripts into outputs/
```

`runs.db` (run history) and everything in `outputs/` are generated data —
they are rebuilt by the commands above, not shipped. The base paper
(`2606.07850v2.pdf`) is reference material only.

## Walkthrough

A step-by-step guide for anyone cloning this for the first time, running on a
plain laptop with no GPU, no database server, and no required API key.

### Prerequisites

| Tool | Minimum | Notes |
|---|---|---|
| Python | 3.12+ | Check: `python3 --version` |
| [uv](https://docs.astral.sh/uv/) | any recent | `curl -LsSf https://astral.sh/uv/install.sh \| sh`; or use plain pip (see below) |
| git | any | for cloning |

An Anthropic / Google / OpenAI API key is **optional** — the project runs
fully offline without one using a deterministic stub (see Step 4).

---

### Step 1 — Clone and set up the environment

```bash
git clone <this-repo> && cd pde-agents-lite
cp .env.example .env        # creates your local config; safe to leave all keys blank for now
uv sync                     # resolves uv.lock into .venv — takes ~30 s on first run
```

**What happened:** `uv sync` reads the exact pinned versions from `uv.lock` and
installs them into `.venv/`. Every dependency (LangChain, LangGraph, NumPy,
SciPy, scikit-learn, matplotlib, rich) is locked — you get the same
environment that was tested, byte-for-byte.

No uv? Use the pip fallback instead:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

### Step 2 — Run the correctness tests

```bash
uv run python tests/test_core.py
# shortcut: make test
```

**What to expect:**

```
[PASS] Gauss elimination matches scipy (rel err < 1e-6)
[PASS] LU decomposition matches scipy (rel err < 1e-6)
[PASS] Power method eigenvalue within 1% of numpy.linalg.eig
[PASS] Steady linear profile — L2 error at machine precision
[PASS] Transient Fourier decay — convergence rate 2.13 (expected ~2.0)
[PASS] Steady Poisson (const. source) — convergence rate 2.00
```

This validates that the custom Gauss elimination, LU decomp, and P1 FEM
assembly are all correct before touching the agent layer. All six tests should
pass with no API key required.

---

### Step 3 — Generate report artifacts

```bash
uv run python demo_report.py
# shortcut: make demo
```

**What happens:** runs two offline agent conversations (one steady-state, one
transient), exercises the full LangGraph graph (supervisor → simulation agent →
database agent → analytics agent), and writes into `outputs/`:

- `convergence_study.png` — log-log convergence plot (paper Fig. 5 equivalent)
- `solver_validation.txt` — solver cross-validation numbers
- `simulation_gallery.png` — temperature field plots across mesh refinements
- `agent_transcript.txt` — the complete tool-call sequence the agents executed

Open `outputs/convergence_study.png` to confirm the O(h²) slope visually.

---

### Step 4 — Run a simulation via the CLI

**Offline (no API key needed):**

```bash
uv run python main.py "Solve a steady-state heat problem in a copper plate, 373K left, 273K right"
# shortcut: make demo (runs the same string)
```

The offline stub parses the request deterministically and executes the same
tool sequence a real ReAct agent would: knowledge-graph warm-start → config
warnings check → validation → FEM solve → report. You'll see a Rich-formatted
panel with mesh info, solver stats, and a max-temperature summary.

**With a real LLM (optional, closer to the paper):**

Edit `.env` and set one key:

```
ANTHROPIC_API_KEY=sk-ant-...    # claude-sonnet-4-6 by default
# or GOOGLE_API_KEY=...         # gemini-2.0-flash by default
# or OPENAI_API_KEY=sk-...      # gpt-4o by default
```

Then re-run the same command. The supervisor now reasons over the request,
routes to the right specialist, and calls tools via actual LLM inference. The
numerical output is identical; only the routing logic is now LLM-driven rather
than deterministic.

**Interactive mode:**

```bash
uv run python main.py
# shortcut: make run
```

Starts a REPL. Try these prompts in sequence to exercise all three agents:

```
> Solve a steady-state heat problem in a copper plate, 373K left, 273K right
> Now run the same case transiently for 10 seconds
> Show me the last 5 simulation runs
> What is the L2 error for the last run?
```

---

### Step 5 — Inspect run history

Every simulation is stored in `runs.db` (SQLite). The Database Agent can query
it conversationally, but you can also inspect it directly:

```bash
sqlite3 runs.db "SELECT id, material, solver_type, max_temp, timestamp FROM runs ORDER BY timestamp DESC LIMIT 5;"
```

---

### Step 6 (optional) — Docker

If you'd rather not install Python locally:

```bash
make docker-build              # builds the uv-based image (~2 min first time)
make docker-run                # interactive CLI inside the container
make docker-demo               # runs the copper-plate demo and exits
```

The database is mounted as a named Docker volume so run history persists across
`docker-compose run` invocations.

---

### Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `ModuleNotFoundError: langchain_anthropic` | env not activated | `source .venv/bin/activate` or prefix with `uv run` |
| `ANTHROPIC_API_KEY not set` warning | key missing in `.env` | add key or leave blank for offline stub |
| `runs.db` permission error in Docker | volume ownership mismatch | `docker compose down -v && make docker-build` |
| convergence test rate outside expected range | NumPy/SciPy version drift | `uv sync` to restore the locked versions |
