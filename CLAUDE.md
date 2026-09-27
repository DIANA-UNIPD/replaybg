# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

## Commands

```bash
# Install dependencies
uv sync

# Run the test suite
uv run pytest

# Skip the slow end-to-end twinning tests
uv run pytest -m "not slow"

# Run an example script
uv run python example/twin_single_meal.py

# Build docs
uv run mkdocs build
uv run mkdocs serve
```

## Testing

A pytest suite lives in `tests/`, covering the distributions, the `Environment`,
the data classes, the `Twinner` and the plotting utilities. The end-to-end
twinning tests are marked `slow` and pin golden parameter values, so they fail if
the model, optimiser or dependencies drift; deselect them with `-m "not slow"`
for a fast loop. `pythonpath = ["."]` in `pyproject.toml` puts the repo root on
`sys.path`, since modules are imported as `from model import ...` without an
install step. See `tests/README.md` for the per-module breakdown.

The Numba `@jitclass` model classes have no direct unit tests — they are covered
only indirectly, through the `slow` end-to-end twinning runs. Running the example
scripts in `example/` remains the way to validate a change end to end.

## Architecture

ReplayBG is a digital-twin framework for Type 1 Diabetes (T1D) glucose dynamics. It fits a physiological ODE model to CGM + insulin + meal data (the **twinning** step), then uses the fitted model to simulate counterfactual scenarios (the **replay** step).

### Core flow

```
raw DataFrame
    │
    ▼
*Data class* (SingleMealT1DData / MultiMealT1DData)
    │  converts raw columns → tsteps, u[t] input matrix, y observations
    ▼
*Model class* (SingleMealT1DModel / MultiMealT1DModel)
    │  Numba jitclass — exposes reset(theta_dict), step(u, t), output(t)
    ▼
Twinner.twin()
    │  MAP estimation via multi-start Powell optimisation (scipy.optimize.minimize)
    │  objective = negative log-posterior = -(log_prior + log_likelihood)
    │  can parallelise starts with multiprocessing
    ▼
ReplayBG.replay()
    │  forward simulation with the estimated theta
    ▼
predicted glucose trace (np.ndarray)
```

### Key modules

| Path | Role |
|------|------|
| `replaybg.py` | `ReplayBG` — top-level API: `.twin()` and `.replay()` |
| `twinner/twinner.py` | `Twinner` — MAP optimiser with multi-start + optional parallelism |
| `model/multi_meal_t1d.py` | Multi-meal T1D ODE model (Numba jitclass); supports B/L/D/S/H meal labels and time-of-day insulin sensitivity (SI_B, SI_L, SI_D) |
| `model/single_meal_t1d.py` | Simplified single-meal variant |
| `data/multi_meal_t1d_data.py` | Prepares multi-meal DataFrames → `rbg_data` consumed by model/twinner |
| `data/single_meal_t1d_data.py` | Single-meal variant |
| `distributions/` | Prior distributions (Normal, Gamma, LogNormal, Uniform) with `.evaluate()` and `.sample()` |
| `environment/__init__.py` | `Environment` dataclass; also exports `jitclass_` decorator (enables/disables Numba JIT compilation) |
| `utils/numba_dicts.py` | Helpers for converting plain Python dicts to Numba typed dicts (`to_typed_f32_dict`) |
| `utils/plot_twinning_history.py` | Diagnostic plot of the optimisation history |

### Model interface contract

Every model class must implement:
- `__init__(u2ss, tsteps, ...)` — allocate state arrays
- `reset(theta_dict)` — set parameters from a Numba typed dict and re-initialise state
- `step(u, t)` — advance ODE by one time step
- `output(t)` — return glucose at time t
- `apply_x0(model, data, x0, previous_theta)` — static; applies carry-over state directly
- `extract_final_x0(model_instance)` — static; extracts end-of-segment state for carry-over

### Unknown parameters prior dict

The `unknown_parameters_prior` dict passed to `ReplayBG.twin()` defines which parameters to estimate:

```python
{
    'param_name': {
        'prior': <distribution>,   # must have .evaluate(val) and .sample(min, max, seed)
        'min': float,
        'max': float,
        'integer': bool,           # optional, defaults to False
    },
    ...
}
```

### Numba JIT

Model classes are compiled with Numba's `@jitclass`. The `jitclass_` decorator from `environment/__init__.py` is the project's switch — it either applies real JIT compilation or falls back to a no-op for debugging. All model state arrays and scalar fields must be declared in `JITCLASS_SPEC` with explicit Numba types. Parameter passing between Python and jitclass always goes through Numba typed dicts (`numba.typed.Dict`).
