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

# Run an example script (from repo root)
python example/try_twin_multi_meal.py
python example/try_twin_single_meal.py

# Serve documentation locally
uv run mkdocs serve
```

There is no test suite yet. Scripts under `example/` serve as integration tests.

## Architecture

ReplayBG is a Python library for fitting ("twinning") a personalized physiological model to CGM data from a type 1 diabetic patient and then simulating ("replaying") glucose trajectories under alternative therapy scenarios.

### High-level flow

1. **`ReplayBG`** (`replaybg.py`) — top-level entry point. Two public methods:
   - `twin(data, ...)` — estimates model parameters via MAP optimization
   - `replay(data, theta, ...)` — runs a forward simulation with given parameters

2. **`Twinner`** (`twinner/twinner.py`) — MAP estimation engine. Runs `n_starts` parallel Powell optimizations (via `multiprocessing`) and picks the lowest negative log-posterior. The objective is `log_prior + log_likelihood`, where the likelihood uses a Gaussian error model with 5% constant CV.

3. **`MultiMealT1DModel`** (`model/multi_meal_t1d.py`) — Numba `jitclass` implementing the ODE model. Extends a Hovorka/UVA-Padova structure with five separate meal-absorption chains (Breakfast, Lunch, Dinner, Snack, Hypo-treatment), time-of-day-varying insulin sensitivity (SI_B/SI_L/SI_D), and a non-symmetric hypoglycaemia risk function. Integration is Backward Euler (Gauss-Seidel order). Interface: `reset(theta0)`, `step(u, t)`, `output(t)`.

4. **`MultiMealT1DData`** (`data/multi_meal_t1d_data.py`) — preprocesses a pandas DataFrame into the arrays consumed by the model. Expands 5-minute CGM data to 1-minute simulation resolution (`yts=5`). Normalizes insulin and meal inputs by body weight. Input columns expected: `t` (datetime), `glucose`, `basal`, `bolus`, `bolus_label`, `cho`, `cho_label` (labels: `B`, `L`, `D`, `S`, `H`).

5. **`distributions/`** — Numba `jitclass` probability distributions (`Normal`, `LogNormal`, `Gamma`, `Uniform`) plus `to_constrained`/`to_unconstrained` sigmoid transforms used by the Twinner. Each distribution must implement `evaluate(x)` and `sample(min_val, max_val, seed)`.

6. **`environment/`** — `Environment` holds runtime settings (`ts`, `seed`, `plot_mode`, `verbose`, `save_folder`). `environment/config.py` provides `DEBUG` flag: when `True`, `jitclass_` and `njit_` become no-ops so the model runs as plain Python (essential for debugging — Numba JIT compilation makes stack traces unreadable).

### Numba JIT pattern

All model classes and distribution classes are decorated with `@jitclass_(JITCLASS_SPEC)` where `jitclass_` is either `numba.experimental.jitclass` (normal) or an identity function (debug). The spec must declare every attribute with its Numba type before the class body. Typed dicts (`numba.typed.Dict`) are used to pass parameter dictionaries into JIT-compiled code; `utils/numba_dicts.py` has helpers for constructing them.

### `unknown_parameters_prior` contract

The `twin()` call receives a dict where each key is a model attribute name and each value is:
```python
{
    'prior': <distribution object with .evaluate() and .sample()>,
    'min': float,
    'max': float,
    'integer': bool  # optional — rounds value before updating model
}
```
Parameter order in this dict determines the order of the optimizer's `x` vector.

### Single-meal vs multi-meal

`model/single_meal_t1d.py` and `data/single_meal_t1d_data.py` are simpler variants with one meal-absorption chain. `MultiMealT1DModel` is the current primary model; the single-meal variant is kept for reference.