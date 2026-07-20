# Changelog

All notable changes to `py-replay-bg` are documented here. This project follows
[PEP 440](https://peps.python.org/pep-0440/) versioning; pre-releases (e.g. `2.0.0b1`) are
**not** installed by a plain `pip install` — pass `--pre` to opt in.

## 2.0.0b1 — unreleased (beta)

First pre-release of the **2.x line**, a ground-up refactor of
[`py_replay_bg`](https://github.com/gcappon/py_replay_bg) 1.x. Published under the same PyPI
name (`py-replay-bg`) as a pre-release, so 1.x remains the default install during the beta.

### Migrating from 1.x

- **Python**: now requires **≥ 3.12** (1.x required ≥ 3.11).
- **Imports**: the public entry point is unchanged — `from py_replay_bg import ReplayBG` still
  works. Internal module paths are now package-qualified, e.g.
  `from py_replay_bg.model.multi_meal_t1d import MultiMealT1DModel`,
  `from py_replay_bg.data.multi_meal_t1d_data import MultiMealT1DData`.
- **API**: the twinning/replay workflow was reworked (see the README "Get started" tour and
  the `example/` scripts). Review your scripts against the 2.0 examples before upgrading.
- **Staying on 1.x**: keep using `pip install py-replay-bg` (resolves to the latest stable
  1.x). Once `2.0.0` final ships it becomes the default — pin `py-replay-bg<2` to stay on 1.x.

---

## 1.x

Maintained on the [`gcappon/py_replay_bg`](https://github.com/gcappon/py_replay_bg) repository.
Bug-fix releases (`1.x.z`) continue during the 2.0 beta period. See that repository's history
for 1.x release notes.
