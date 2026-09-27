"""End-to-end twinning-replicability test for the multi-meal T1D model.

The multi-meal counterpart of ``test_single_meal_model.py``. MAP twinning must
be reproducible: running :meth:`Twinner.twin` twice with the same configuration
returns byte-identical parameters. This guards against non-determinism creeping
into the optimisation (unseeded sampling, order-dependent parallelism, model
state leaking between runs).

The check is done **per** ``n_starts`` setting, not across settings. Two runs
at ``n_starts=1`` must match each other, and two runs at ``n_starts=4`` must
match each other. ``n_starts=1`` and ``n_starts=4`` are *not* expected to agree
with one another: each start ``i`` is seeded by its index, so ``n_starts=4``
explores starts 0-3 and keeps a better optimum than start 0 alone — correct
behavior, not a bug.

The test also pins the twinned parameters as **golden values** (``GOLDEN_X``),
so the fit can't silently change between suite runs. :meth:`Twinner.twin`
returns a single best vector regardless of ``n_starts`` (it keeps the
lowest-objective start), so there is one golden vector per setting. Regenerate
``GOLDEN_X`` (rerun this test and repaste ``result["x"]``) if the physiological
model, the optimiser, or a numerical dependency (scipy/numpy/BLAS/Numba)
legitimately changes.

The run fits the full 18-parameter prior (per-meal ``SI_*`` / ``kabs_*`` /
``beta_*`` channels) over ~22 hours of data
(``example/data_two_day_extended.parquet`` sliced to 04:00 of the second day,
265 5-minute samples), so it pays a one-time Numba JIT cost and is heavier than
the single-meal test.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_replay_bg.distributions import Gamma, LogNormal, Normal, Uniform
from py_replay_bg.data.multi_meal_t1d_data import MultiMealT1DData
from py_replay_bg.model.multi_meal_t1d import MultiMealT1DModel
from py_replay_bg.twinner.twinner import Twinner


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"
# 04:00 of the second day of data_two_day_extended (which starts 2027-05-11 06:00).
SECOND_DAY_4AM = pd.Timestamp("2027-05-12 04:00:00")

# Golden twinned parameters (order matches _prior()). One best-fit vector per
# n_starts setting. Captured from a real run; see the module docstring for
# when/how to regenerate.
#
# Regenerated after switching Gamma's _gammaln from a Lanczos approximation to
# math.lgamma and tightening _log_prior's truncation bounds (both already
# uncommitted before this pin was last refreshed) — Powell on this prior's
# ill-conditioned ridge lands somewhere qualitatively different.
GOLDEN_X = {
    1: np.array([
        126.80633524477847, 0.024407954269182332, 0.5370163552710369,
        0.16591203452037923, 0.008962248071978336, 0.008938494774401378,
        0.04808158990528963, 0.0006343962291786888, 0.0004462448255138157,
        0.0005634914549740268, 0.09089820024577186, 0.20335637700794595,
        0.10632502363152713, 0.011120667428711026, 1.0, 1.0, 1.0, 1.0,
    ]),
    4: np.array([
        119.03990195277775, 0.022165922838110346, 0.8101626459807986,
        0.02422029517023964, 0.016112372679134675, 0.01193093892611895,
        0.22969533584998553, 0.0006422656254544341, 0.0006384016328639799,
        0.0006458118432864122, 0.012321851797699726, 0.012224760913899143,
        0.01203785426688198, 0.012185755681786523, 2.0, 1.0, 1.0, 1.0,
    ]),
}


def _prior():
    """The 18-parameter multi-meal prior used for the replicability check."""
    return {
        "Gb": {"prior": Normal(mu=119.13, sigma=7.11), "min": 70, "max": 150},
        "SG": {"prior": LogNormal(mu=-3.8, sigma=0.05), "min": 0, "max": 0.5},
        "f": {"prior": Normal(mu=0.8, sigma=0.05), "min": 0, "max": 1},
        "p2": {"prior": Normal(mu=0.11, sigma=0.05), "min": 0, "max": 0.5},
        "ka2": {"prior": LogNormal(mu=-4.2875, sigma=0.4274), "min": 0, "max": 0.5},
        "kd": {"prior": LogNormal(mu=-3.5090, sigma=0.6187), "min": 0, "max": 0.5},
        "kempt": {"prior": LogNormal(mu=-1.9646, sigma=0.7069), "min": 0, "max": 0.75},
        "SI_B": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
        "SI_L": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
        "SI_D": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
        "kabs_B": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_L": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_D": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_S": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "beta_B": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_L": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_D": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_S": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
    }


@pytest.fixture
def mm_extended_setup(env):
    """A multi-meal data object + fresh model over the sliced extended trace.

    Loads ``example/data_two_day_extended.parquet`` and keeps everything up to
    (and including) 04:00 of the second day — a ~22-hour, 265-sample window.
    The multi-meal model needs ``t_start`` (minutes past midnight of the first
    sample) to align its time-of-day insulin sensitivity.
    """
    df = pd.read_parquet(EXAMPLE_DIR / "data_two_day_extended.parquet")
    df["t"] = pd.to_datetime(df["t"])
    df = df[df["t"] <= SECOND_DAY_4AM].reset_index(drop=True)

    rbg_data = MultiMealT1DData(data=df, body_weight=100, environment=env)
    t_start = int(
        (df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60
    )
    model = MultiMealT1DModel(
        u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start
    )
    return model, rbg_data


@pytest.mark.slow
@pytest.mark.parametrize("n_starts", [1, 4])
def test_twin_is_replicable(mm_extended_setup, n_starts):
    model, rbg_data = mm_extended_setup
    prior = _prior()

    twinner = Twinner(parallelize=False, n_starts=n_starts, verbose=False)
    r1 = twinner.twin(model, rbg_data, prior)
    r2 = twinner.twin(model, rbg_data, prior)

    # Reproducible within this run: two twins give byte-identical parameters.
    np.testing.assert_array_equal(r1["x"], r2["x"])
    assert r1["fun"] == r2["fun"]

    # Reproducible across suite runs: parameters match the pinned golden values.
    np.testing.assert_allclose(r1["x"], GOLDEN_X[n_starts], rtol=1e-6)
