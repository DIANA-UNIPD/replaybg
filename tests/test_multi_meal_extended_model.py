"""End-to-end twinning-replicability test for the multi-meal *extended* model.

The extended counterpart of ``test_multi_meal_model.py``. MAP twinning must be
reproducible: running :meth:`Twinner.twin` twice with the same configuration
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
returns a single best vector regardless of ``n_starts``, so there is one golden
vector per setting. Regenerate ``GOLDEN_X`` (rerun this test and repaste
``result["x"]``) if the physiological model, the optimiser, or a numerical
dependency (scipy/numpy/BLAS/Numba) legitimately changes.

Prior: the 18-parameter multi-meal prior plus seven second-occurrence channels
(``kabs_B2, kabs_L2, kabs_S2, beta_B2, beta_L2, beta_S2, SI_B2``) → 25
parameters. Unlike the other two tests, this one uses the *entire*
``data_two_day_extended`` trace (both days), so the second-day B2/S2 meals are
present and their channels are exercised. (There is no L2 meal in the data, so
``kabs_L2`` / ``beta_L2`` stay unconstrained — inherent to the trace.)

The run fits 25 parameters over ~29 hours of data
(``example/data_two_day_extended.parquet``, 349 5-minute samples), so it is the
heaviest test in the suite and pays a one-time Numba JIT cost.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_replay_bg.distributions import Gamma, LogNormal, Normal, Uniform
from py_replay_bg.data.multi_meal_extended_t1d_data import MultiMealExtendedT1DData
from py_replay_bg.model.multi_meal_extended_t1d import MultiMealExtendedT1DModel
from py_replay_bg.twinner.twinner import Twinner


EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"

# Golden twinned parameters (order matches _prior()). One best-fit vector per
# n_starts setting. Captured from a real run; see the module docstring for
# when/how to regenerate.
GOLDEN_X = {
    1: np.array([
        123.66108666319761, 0.03117696847089137, 0.8258581009682973,
        0.12989436767019544, 0.019044807518641314, 0.005390507631454614,
        0.03341269252196987, 0.0008991252279006548, 0.0006512216811650557,
        0.0007844273542782817, 0.11371809669730057, 0.23586661576120246,
        0.15328513198018823, 0.008957687957458332, 0.0, 0.0, 0.0, 0.0,
        0.2911144159976012, 0.0005359080722202244, 0.10746757260720276,
        0.0, 33.0, 0.0, 0.0008319821826200969,
    ]),
    4: np.array([
        121.8225518620203, 0.023481364360537414, 0.7767550149445519,
        0.03427572193463663, 0.020948023494642816, 0.007887049157315433,
        0.16794539720871912, 0.0006998681315958133, 0.0006140873718122818,
        0.0006518098207281177, 0.01413102176103931, 0.014979970093434138,
        0.014692437030151717, 0.009828147952179287, 4.0, 0.0, 0.0, 0.0,
        0.014503426883922036, 0.000535907786253652, 0.013717549922735222,
        0.0, 26.0, 0.0, 0.0006956058838832905,
    ]),
}


def _prior():
    """The 25-parameter multi-meal-extended prior used for the check.

    The 18-parameter multi-meal prior plus the seven second-occurrence channels
    (``kabs_B2, kabs_L2, kabs_S2, beta_B2, beta_L2, beta_S2, SI_B2``).
    """
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
        "kabs_B2": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_L2": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "kabs_S2": {"prior": LogNormal(mu=-5.4591, sigma=1.4396), "min": 0, "max": 0.5},
        "beta_B2": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_L2": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "beta_S2": {"prior": Uniform(a=0, b=60), "min": 0, "max": 60, "integer": True},
        "SI_B2": {"prior": Gamma(alpha=3.3, beta=1 / 5e-4), "min": 0, "max": 0.1},
    }


@pytest.fixture
def mme_extended_setup(env):
    """A multi-meal-extended data object + fresh model over the full trace.

    Loads the entire ``example/data_two_day_extended.parquet`` (~29 h, both
    days), so the second-day B2/S2 meals are present and their channels are
    exercised. The extended model needs ``t_start`` (minutes past midnight of
    the first sample) to align its time-of-day insulin sensitivity.
    """
    df = pd.read_parquet(EXAMPLE_DIR / "data_two_day_extended.parquet")
    df["t"] = pd.to_datetime(df["t"])

    rbg_data = MultiMealExtendedT1DData(data=df, body_weight=100, environment=env)
    t_start = int(
        (df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60
    )
    model = MultiMealExtendedT1DModel(
        u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start
    )
    return model, rbg_data


@pytest.mark.slow
@pytest.mark.parametrize("n_starts", [1, 4])
def test_twin_is_replicable(mme_extended_setup, n_starts):
    model, rbg_data = mme_extended_setup
    prior = _prior()

    twinner = Twinner(parallelize=False, n_starts=n_starts, verbose=False)
    r1 = twinner.twin(model, rbg_data, prior)
    r2 = twinner.twin(model, rbg_data, prior)

    # Reproducible within this run: two twins give byte-identical parameters.
    np.testing.assert_array_equal(r1["x"], r2["x"])
    assert r1["fun"] == r2["fun"]

    # Reproducible across suite runs: parameters match the pinned golden values.
    np.testing.assert_allclose(r1["x"], GOLDEN_X[n_starts], rtol=1e-6)
