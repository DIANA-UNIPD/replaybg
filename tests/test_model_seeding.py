"""Tests for the model-level cold-start/data-seeding contract of ``reset()``.

Each physiological model (``SingleMealT1DModel``, ``MultiMealT1DModel``,
``MultiMealExtendedT1DModel``) accepts optional ``g0``/``u0`` on ``reset()`` so
callers with real data (``replay()``, ``twin()``, ``plot_twinning``,
``analyze_twin``) can start the simulation at the observed glucose and the true
recorded t=0 input, instead of the construction-time placeholder (the basal
steady state / ``Gb``). These tests pin down exactly when that seed applies,
that it survives repeated ``reset()`` calls (the twinning hot loop), and when
it is correctly ignored (a carry-over segment, or an explicit ``x0``).
"""
import numpy as np
import pytest
from numba import float64, types
from numba.typed import Dict

from py_replay_bg.model.multi_meal_extended_t1d import MultiMealExtendedT1DModel
from py_replay_bg.model.multi_meal_t1d import MultiMealT1DModel
from py_replay_bg.model.single_meal_t1d import SingleMealT1DModel


def _f64_dict(**kwargs):
    d = Dict.empty(key_type=types.unicode_type, value_type=float64)
    for k, v in kwargs.items():
        d[k] = float(v)
    return d


# Each model class alongside the extra constructor kwargs it needs and its
# input-channel count (n_u), so the shared assertions below exercise every
# model's own _reset_u channel wiring, not just one.
MODEL_CASES = [
    pytest.param(SingleMealT1DModel, {}, 5, id="single_meal"),
    pytest.param(MultiMealT1DModel, {"t_start": 240}, 10, id="multi_meal"),
    pytest.param(MultiMealExtendedT1DModel, {"t_start": 240}, 13, id="multi_meal_extended"),
]


@pytest.mark.parametrize("model_cls, extra_kwargs, n_u", MODEL_CASES)
def test_reset_without_seed_falls_back_to_placeholder(model_cls, extra_kwargs, n_u):
    model = model_cls(u2ss=1.0, tsteps=10, **extra_kwargs)

    # No g0/u0 given: G0/IG0 fall back to Gb, and the only nonzero t=0 input
    # channel is the basal placeholder (u2ss); bolus and everything else is 0.
    assert model.G[0] == pytest.approx(model.Gb)
    assert model.IG[0] == pytest.approx(model.Gb)
    assert model.u[:, 0].sum() == pytest.approx(model.u2ss)


@pytest.mark.parametrize("model_cls, extra_kwargs, n_u", MODEL_CASES)
def test_reset_seeds_cold_start_from_data(model_cls, extra_kwargs, n_u):
    model = model_cls(u2ss=1.0, tsteps=10, **extra_kwargs)
    g0 = 150.0
    u0 = np.arange(1, n_u + 1, dtype=np.float64)  # distinct, nonzero per channel

    model.reset(model.theta0, g0, u0)

    assert model.G[0] == pytest.approx(g0)
    assert model.IG[0] == pytest.approx(g0)
    np.testing.assert_array_equal(model.u[:, 0], u0)


@pytest.mark.parametrize("model_cls, extra_kwargs, n_u", MODEL_CASES)
def test_reset_reseeds_on_every_call(model_cls, extra_kwargs, n_u):
    # Mirrors the twinning hot loop: reset() is called once per candidate
    # theta, and each call must re-seed from the *current* g0/u0, not just the
    # first one (reset() reallocates the input buffer every time).
    model = model_cls(u2ss=1.0, tsteps=10, **extra_kwargs)
    u0_a = np.full(n_u, 1.0)
    u0_b = np.full(n_u, 2.0)

    model.reset(model.theta0, 100.0, u0_a)
    assert model.G[0] == pytest.approx(100.0)
    np.testing.assert_array_equal(model.u[:, 0], u0_a)

    model.reset(model.theta0, 200.0, u0_b)
    assert model.G[0] == pytest.approx(200.0)
    np.testing.assert_array_equal(model.u[:, 0], u0_b)


@pytest.mark.parametrize("model_cls, extra_kwargs, n_u", MODEL_CASES)
def test_reset_ignores_wrong_length_u0(model_cls, extra_kwargs, n_u):
    # A u0 that doesn't match n_u (e.g. the still-unseeded default empty
    # array) is treated as "not provided", not as a malformed seed.
    model = model_cls(u2ss=1.0, tsteps=10, **extra_kwargs)
    model.reset(model.theta0, 150.0, np.empty(0, dtype=np.float64))
    assert model.u[:, 0].sum() == pytest.approx(model.u2ss)


def test_reset_ignores_g0_when_x0_has_explicit_g0():
    x0 = _f64_dict(G0=142.0, IG0=141.0)
    model = SingleMealT1DModel(u2ss=1.0, tsteps=10, x0=x0)

    # A data g0 is offered but x0 already carries an explicit initial
    # condition, which must win.
    model.reset(model.theta0, 999.0, np.zeros(5))

    assert model.G[0] == pytest.approx(142.0)
    assert model.IG[0] == pytest.approx(141.0)


def test_reset_ignores_g0_on_carry_over_segment():
    # theta_prev non-empty marks this as a carry-over segment: the true
    # initial condition is the previous segment's end state (via x0), not a
    # fresh data measurement, so g0 must be ignored even without an explicit
    # x0["G0"].
    theta_prev = _f64_dict(kempt=0.18, kabs=0.012, f=0.9, kd=0.026, ka2=0.014)
    model = SingleMealT1DModel(u2ss=1.0, tsteps=10, theta_prev=theta_prev)

    model.reset(model.theta0, 999.0, np.zeros(5))

    assert model.G[0] == pytest.approx(model.Gb)
    assert model.IG[0] == pytest.approx(model.Gb)


def test_reset_without_g0_still_seeds_u0():
    # g0 and u0 are independent: an omitted g0 (nan) must not block a given u0.
    model = SingleMealT1DModel(u2ss=1.0, tsteps=10)
    u0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    model.reset(model.theta0, np.nan, u0)

    assert model.G[0] == pytest.approx(model.Gb)
    np.testing.assert_array_equal(model.u[:, 0], u0)
