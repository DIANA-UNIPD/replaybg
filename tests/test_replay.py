"""Tests for ``ReplayBG.replay`` (the forward-simulation entry point).

Focuses on the data-seeding contract added to ``replay()``: before simulating,
it must re-seed the model's cold-start glucose from the first observed value
and the t=0 input row from the recorded data, via ``Model.reset``'s ``g0``/
``u0`` (see the model classes and ``Model.reset`` for the full contract).
"""
import numpy as np
import pytest

from py_replay_bg.data.single_meal_t1d_data import SingleMealT1DData
from py_replay_bg.model.single_meal_t1d import SingleMealT1DModel
from py_replay_bg.replaybg import ReplayBG


@pytest.fixture
def sm_setup(single_meal_df, env):
    """A prepared single-meal data object plus a fresh model over it.

    ``single_meal_df``'s first glucose sample (120.0) differs from the model's
    steady-state ``Gb`` default (119.13), so a seeded replay is distinguishable
    from an unseeded one.
    """
    rbg_data = SingleMealT1DData(data=single_meal_df, body_weight=100.0, environment=env)
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)
    return model, rbg_data


def test_replay_seeds_model_from_first_observed_glucose(sm_setup):
    model, rbg_data = sm_setup
    g0 = float(rbg_data.y[rbg_data.y_idxs[0]])
    assert g0 != pytest.approx(model.Gb)  # otherwise a missing seed would go unnoticed

    result = ReplayBG(plot_mode=False, verbose=False).replay(rbg_data, model=model)

    # The model itself is re-seeded, not just the returned trace.
    assert model.G[0] == pytest.approx(g0)
    assert model.IG[0] == pytest.approx(g0)
    np.testing.assert_array_equal(model.u[:, 0], rbg_data.u[0])

    # output[0] is model.output(0) taken right after the seed, before any step.
    assert result["output"][0] == pytest.approx(g0)
    np.testing.assert_array_equal(result["input"][0], rbg_data.u[0])
