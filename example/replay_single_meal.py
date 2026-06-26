"""Replay-only example (single-meal model).

Loads the twinning results produced by ``twin_single_meal.py`` (the fitted ``theta``
and the prepared ``rbg_data``) and runs a forward simulation. No twinning happens here.
"""

from multiprocessing import freeze_support

import os
import matplotlib.pyplot as plt

from model.single_meal_t1d import SingleMealT1DModel
from replaybg import ReplayBG
from utils.agata_analysis import analyze_replay
from utils.load_results import load_results
from utils.numba_dicts import to_typed_f32_dict


if __name__ == '__main__':
    freeze_support()
    save_folder = os.path.join(os.path.abspath(''), 'results')
    save_name = 'single_meal_day_1'

    # Create ReplayBg instance
    rbg = ReplayBG()

    # Load the twinning results (theta + rbg_data) saved by twin_single_meal.py
    twin = load_results(save_folder, save_name, prefix='twin')
    theta_estimated = twin['theta']
    rbg_data = twin['rbg_data']

    # Initialize the model with the estimated parameters
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                               theta0=to_typed_f32_dict(theta_estimated))

    # Run replay and save the simulation
    out = rbg.replay(rbg_data=rbg_data,
                     model=model,
                     path=save_folder, save_name=save_name)

    # Analyze the replayed glucose with AGATA and fold the metrics back into the
    # saved replay_<save_name>.pkl (adds an 'analysis' key).
    analyze_replay(out, ts=5, verbose=rbg.environment.verbose,
                   path=save_folder, save_name=save_name)

    # Plot the measured glucose against the replayed output
    plt.plot(rbg_data.y, label='measured')
    plt.plot(out["output"][0::rbg_data.yts], label='replay')
    plt.ylabel('glucose [mg/dL]')
    plt.xlabel('sample')
    plt.legend()
    plt.show()
