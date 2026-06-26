
from multiprocessing import freeze_support

import os
import pandas as pd

from data.single_meal_t1d_data import SingleMealT1DData
from model.single_meal_t1d import SingleMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform
from utils.plot_twinning_history import plot_twinning_history
import matplotlib.pyplot as plt
from replaybg import ReplayBG
from utils.numba_dicts import to_typed_f32_dict


if __name__ == '__main__':
    freeze_support()
    df = pd.read_parquet("data_day_1.parquet")
    df['t'] = pd.to_datetime(df['t'])
    save_folder = os.path.join(os.path.abspath(''))
    save_name = 'test'

    # Impute breakfast (clearly missing)
    df.loc[10, 'cho'] = 10
    df.loc[10, 'cho_label'] = 'B'

    # Create ReplayBg instance
    rbg = ReplayBG()

    # Create data in required format
    rbg_data = SingleMealT1DData(data=df,
                                 body_weight=100,
                                environment=rbg.environment)

    # Initialize model
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)

    unknown_parameters_prior = {
        'Gb': {'prior': Normal(mu=119.13, sigma=7.11), 'min': 70, 'max': 150},
        'SG': {'prior': LogNormal(mu=-3.8, sigma=0.5), 'min': 0, 'max': .5},
        'p2': {'prior': Normal(mu=0.11, sigma=0.004), 'min': 0, 'max': .5},
        'f': {'prior': Normal(mu=0.8, sigma=0.05), 'min': 0, 'max': 1},
        'ka2': {'prior': LogNormal(mu=-4.2875, sigma=0.4274), 'min': 0, 'max': .5},
        'kd': {'prior': LogNormal(mu=-3.5090, sigma=0.6187), 'min': 0, 'max': .5},
        'kempt': {'prior': LogNormal(mu=-1.9646, sigma=0.7069), 'min': 0, 'max': .75},
        'SI': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'kabs': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'beta': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
    }

    result = rbg.twin(rbg_data=rbg_data,
                               model=model,
                               unknown_parameters_prior=unknown_parameters_prior,
                               parallelize=True, n_jobs=-1, n_starts=16,
                               #log_history=True
                               path=save_folder, name=save_name,
                      )
    theta_estimated = result['theta']
    print(result)

    # Create data in required format
    rbg_data = SingleMealT1DData(data=df,
                                environment=rbg.environment)

    # Initialize model
    model = SingleMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, theta0=to_typed_f32_dict(theta_estimated))

    out = rbg.replay(rbg_data=rbg_data,
                     model=model,
                     path=save_folder)

    if result['history'] is not None:
        fig = plot_twinning_history(result['history'], param_names=list(unknown_parameters_prior.keys()))
        plt.show()

    # TODO: add a plot utility
    import matplotlib.pyplot as plt
    plt.plot(df.glucose)
    plt.plot(out["output"][0::rbg_data.yts])
    plt.show()