
from multiprocessing import freeze_support

import os
import pandas as pd

from data.multi_meal_t1d_data import MultiMealT1DData
from model.multi_meal_t1d import MultiMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform

from replaybg import ReplayBG

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
    rbg = ReplayBG(save_folder=save_folder)

    # Create data in required format
    rbg_data = MultiMealT1DData(data=df,
                                environment=rbg.environment)

    # Initialize model
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)

    unknown_parameters_prior = {
        'Gb': {'prior': Normal(mu=119.13, sigma=7.11), 'min': 70, 'max': 150},
        'SG': {'prior': LogNormal(mu=-3.8, sigma=0.5), 'min': 0, 'max': .5},
        'p2': {'prior': Normal(mu=0.11, sigma=0.004), 'min': 0, 'max': .5},
        'ka2': {'prior': LogNormal(mu=-4.2875, sigma=0.4274), 'min': 0, 'max': .5},
        'kd': {'prior': LogNormal(mu=-3.5090, sigma=0.6187), 'min': 0, 'max': .5},
        'kempt': {'prior': LogNormal(mu=-1.9646, sigma=0.7069), 'min': 0, 'max': .75},
        'SI_B': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'SI_L': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'SI_D': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'kabs_B': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'kabs_L': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'kabs_D': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'kabs_S': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        'beta_B': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        'beta_L': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        'beta_D': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        'beta_S': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
    }

    theta_estimated = rbg.twin(data=rbg_data, bw=100,
                               model=model,
                               save_name=save_name,
                               unknown_parameters_prior=unknown_parameters_prior,
                               parallelize=True, n_jobs=-1, n_starts=32)
    print(theta_estimated)
    out = rbg.replay(data=df,theta=theta_estimated, bw=100, save_name=save_name)

    # TODO: add a plot utility
    import matplotlib.pyplot as plt
    plt.plot(df.glucose)
    plt.plot(out)
    plt.show()