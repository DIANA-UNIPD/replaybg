
from multiprocessing import freeze_support

import os
import pandas as pd

from data.multi_meal_t1d_data import MultiMealT1DData
from model.multi_meal_t1d import MultiMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform

from replaybg import ReplayBG
from callbacks import CorrectionBolus, HypoTreatment
from utils.numba_dicts import to_typed_f32_dict

import matplotlib.pyplot as plt

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
    rbg_data = MultiMealT1DData(data=df, environment=rbg.environment)

    # Initialize model
    t_start = int((df["t"].iloc[0] - df["t"].iloc[0].normalize()).total_seconds() / 60)
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps, t_start=t_start)

    unknown_parameters_prior = {
        'Gb': {'prior': Normal(mu=119.13, sigma=7.11), 'min': 70, 'max': 150},
        #'SG': {'prior': LogNormal(mu=-3.8, sigma=0.05), 'min': 0, 'max': .5},
        #'p2': {'prior': Normal(mu=0.11, sigma=0.05), 'min': 0, 'max': .5},
        #'ka2': {'prior': LogNormal(mu=-4.2875, sigma=0.4274), 'min': 0, 'max': .5},
        #'kd': {'prior': LogNormal(mu=-3.5090, sigma=0.6187), 'min': 0, 'max': .5},
        #'kempt': {'prior': LogNormal(mu=-1.9646, sigma=0.7069), 'min': 0, 'max': .75},
        'SI_B': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'SI_L': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        'SI_D': {'prior': Gamma(alpha=3.3, beta=1 / 5e-4), 'min': 0, 'max': .1},
        #'kabs_B': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        #'kabs_L': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        #'kabs_D': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        #'kabs_S': {'prior': LogNormal(mu=-5.4591, sigma=1.4396), 'min': 0, 'max': .5},
        #'beta_B': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        #'beta_L': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        #'beta_D': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
        #'beta_S': {'prior': Uniform(a=0, b=60), 'min': 0, 'max': 60, 'integer': True},
    }
    result = rbg.twin(rbg_data=rbg_data,
                      model=model,
                      unknown_parameters_prior=unknown_parameters_prior,
                      parallelize=True, n_jobs=-1, n_starts=1,
                      path=save_folder, name=save_name)
    theta_estimated = result['theta']

    # --- Baseline replay (no policy) ---
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                              theta0=to_typed_f32_dict(theta_estimated), t_start=t_start)
    baseline = rbg.replay(rbg_data=rbg_data, model=model, path=save_folder, name='baseline')

    # --- Replay with a correction-bolus callback ---
    correction_bolus_callback = CorrectionBolus(threshold=180, target=120, cf=20, lockout_min=60)
    hypotreatment_callback = HypoTreatment(threshold=70, carbs=15, lockout_min=15)
    model = MultiMealT1DModel(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps,
                              theta0=to_typed_f32_dict(theta_estimated), t_start=t_start)
    controlled = rbg.replay(rbg_data=rbg_data, model=model, path=save_folder, name='controlled',
                            callbacks=[correction_bolus_callback, hypotreatment_callback],)

    # Exhaustive log of what the callback did, as a tidy table
    actions = pd.DataFrame(controlled['actions'])
    print(actions)

    # Plot baseline vs. controlled glucose, and the applied bolus channel
    bolus_idx = next(i for i, n in controlled['data_to_input'].items() if n == 'bolus')
    fig, (ax_g, ax_b) = plt.subplots(2, 1, sharex=True)
    ax_g.plot(baseline['output'], label='baseline')
    ax_g.plot(controlled['output'], label='with correction bolus')
    ax_g.axhline(180, color='r', ls='--', lw=0.8)
    ax_g.set_ylabel('glucose [mg/dL]')
    ax_g.legend()
    ax_b.plot(controlled['input'][:, bolus_idx])
    ax_b.set_ylabel('bolus [mU/(kg·min)]')
    ax_b.set_xlabel('sample')
    plt.show()
