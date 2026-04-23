
from multiprocessing import freeze_support

import os
import pandas as pd

from data.multi_meal_t1d_data import MultiMealT1DData
from model.multi_meal_t1d import MultiMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform

from replaybg import ReplayBG
from orchestrator.twinner_orchestrator import TwinnerOrchestrator
from simulator.simulator_orchestrator import SimulatorOrchestrator

if __name__ == '__main__':
    freeze_support()

    df = pd.read_parquet("data_day_1_2.parquet")
    df['t'] = pd.to_datetime(df['t'])
    save_folder = os.path.join(os.path.abspath(''))

    df.loc[10, 'cho'] = 10
    df.loc[10, 'cho_label'] = 'B'

    rbg = ReplayBG(save_folder=save_folder)

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

    orchestrator = TwinnerOrchestrator(
        model_class=MultiMealT1DModel,
        data_class=MultiMealT1DData,
        data=df,
        bw=100,
        rbg=rbg,
        unknown_parameters_prior=unknown_parameters_prior,
        save_name_prefix='twin',
        n_starts=1,
        parallelize=False,
        n_jobs=-1,
    )

    results = orchestrator.twin()

    for r in results:
        print(f"Segment {r['segment_index']}: {r['segment_start']} -> {r['segment_end']}")
        print(f"  Parameters estimated: {list(r['prior_used'].keys())}")
        print(f"  theta: {r['theta']}")

    sim = SimulatorOrchestrator(
        model_class=MultiMealT1DModel,
        data_class=MultiMealT1DData,
        data=df,
        bw=100,
        rbg=rbg,
        twin_results=results,
        save_name_prefix='replay',
    )

    sim_results = sim.replay()

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df['t'], df['glucose'], color='tab:gray', linewidth=1.5, label='CGM')

    for r in sim_results:
        seg_df = df[(df['t'] >= r['segment_start']) & (df['t'] <= r['segment_end'])]
        ax.plot(seg_df['t'], r['glucose'], linewidth=1.5, label=f"Segment {r['segment_index']}")

    ax.set_xlabel('Time')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
