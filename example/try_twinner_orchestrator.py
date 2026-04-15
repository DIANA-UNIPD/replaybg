
from multiprocessing import freeze_support

import os
import pandas as pd

from data.multi_meal_t1d_data import MultiMealT1DData
from model.multi_meal_t1d import MultiMealT1DModel
from distributions import Normal, Gamma, LogNormal, Uniform

from replaybg import ReplayBG
from orchestrator.twinner_orchestrator import TwinnerOrchestrator
from utils.numba_dicts import to_typed_f32_dict

if __name__ == '__main__':
    freeze_support()

    df = pd.read_parquet("data_day_1_2.parquet")
    df['t'] = pd.to_datetime(df['t'])

    save_folder = os.path.join(os.path.abspath(''))

    rbg = ReplayBG(save_folder=save_folder)

    unknown_parameters_prior = {
        'Gb': {'prior': Normal(mu=119.13, sigma=7.11), 'min': 70, 'max': 150},
        'SG': {'prior': LogNormal(mu=-3.8, sigma=0.5), 'min': 0, 'max': .5},
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
        n_starts=4,
        parallelize=True,
        n_jobs=-1,
    )

    results = orchestrator.twin()

    for r in results:
        print(f"Segment {r['segment_index']}: {r['segment_start']} -> {r['segment_end']}")
        print(f"  Parameters estimated: {list(r['prior_used'].keys())}")
        print(f"  theta: {r['theta']}")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(df['t'], df['glucose'], color='tab:gray', linewidth=1.5, label='CGM')

    prev_x0 = None
    prev_theta = None
    for r in results:
        seg_df = df[(df['t'] >= r['segment_start']) & (df['t'] <= r['segment_end'])].copy()

        x0_setup = None
        if prev_x0 is not None:
            x0_setup = MultiMealT1DModel.setup_x0(x0=prev_x0, previous_theta=prev_theta)

        out = rbg.replay(data=seg_df, theta=r['theta'], bw=100,
                         save_name=f"replay_{r['segment_index']:04d}",
                         x0_setup=x0_setup)

        # Extract final state for carry-over into the next segment's replay
        rbg_data = MultiMealT1DData(data=seg_df, environment=rbg.environment)
        theta_typed = to_typed_f32_dict(r['theta'])
        final_model = MultiMealT1DModel(u2ss=rbg_data.u2ss, theta0=theta_typed, tsteps=rbg_data.tsteps)
        for t in range(1, rbg_data.tsteps):
            final_model.step(rbg_data.u[t], t)
        prev_x0 = MultiMealT1DModel.extract_final_x0(final_model)
        prev_theta = r['theta']

        label = f"Segment {r['segment_index']}"
        ax.plot(seg_df['t'], out, linewidth=1.5, label=label)

    ax.set_xlabel('Time')
    ax.set_ylabel('Glucose (mg/dL)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
