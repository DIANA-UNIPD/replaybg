from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd

from data.multi_meal_t1d_data import MultiMealT1DData

from environment import Environment
from model.multi_meal_t1d import MultiMealT1DModel
from twinner.twinner import Twinner
from utils.numba_dicts import to_typed_f32_dict


class ReplayBG:
    """Core class of ReplayBG.
    """

    def __init__(self, save_folder: str,
                 ts: int = 1,
                 seed: int = 1,
                 plot_mode: bool = True, verbose: bool = True
                 ):
        """Constructs all the necessary attributes for the ReplayBG object.
        """

        # TODO: Validate input

        # Initialize the environment parameters
        self.environment = Environment(save_folder=save_folder,
                                       ts=ts,
                                       seed=seed,
                                       plot_mode=plot_mode,
                                       verbose=verbose)

    def twin(self, rbg_data, save_name: str,
             model: object = None,
             unknown_parameters_prior: Dict = None,
             n_starts: int = 64,
             x0_setup: Optional[Callable] = None,
             parallelize: bool = False, n_jobs: int | None = None,
             ) -> Dict:
        """Runs ReplayBG twinning procedure.
        """
        # TODO: validate the inputs

        if x0_setup is not None:
            x0_setup(model, rbg_data)

        # Initialize the twinner
        twinner = Twinner(parallelize=parallelize, n_jobs=n_jobs, n_starts=n_starts)

        # Run the twinning procedure
        theta_estimated = twinner.twin(model=model,
                                       rbg_data=rbg_data,
                                       unknown_parameters_prior=unknown_parameters_prior)


        # TODO: save results before return

        return {
            'theta': dict(zip(unknown_parameters_prior.keys(), theta_estimated['x'])),
            'history': twinner.history,
        }


    def replay(self, data: pd.DataFrame, theta, bw: float, save_name: str,
               # twinning_method: str = 'mcmc', TODO: <-- this will become "custom_twinner : object = None" so user can pass their own Twinner
               # model: TODO: <-- this will become "model : object = None" so user can pass their own model
               u2ss: float | None = None, x0_setup: Optional[Callable] = None,
               # previous_data_name: str | None = None, # TODO: decide whether we still need it
               parallelize: bool = False, n_processes: int | None = None,
               ) -> Dict:
        """
        Runs ReplayBG replay procedure.
        """
        # TODO: validate the inputs

        # Unpack data to optimize performance during simulation
        rbg_data = MultiMealT1DData(data=data, environment=self.environment)  # TODO: use also the inputs model=model, environment=self.environment to set up the data in a model-agnostic fashion

        # convert theta to numba typed dict
        theta_typed = to_typed_f32_dict(theta)

        # Initialize model TODO: change this to the model provided in input
        model = MultiMealT1DModel(u2ss=rbg_data.u2ss, theta0=theta_typed)  # TODO: can we set u2ss AFTER data?

        if x0_setup is not None:
            x0_setup(model, rbg_data)

        out = np.zeros(rbg_data.tsteps, )
        out[0] = model.output(0)
        for k in np.arange(1,out.shape[0]):
            model.step(rbg_data.u[k], k)
            out[k] = model.output(k)

        out = out[0::rbg_data.yts]

        return out
