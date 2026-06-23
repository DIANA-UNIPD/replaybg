from typing import Dict

import numpy as np

from environment import Environment
from twinner.twinner import Twinner


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
             parallelize: bool = False, n_jobs: int | None = None,
             log_history: bool = False,
             ) -> Dict:
        """Runs ReplayBG twinning procedure.
        """
        # TODO: validate the inputs

        # Initialize the twinner
        twinner = Twinner(parallelize=parallelize, n_jobs=n_jobs, n_starts=n_starts, log_history=log_history)

        # Run the twinning procedure
        theta_estimated = twinner.twin(model=model,
                                       rbg_data=rbg_data,
                                       unknown_parameters_prior=unknown_parameters_prior)

        # TODO: save results before return

        return {
            'theta': dict(zip(unknown_parameters_prior.keys(), theta_estimated['x'])),
            'history': twinner.history if log_history else None,
        }

    def replay(self, rbg_data, save_name: str,
               model: object = None,
               theta: Dict = None,
               parallelize: bool = False, n_processes: int | None = None,
               ) -> Dict:
        """Runs ReplayBG replay procedure.
        """
        # TODO: validate the inputs

        out = np.zeros(rbg_data.tsteps, )
        out[0] = model.output(0)
        for k in np.arange(1, out.shape[0]):
            model.step(rbg_data.u[k], k)
            out[k] = model.output(k)

        out = out[0::rbg_data.yts]

        # TODO: save results before return

        return out
