from typing import Dict

import numpy as np

from callbacks.context import ReplayContext
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
               callbacks: list | None = None,
               parallelize: bool = False, n_processes: int | None = None,
               ) -> Dict:
        """Runs ReplayBG replay procedure.

        Replays the recorded inputs through the model, optionally letting
        user-supplied control policies act at every integration minute. Each
        ``callback`` (a :class:`~control.callback.ReplayCallback`) is invoked
        before the model steps and may modify the inputs for the current step via
        the :class:`~control.context.ReplayContext` it receives.

        Returns:
            dict with keys:
                ``output``: predicted interstitial glucose at integration resolution.
                ``input``: applied inputs at integration resolution, shape (tsteps, n).
                ``data_to_input``: channel index -> name mapping.
                ``actions``: flat list of action records logged by callbacks.
        """
        # TODO: validate the inputs

        n_ch = rbg_data.u.shape[1]
        out = np.zeros(rbg_data.tsteps, )
        replayed_u = np.zeros((rbg_data.tsteps, n_ch))
        out[0] = model.output(0)
        replayed_u[0] = rbg_data.u[0]

        callbacks = callbacks or []
        for cb in callbacks:
            cb.rbg_data = rbg_data
        ctx = ReplayContext(rbg_data=rbg_data, model=model,
                            output_history=out, input_history=replayed_u)

        for k in range(1, rbg_data.tsteps):
            ctx._advance(k, rbg_data.u[k].copy())
            for cb in callbacks:
                ctx._active_cb = type(cb).__name__
                cb.action(ctx)
            model.step(ctx.u, k)
            out[k] = model.output(k)
            replayed_u[k] = ctx.u

        return {
            'output': out,
            'input': replayed_u,
            'data_to_input': rbg_data.data_to_input,
            'actions': ctx._actions,
        }
