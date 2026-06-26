from typing import Dict

import numpy as np

from callbacks.context import ReplayContext
from environment import Environment
from twinner.twinner import Twinner
from utils.save_results import save_results


class ReplayBG:
    """Core class of ReplayBG.
    """

    def __init__(self,
                 ts: int = 1,
                 seed: int = 1,
                 plot_mode: bool = True, verbose: bool = True
                 ):
        """Constructs all the necessary attributes for the ReplayBG object.
        """

        # TODO: Validate input

        # Initialize the environment parameters
        self.environment = Environment(ts=ts,
                                       seed=seed,
                                       plot_mode=plot_mode,
                                       verbose=verbose)

    def twin(self, rbg_data,
             model: object = None,
             unknown_parameters_prior: Dict = None,
             n_starts: int = 64,
             parallelize: bool = False, n_jobs: int | None = None,
             log_history: bool = False,
             path: str | None = None, save_name: str | None = None,
             ) -> Dict:
        """Runs ReplayBG twinning procedure.

        If ``path`` is provided, the twinning results (``theta``, ``history`` and
        ``rbg_data``) are pickled to ``<path>/<name>.pkl``. When ``name`` is
        omitted it defaults to ``twin_YYYY_mm_dd.pkl``.
        """
        # TODO: validate the inputs

        # Initialize the twinner
        twinner = Twinner(parallelize=parallelize, n_jobs=n_jobs, n_starts=n_starts, log_history=log_history)

        # Run the twinning procedure
        theta_estimated = twinner.twin(model=model,
                                       rbg_data=rbg_data,
                                       unknown_parameters_prior=unknown_parameters_prior)

        ret = {
            'theta': dict(zip(unknown_parameters_prior.keys(), theta_estimated['x'])),
            'history': twinner.history if log_history else None,
        }

        # Save results if a destination path was provided
        if path is not None:
            save_results({**ret, 'rbg_data': rbg_data}, path, save_name, prefix='twin')

        return ret

    def replay(self, rbg_data,
               model: object = None,
               theta: Dict = None,
               callbacks: list | None = None,
               sensor: object = None,
               parallelize: bool = False, n_processes: int | None = None,
               path: str | None = None, save_name: str | None = None,
               ) -> Dict:
        """Runs ReplayBG replay procedure.

        Replays the recorded inputs through the model, optionally letting
        user-supplied control policies act at every integration minute. Each
        ``callback`` (a :class:`~control.callback.ReplayCallback`) is invoked
        before the model steps and may modify the inputs for the current step via
        the :class:`~control.context.ReplayContext` it receives.

        An optional ``sensor`` (a :class:`~sensors.sensor.Sensor`) observes the model
        output and produces a measurement of it at its own cadence (``sensor.ts``). The
        measurement is exposed to the callbacks via ``ctx.measurement`` (held between
        samples), so closed-loop policies act on the realistic, noisy signal rather than
        the true output. The sensor is output-agnostic: it sees only ``model.output(k)``.
        When no sensor is supplied, ``ctx.measurement`` carries the true output, so
        behaviour is unchanged.

        If ``path`` is provided, the replay results (the returned dict plus
        ``rbg_data``) are pickled to ``<path>/<name>.pkl``. When ``name`` is
        omitted it defaults to ``replay_YYYY_mm_dd.pkl``.

        Returns:
            dict with keys:
                ``output``: predicted interstitial glucose at integration resolution.
                ``input``: applied inputs at integration resolution, shape (tsteps, n).
                ``data_to_input``: channel index -> name mapping.
                ``actions``: flat list of action records logged by callbacks.
                ``measurement``: (only with a sensor) sensor samples at sensor cadence.
                ``measurement_time``: (only with a sensor) integration-step index of
                    each sample.
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

        use_sensor = sensor is not None
        if use_sensor:
            # Seed the global RNG and (re)connect the sensor under that seed so its
            # whole error realization (sampled parameters + per-step noise) is
            # reproducible, regardless of when the sensor object was constructed.
            np.random.seed(self.environment.seed)
            sensor.connect_new(connected_at=0)
            ts = sensor.ts
            n_samp = (rbg_data.tsteps - 1) // ts + 1
            measurement = np.full(n_samp, np.nan)
            measurement_time = np.zeros(n_samp, dtype=int)
            measurement[0] = sensor.measure(out[0], past_values=out[:0], t=0.0)
            measurement_time[0] = 0
            # Zero-order-held measurement at integration resolution (for ctx history).
            meas_hold = out.copy()
            meas_hold[0] = measurement[0]
            ctx.measurement_history = meas_hold

        for k in range(1, rbg_data.tsteps):
            ctx._advance(k, rbg_data.u[k].copy())
            # Latest measurement available to a controller acting at step k.
            ctx.measurement = meas_hold[k - 1] if use_sensor else out[k - 1]
            for cb in callbacks:
                ctx._active_cb = type(cb).__name__
                cb.action(ctx)
            model.step(ctx.u, k)
            out[k] = model.output(k)
            replayed_u[k] = ctx.u

            if use_sensor:
                if k % ts == 0:
                    if (k + sensor.t_offset) % sensor.max_lifetime == 0:
                        sensor.connect_new(connected_at=k)
                    j = k // ts
                    measurement[j] = sensor.measure(
                        out[k], past_values=out[:k],
                        t=(k - sensor.connected_at) / (24 * 60))
                    measurement_time[j] = k
                    meas_hold[k] = measurement[j]
                else:
                    meas_hold[k] = meas_hold[k - 1]

        ret = {
            'output': out,
            'input': replayed_u,
            'data_to_input': rbg_data.data_to_input,
            'actions': ctx._actions,
        }
        if use_sensor:
            ret['measurement'] = measurement
            ret['measurement_time'] = measurement_time

        # Save results if a destination path was provided
        if path is not None:
            save_results({**ret}, path, save_name, prefix='replay')

        return ret
