"""Multi-segment replay orchestrator.

:class:`SimulatorOrchestrator` takes the output of
:class:`~orchestrator.TwinnerOrchestrator` and replays each twinned segment
with proper physiological carry-over state between consecutive segments.

The carry-over mechanism is identical to the one used during twinning: after
replaying each segment, a forward simulation is run to extract the final model
state, which is then injected into the next segment via
:meth:`model_class.setup_x0`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.numba_dicts import to_typed_f32_dict


class SimulatorOrchestrator:
    """Orchestrate multi-segment replay over the results of twinning.

    Args:
        model_class: Model class used for the forward simulation (e.g.
            ``MultiMealT1DModel``).  Must expose:

            * ``__init__(u2ss, theta0, tsteps)``
            * ``step(u, t)``
            * ``setup_x0(x0, previous_theta)`` — static method
            * ``extract_final_x0(model_instance)`` — static method

        data_class: Data-preprocessing class paired with *model_class*
            (e.g. ``MultiMealT1DData``).  Must accept
            ``__init__(data, body_weight, environment)``.
        data: Full multi-day raw DataFrame — the same one originally passed to
            :class:`~orchestrator.TwinnerOrchestrator`.  Must contain a ``t``
            column of ``datetime64`` values.
        bw: Patient body weight in kg.
        rbg: Initialised :class:`~replaybg.ReplayBG` instance.
        twin_results: List of result dicts returned by
            :meth:`~orchestrator.TwinnerOrchestrator.twin`.  Each dict must
            contain ``segment_index``, ``segment_start``, ``segment_end``, and
            ``theta``.
        save_name_prefix: Prefix for the ``save_name`` argument passed to
            ``ReplayBG.replay()``; each segment gets
            ``"{prefix}_{segment_index:04d}"``.

    Example::

        from orchestrator import TwinnerOrchestrator
        from simulator import SimulatorOrchestrator
        from model.multi_meal_t1d import MultiMealT1DModel
        from data.multi_meal_t1d_data import MultiMealT1DData
        from replaybg import ReplayBG

        rbg = ReplayBG(save_folder="results/")

        orchestrator = TwinnerOrchestrator(
            model_class=MultiMealT1DModel,
            data_class=MultiMealT1DData,
            data=df, bw=75.0, rbg=rbg,
            unknown_parameters_prior=prior,
        )
        twin_results = orchestrator.twin()

        sim = SimulatorOrchestrator(
            model_class=MultiMealT1DModel,
            data_class=MultiMealT1DData,
            data=df, bw=75.0, rbg=rbg,
            twin_results=twin_results,
        )
        sim_results = sim.replay()
        for r in sim_results:
            print(r['segment_start'], r['glucose'].shape)
    """

    def __init__(
        self,
        model_class: type,
        data_class: type,
        data: pd.DataFrame,
        bw: float,
        rbg,
        twin_results: list[dict],
        save_name_prefix: str = "replay",
    ) -> None:
        self.model_class = model_class
        self.data_class = data_class
        self.data = data
        self.bw = bw
        self.rbg = rbg
        self.twin_results = twin_results
        self.save_name_prefix = save_name_prefix

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def replay(self) -> list[dict]:
        """Replay all twinned segments with physiological carry-over.

        Returns:
            List of result dicts, one per segment in the same order as
            *twin_results*.  Each dict contains:

            * ``segment_index`` (int): position index from the original
              segmentation.
            * ``segment_start`` (pd.Timestamp): first ``t`` value of the
              segment.
            * ``segment_end`` (pd.Timestamp): last ``t`` value of the
              segment.
            * ``theta`` (dict[str, float]): parameter values used for this
              segment (copied from *twin_results*).
            * ``glucose`` (np.ndarray): simulated interstitial glucose trace
              at data resolution (one sample per ``yts`` simulation steps).
        """
        results: list[dict] = []
        prev_x0 = None
        prev_theta: dict | None = None

        for r in self.twin_results:
            seg_df = self.data[
                (self.data["t"] >= r["segment_start"])
                & (self.data["t"] <= r["segment_end"])
            ].copy()

            # --- x0 carry-over from the previous segment ---
            x0_setup = None
            if prev_x0 is not None:
                x0_setup = self.model_class.setup_x0(
                    x0=prev_x0, previous_theta=prev_theta
                )

            # --- Replay this segment ---
            save_name = f"{self.save_name_prefix}_{r['segment_index']:04d}"
            glucose = self.rbg.replay(
                data=seg_df,
                theta=r["theta"],
                bw=self.bw,
                save_name=save_name,
                x0_setup=x0_setup,
            )

            # --- Extract final state for carry-over into the next segment ---
            rbg_data = self.data_class(
                data=seg_df,
                body_weight=self.bw,
                environment=self.rbg.environment,
            )
            final_model = self._run_forward(rbg_data, r["theta"], prev_x0=prev_x0, prev_theta=prev_theta)
            prev_x0 = self.model_class.extract_final_x0(final_model)
            prev_theta = r["theta"]

            results.append(
                {
                    "segment_index": r["segment_index"],
                    "segment_start": r["segment_start"],
                    "segment_end": r["segment_end"],
                    "theta": r["theta"],
                    "glucose": glucose,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_forward(self, rbg_data, theta: dict, prev_x0=None, prev_theta: dict | None = None):
        """Run a forward simulation with *theta* and return the final model.

        The model is initialised with *theta* and stepped through all
        ``rbg_data.tsteps`` time steps.  The returned instance has its state
        arrays populated at every time step, so ``model.G[-1]`` etc. hold the
        end-of-segment values needed by ``extract_final_x0``.

        When *prev_x0* is supplied the model is initialised with the same
        carry-over state that was applied during twinning (``_kd_prev``,
        ``_ka2_prev``, and ``model.x0``), so that the forward trajectory is
        consistent with the fitted trajectory and the extracted final state is
        correct for carry-over into the next segment.

        Note: by the time this method is called, *prev_x0* has already had its
        meal-gut compartments zeroed and ``rbg_data.u[:, 8]`` already contains
        the carry-over glucose appearance rate — both set by the ``x0_setup``
        callable that was applied during twinning.  We therefore only need to
        replicate the model-side part of that setup here.

        Args:
            rbg_data: Pre-processed data object for the current segment.
            theta: Plain Python dict of estimated parameters (as returned by
                ``ReplayBG.twin()``).
            prev_x0: Numba typed dict of carry-over initial conditions from the
                previous segment (same object passed to ``x0_setup``).  ``None``
                for the first accepted segment (cold start).
            prev_theta: Plain Python dict of estimated parameters from the
                previous segment, used to scale the insulin compartment initial
                conditions.  ``None`` for cold start.

        Returns:
            Model instance at the end of the simulation.
        """
        theta_typed = to_typed_f32_dict(theta)
        model = self.model_class(
            u2ss=rbg_data.u2ss,
            tsteps=rbg_data.tsteps,
        )
        if prev_x0 is not None:
            pt = prev_theta or {}
            model._kd_prev  = np.float64(pt.get('kd',  0.026))
            model._ka2_prev = np.float64(pt.get('ka2', 0.014))
            model.x0 = prev_x0
        model.reset(theta_typed)
        for t in range(1, rbg_data.tsteps):
            model.step(rbg_data.u[t], t)
        return model
