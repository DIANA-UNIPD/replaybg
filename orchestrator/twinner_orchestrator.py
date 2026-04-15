"""Multi-segment twinning orchestrator.

:class:`TwinnerOrchestrator` wraps :class:`~replaybg.ReplayBG` and automates
the full pipeline for twinning multiple consecutive data segments:

1. **Segmentation** — a user-supplied callable cuts the DataFrame into an
   ordered list of segments.
2. **Quality filtering** — a user-supplied callable discards segments that
   do not meet the minimum data-quality bar; discarding a segment resets the
   physiological carry-over state so the next accepted segment starts cold.
3. **Prior selection** — the orchestrator decides which parameters to
   estimate for each segment based on the data it contains:

   * ``kabs_X`` / ``beta_X``: identified only when the segment contains a
     meal event whose ``cho_label`` starts with ``X``.
   * ``SI_B``: identified only when non-NaN glucose is observed in
     04:00–10:59 (the Breakfast insulin-sensitivity window).
   * ``SI_L``: identified only when non-NaN glucose is observed in
     11:00–16:59 (the Lunch window).
   * ``SI_D``: identified only when non-NaN glucose is observed in
     17:00–03:59 (the Dinner / night window).
   * All other parameters are always estimated.

4. **Carry-over gluing** — after twinning each segment a forward simulation
   is run to extract the final model state, which is then injected into the
   next segment via :meth:`model_class.setup_x0`.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from orchestrator.quality import default_quality_fn
from orchestrator.segmentation import segment_by_first_event
from utils.numba_dicts import to_typed_f32_dict


class TwinnerOrchestrator:
    """Orchestrate multi-segment twinning over a long-duration DataFrame.

    Args:
        model_class: Model class used as the twin blueprint for every
            segment (e.g. ``MultiMealT1DModel``).  The class must expose:

            * ``__init__(u2ss, theta0, tsteps)``
            * ``step(u, t)``
            * ``setup_x0(x0, previous_theta)`` — static method
            * ``extract_final_x0(model_instance)`` — static method

        data_class: Data-preprocessing class paired with *model_class*
            (e.g. ``MultiMealT1DData``).  Must accept
            ``__init__(data, body_weight, environment)``.
        data: Full multi-day raw DataFrame.  Must contain at minimum a ``t``
            column of ``datetime64`` values.
        bw: Patient body weight in kg.
        rbg: Initialised :class:`~replaybg.ReplayBG` instance.  Its
            ``environment`` is reused for every segment.
        unknown_parameters_prior: Full candidate prior dictionary (same
            structure as passed to ``ReplayBG.twin()``).  The orchestrator
            filters this dict for each segment based on the data available.
        segment_fn: Callable ``(data) -> list[pd.DataFrame]`` that cuts
            *data* into an ordered list of segment DataFrames.  Defaults to
            :func:`~orchestrator.segmentation.segment_by_first_event`.
        quality_fn: Callable ``(segment) -> bool`` that returns ``True`` when
            a segment is acceptable for twinning.  Defaults to
            :func:`~orchestrator.quality.default_quality_fn`.
        save_name_prefix: Prefix for the ``save_name`` argument passed to
            ``ReplayBG.twin()``; each segment gets ``"{prefix}_{index:04d}"``.
        n_starts: Number of random starts for the MAP optimiser.
        parallelize: Whether to parallelise the optimiser across starts.
        n_jobs: Worker count when *parallelize* is ``True``; ``None`` uses all
            CPU cores.

    Example::

        from orchestrator import TwinnerOrchestrator
        from model.multi_meal_t1d import MultiMealT1DModel
        from data.multi_meal_t1d_data import MultiMealT1DData
        from replaybg import ReplayBG

        rbg = ReplayBG(save_folder="results/")
        orchestrator = TwinnerOrchestrator(
            model_class=MultiMealT1DModel,
            data_class=MultiMealT1DData,
            data=df,
            bw=75.0,
            rbg=rbg,
            unknown_parameters_prior=prior,
        )
        results = orchestrator.twin()
        for r in results:
            print(r['segment_start'], r['theta'])
    """

    def __init__(
        self,
        model_class: type,
        data_class: type,
        data: pd.DataFrame,
        bw: float,
        rbg,
        unknown_parameters_prior: dict,
        segment_fn: Callable[[pd.DataFrame], list[pd.DataFrame]] = segment_by_first_event,
        quality_fn: Callable[[pd.DataFrame], bool] = default_quality_fn,
        save_name_prefix: str = "twin",
        n_starts: int = 64,
        parallelize: bool = False,
        n_jobs: int | None = None,
    ) -> None:
        self.model_class = model_class
        self.data_class = data_class
        self.data = data
        self.bw = bw
        self.rbg = rbg
        self.unknown_parameters_prior = unknown_parameters_prior
        self.segment_fn = segment_fn
        self.quality_fn = quality_fn
        self.save_name_prefix = save_name_prefix
        self.n_starts = n_starts
        self.parallelize = parallelize
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def twin(self) -> list[dict]:
        """Run twinning for all accepted segments.

        Returns:
            List of result dicts, one per accepted segment, in chronological
            order.  Each dict contains:

            * ``segment_index`` (int): position in the ordered segment list
              returned by *segment_fn*.
            * ``segment_start`` (pd.Timestamp): first ``t`` value of the
              segment.
            * ``segment_end`` (pd.Timestamp): last ``t`` value of the
              segment.
            * ``theta`` (dict[str, float]): estimated parameter values.
            * ``prior_used`` (dict): subset of *unknown_parameters_prior*
              that was actually estimated for this segment.
        """
        segments = self.segment_fn(self.data)

        results: list[dict] = []
        prev_x0 = None
        prev_theta: dict | None = None

        for i, seg_df in enumerate(segments):
            # --- Quality check ---
            if not self.quality_fn(seg_df):
                # Reset carry-over so the next accepted segment starts cold.
                prev_x0 = None
                prev_theta = None
                continue

            # --- Prior selection for this segment ---
            prior = self._build_prior(seg_df)

            # --- x0 carry-over from the previous accepted segment ---
            x0_setup = None
            if prev_x0 is not None:
                x0_setup = self.model_class.setup_x0(
                    x0=prev_x0, previous_theta=prev_theta
                )

            # --- Prepare data object and model instance ---
            rbg_data = self.data_class(
                data=seg_df,
                body_weight=self.bw,
                environment=self.rbg.environment,
            )
            model = self.model_class(u2ss=rbg_data.u2ss, tsteps=rbg_data.tsteps)

            # --- Twin this segment ---
            save_name = f"{self.save_name_prefix}_{i:04d}"
            theta = self.rbg.twin(
                data=rbg_data,
                bw=self.bw,
                model=model,
                save_name=save_name,
                unknown_parameters_prior=prior,
                n_starts=self.n_starts,
                x0_setup=x0_setup,
                parallelize=self.parallelize,
                n_jobs=self.n_jobs,
            )

            # --- Extract final state for carry-over into next segment ---
            final_model = self._run_forward(rbg_data, theta)
            prev_x0 = self.model_class.extract_final_x0(final_model)
            prev_theta = theta

            results.append(
                {
                    "segment_index": i,
                    "segment_start": seg_df["t"].iloc[0],
                    "segment_end": seg_df["t"].iloc[-1],
                    "theta": theta,
                    "prior_used": prior,
                }
            )

        return results

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prior(self, seg_df: pd.DataFrame) -> dict:
        """Return the subset of *unknown_parameters_prior* to estimate.

        Parameters whose identifiability depends on data availability:

        * ``kabs_X`` / ``beta_X`` — included only when the segment has at
          least one meal row with ``cho_label`` starting with ``X`` and
          ``cho > 0``.
        * ``SI_B`` — included only when non-NaN glucose is observed during
          04:00–10:59.
        * ``SI_L`` — included only when non-NaN glucose is observed during
          11:00–16:59.
        * ``SI_D`` — included only when non-NaN glucose is observed during
          17:00–03:59 (wraps around midnight).

        All other parameters are always included.
        """
        prior = {}
        for param, spec in self.unknown_parameters_prior.items():
            if self._should_identify(param, seg_df):
                prior[param] = spec
        return prior

    def _should_identify(self, param: str, seg_df: pd.DataFrame) -> bool:
        """Return ``True`` if *param* should be estimated for *seg_df*."""
        if param.startswith("kabs_") or param.startswith("beta_"):
            label = param.split("_", 1)[1]
            return self._has_meal(seg_df, label)
        if param.startswith("SI_"):
            period = param.split("_", 1)[1]
            return self._has_cgm_in_window(seg_df, period)
        return True

    def _has_meal(self, seg_df: pd.DataFrame, label: str) -> bool:
        """Return ``True`` if *seg_df* contains a meal labelled ``label*``."""
        if "cho_label" not in seg_df.columns or "cho" not in seg_df.columns:
            return False
        meal_rows = seg_df[(seg_df["cho"] > 0) & seg_df["cho_label"].notna()]
        return meal_rows["cho_label"].str.startswith(label).any()

    def _has_cgm_in_window(self, seg_df: pd.DataFrame, period: str) -> bool:
        """Return ``True`` if non-NaN glucose is present in *period*'s window.

        Windows (matching the model's SI time-of-day selection):

        * ``'B'``: 04:00–10:59
        * ``'L'``: 11:00–16:59
        * ``'D'``: 17:00–03:59 (wraps midnight)
        """
        if "glucose" not in seg_df.columns or "t" not in seg_df.columns:
            return False
        hours = seg_df["t"].dt.hour
        if period == "B":
            mask = (hours >= 4) & (hours < 11)
        elif period == "L":
            mask = (hours >= 11) & (hours < 17)
        elif period == "D":
            mask = (hours >= 17) | (hours < 4)
        else:
            return True  # Unknown period — include by default.
        return seg_df.loc[mask, "glucose"].notna().any()

    def _run_forward(self, rbg_data, theta: dict):
        """Run a forward simulation with *theta* and return the final model.

        The model is initialised with *theta* and stepped through all
        ``rbg_data.tsteps`` time steps.  The returned instance has its state
        arrays populated at every time step, so ``model.G[-1]`` etc. hold the
        end-of-segment values needed by ``extract_final_x0``.

        Args:
            rbg_data: Pre-processed data object for the current segment.
            theta: Plain Python dict of estimated parameters (as returned by
                ``ReplayBG.twin()``).

        Returns:
            Model instance at the end of the simulation.
        """
        theta_typed = to_typed_f32_dict(theta)
        model = self.model_class(
            u2ss=rbg_data.u2ss,
            theta0=theta_typed,
            tsteps=rbg_data.tsteps,
        )
        for t in range(1, rbg_data.tsteps):
            model.step(rbg_data.u[t], t)
        return model
