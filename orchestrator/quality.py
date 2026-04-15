"""Default quality-filtering policy for :class:`TwinnerOrchestrator`.

A *quality function* has the signature::

    quality_fn(segment: pd.DataFrame) -> bool

It returns ``True`` when the segment meets the minimum data-quality bar for
twinning, or ``False`` to discard it (which also resets the carry-over state
so the next accepted segment starts cold).

This module provides :func:`default_quality_fn`, which enforces four criteria
that are necessary for a well-conditioned MAP optimisation:

1. At least 70 % of expected CGM samples are present (≤ 30 % NaN glucose).
2. At least one bolus event is recorded (non-zero ``bolus`` column).
3. At least one meal event is recorded (non-zero ``cho`` column).
4. The basal insulin trace is complete (no NaN in ``basal`` column).
"""

from __future__ import annotations

import pandas as pd


def default_quality_fn(segment: pd.DataFrame) -> bool:
    """Return ``True`` iff *segment* meets the minimum quality criteria.

    Criteria (all must hold):

    * ``len(segment) > 0`` — segment is non-empty.
    * ``glucose`` NaN fraction ≤ 0.30 — at most 30 % of CGM samples missing.
    * ``bolus.sum() > 0`` — at least one bolus event present.
    * ``cho.sum() > 0`` — at least one meal event present.
    * ``basal.isna().any() == False`` — basal trace is gap-free.

    Args:
        segment: A single segment DataFrame as returned by a segmentation
            function.  Must contain ``glucose``, ``bolus``, ``cho``, and
            ``basal`` columns.

    Returns:
        ``True`` if the segment passes all checks, ``False`` otherwise.

    Example::

        from orchestrator.quality import default_quality_fn

        # use as-is:
        orchestrator = TwinnerOrchestrator(..., quality_fn=default_quality_fn)

        # or tighten the CGM threshold via a wrapper:
        orchestrator = TwinnerOrchestrator(...,
            quality_fn=lambda seg: _strict_quality(seg))

        def _strict_quality(seg):
            if seg['glucose'].isna().mean() > 0.10:   # max 10 % NaN
                return False
            return default_quality_fn(seg)
    """
    if len(segment) == 0:
        return False

    # --- CGM coverage ---
    if "glucose" not in segment.columns:
        return False
    if segment["glucose"].isna().mean() > 0.30:
        return False

    # --- Bolus presence ---
    if "bolus" not in segment.columns or segment["bolus"].sum() == 0:
        return False

    # --- Meal presence ---
    if "cho" not in segment.columns or segment["cho"].sum() == 0:
        return False

    # --- Basal completeness ---
    if "basal" not in segment.columns or segment["basal"].isna().any():
        return False

    return True
