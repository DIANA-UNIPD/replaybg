"""Default segmentation policy for :class:`TwinnerOrchestrator`.

A *segmentation function* has the signature::

    segment_fn(data: pd.DataFrame) -> list[pd.DataFrame]

It receives the full, unmodified multi-day DataFrame and returns an **ordered**
list of segment DataFrames.  Each segment is passed as-is to the quality
filter and, if accepted, to ``ReplayBG.twin()``.

This module provides :func:`segment_by_first_event`, the physiologically
motivated default: each segment starts 30 minutes before the first meal or
bolus event of the calendar day and ends at 04:00 the following morning,
aligning with the model's SI_D / SI_B day boundary.
"""

from __future__ import annotations

import pandas as pd


def segment_by_first_event(
    data: pd.DataFrame,
    pre_event_minutes: int = 30,
) -> list[pd.DataFrame]:
    """Segment a multi-day DataFrame into physiological day chunks.

    Each segment is anchored to the **first meal or bolus event** of a
    calendar day:

    * **Start**: ``first_event_time − pre_event_minutes`` (data before this
      point is excluded from the segment).
    * **End**: 04:00 AM of the *next* calendar day (exclusive), matching the
      model's SI_D → SI_B time-of-day boundary.

    Calendar days with no meal (``cho > 0``) and no bolus (``bolus > 0``)
    events are skipped entirely — no segment is produced for them.

    Args:
        data: Full multi-day DataFrame.  Must contain a ``t`` column of
            ``datetime64`` values and at least ``cho`` and ``bolus`` columns.
        pre_event_minutes: Minutes before the first event to include as
            warm-up context.  Defaults to 30.

    Returns:
        Ordered list of segment DataFrames (one per accepted calendar day),
        sorted chronologically.  Each DataFrame is a copy with the original
        index reset.

    Example::

        from orchestrator.segmentation import segment_by_first_event

        segments = segment_by_first_event(df)
        # or with a custom pre-event window:
        segments = segment_by_first_event(df, pre_event_minutes=60)
        # inject as segment_fn:
        orchestrator = TwinnerOrchestrator(...,
            segment_fn=lambda d: segment_by_first_event(d, pre_event_minutes=60))
    """
    data = data.copy()
    data["_cal_date"] = data["t"].dt.normalize()

    segments: list[pd.DataFrame] = []

    for cal_date in sorted(data["_cal_date"].unique()):
        day_mask = data["_cal_date"] == cal_date
        day_data = data[day_mask]

        # Collect meal and bolus event timestamps on this calendar day.
        event_times: list[pd.Timestamp] = []
        if "cho" in day_data.columns:
            event_times.extend(day_data.loc[day_data["cho"] > 0, "t"].tolist())
        if "bolus" in day_data.columns:
            event_times.extend(day_data.loc[day_data["bolus"] > 0, "t"].tolist())

        if not event_times:
            # No metabolic event on this day — skip.
            continue

        first_event: pd.Timestamp = min(event_times)
        seg_start: pd.Timestamp = first_event - pd.Timedelta(minutes=pre_event_minutes)
        seg_end: pd.Timestamp = pd.Timestamp(cal_date) + pd.Timedelta(days=1, hours=4)

        seg_df = (
            data[(data["t"] >= seg_start) & (data["t"] < seg_end)]
            .drop(columns=["_cal_date"])
            .copy()
            .reset_index(drop=True)
        )

        if len(seg_df) > 0:
            segments.append(seg_df)

    return segments
