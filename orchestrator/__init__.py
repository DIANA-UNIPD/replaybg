"""Orchestrator package for multi-segment ReplayBG twinning.

Public API
----------
TwinnerOrchestrator
    Main orchestrator class — segments data, filters by quality, builds
    per-segment priors, and glues twins together via ``x0_setup``.

segment_by_first_event
    Default segmentation policy: each segment starts 30 min before the first
    meal/bolus event of the calendar day and ends at 04:00 the next morning.

default_quality_fn
    Default quality policy: rejects segments with > 30 % NaN glucose, no
    bolus, no meal events, or any NaN in basal.
"""

from orchestrator.quality import default_quality_fn
from orchestrator.segmentation import segment_by_first_event
from orchestrator.twinner_orchestrator import TwinnerOrchestrator

__all__ = [
    "TwinnerOrchestrator",
    "segment_by_first_event",
    "default_quality_fn",
]
