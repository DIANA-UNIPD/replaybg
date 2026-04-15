"""Orchestrator package for multi-segment ReplayBG twinning.

Public API
----------
TwinnerOrchestrator
    Main orchestrator class — segments data, filters by quality, builds
    per-segment priors, and glues twins together via ``x0_setup``.

segment_4_to_4
    Default segmentation policy: the first segment starts ``pre_event_minutes``
    before the first meal/bolus event; all subsequent segments start at 04:00
    AM of their calendar day; all segments end at 04:00 the next morning.

default_quality_fn
    Default quality policy: rejects segments with > 30 % NaN glucose, no
    bolus, no meal events, or any NaN in basal.
"""

from orchestrator.quality import default_quality_fn
from orchestrator.segmentation import segment_4_to_4
from orchestrator.twinner_orchestrator import TwinnerOrchestrator

__all__ = [
    "TwinnerOrchestrator",
    "segment_4_to_4",
    "default_quality_fn",
]
