"""Simulator package for multi-segment ReplayBG replay.

Public API
----------
SimulatorOrchestrator
    Takes the output of TwinnerOrchestrator.twin() and replays each segment
    with proper physiological carry-over state between segments.
"""

from simulator.simulator_orchestrator import SimulatorOrchestrator

__all__ = ["SimulatorOrchestrator"]