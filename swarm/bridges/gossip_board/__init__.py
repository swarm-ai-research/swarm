"""Gossip-board search swarm: information fidelity as a governance lever.

A ground-truth model of the Hyperspace "gossiping agent swarm" (Mathur,
2026-09-04): N agents run an autoresearch-style loop over a discrete config
space and publish improvements to a shared board that other agents read.
The single lever is *how much* of a published result travels:

    code         the full config is adopted verbatim (executable recipe)
    description  only the diff travels ("a peer got 3.21 by switching to
                 RMSNorm"): readers apply that one change to their own config
    score_only   readers learn that someone improved, and nothing else

Observables: identical-config cluster size (the "17 agents to four decimal
places" fingerprint), frontier progress, late-joiner cold start, the
survivorship gap of a success-only board, dimensions the whole population
never explores, and ground-truth lineage (copied vs rediscovered) that a
provenance-blind detector cannot see.

CLI: ``python -m swarm.bridges.gossip_board <scenario.yaml> [--axis k=v1,v2]``.
"""

from swarm.bridges.gossip_board.model import (
    BoardConfig,
    RoundMetrics,
    RunResult,
    run_board,
    sweep_board,
)

__all__ = ["BoardConfig", "RoundMetrics", "RunResult", "run_board", "sweep_board"]
