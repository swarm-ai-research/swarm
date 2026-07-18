"""A-Evolve domain: the evolution loop under SWARM's soft-label lens."""

from swarm.domains.aevolve.adapter import AevolveEvolutionAdapter, EvolutionReport
from swarm.domains.aevolve.config import AevolveDomainConfig
from swarm.domains.aevolve.entities import CandidateOutcome, GateDecision

__all__ = [
    "AevolveDomainConfig",
    "AevolveEvolutionAdapter",
    "CandidateOutcome",
    "EvolutionReport",
    "GateDecision",
]
