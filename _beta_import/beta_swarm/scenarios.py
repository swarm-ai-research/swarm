"""Declarative scenarios: a whole governed experiment as one YAML file.

The parent SWARM framework's reproducibility rests on scenario files — a run is
fully specified by a config (population, payoff, governance, seed), so it can be
version-controlled, diffed, and replayed exactly. This ports that discipline to
beta-swarm: a YAML file wires up any combination of the six subsystems
(population, proxy, payoff, governor, lever stack, adaptive controller) into a
runnable :class:`~beta_swarm.simulation.Simulation`, and identical files with
identical seeds produce identical results.

The schema mirrors the module layout:

    scenario_id: governor_showdown
    description: "..."
    seed: 7
    simulation: {n_epochs: 20, steps_per_epoch: 5, audit_scales_with_concentration: false}
    population:
      - {archetype: honest, count: 4}
      - {archetype: borderline, count: 8, quality_min: 0.35, quality_max: 0.68}
      - {archetype: volume_forger, count: 3, forge_volume: 10}
    proxy: {evidence_scale: 1.5, max_concentration: null}
    payoff: {s_plus: 2.0, h: 2.0}
    governor: {tau: 0.4, max_tail_mass: 0.2, defer_below_concentration: 20.0}
    governance_stack:
      initial_resources: 5.0
      levers:
        - {type: staking, min_stake: 1.0, slash_rate: 0.5}
        - {type: circuit_breaker, max_tail_exposure: 0.5}
    adaptive: {target_toxicity: 0.37, band: 0.01}

Only ``scenario_id`` and ``population`` are required; every other section falls
back to the same defaults the constructors use. ``load_scenario`` validates the
file (unknown keys are rejected, so typos fail loudly) and ``run_scenario``
loads, builds, and runs it in one call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from beta_swarm.adaptive import AdaptiveController
from beta_swarm.agents import Archetype, SimAgent, make_population
from beta_swarm.governance import TailMassGovernor
from beta_swarm.levers import (
    GovernanceLever,
    GovernanceStack,
    RiskTaxLever,
    StakingLever,
    TailCircuitBreaker,
)
from beta_swarm.payoff import DistributionalPayoffEngine, PayoffConfig
from beta_swarm.proxy import BetaProxyComputer
from beta_swarm.redteam import ReputationFarmer, VolumeForger
from beta_swarm.simulation import Simulation, SimulationConfig

# Archetypes that map straight to a stock SimAgent.
_STOCK = {a.value: a for a in Archetype}
# Poor true-quality used by the red-team adversary specs (matches redteam._POOR).
_POOR = (2.0, 8.0)


class _Strict(BaseModel):
    """Base model that rejects unknown keys, so typos in a scenario fail loudly."""

    model_config = ConfigDict(extra="forbid")


class AgentSpec(_Strict):
    """One population entry: an archetype and how many of it.

    Extra fields apply to specific archetypes: ``borderline`` spreads
    ``count`` agents evenly across quality means in ``[quality_min, quality_max]``
    at fixed ``concentration``; ``volume_forger`` takes ``forge_volume``;
    ``reputation_farmer`` takes ``flip_epoch``.
    """

    archetype: str
    count: int = Field(gt=0)
    # borderline
    quality_min: float = 0.35
    quality_max: float = 0.68
    concentration: float = 12.0
    # volume_forger
    forge_volume: int = 10
    # reputation_farmer
    flip_epoch: int = 10

    def build(self) -> list[SimAgent]:
        if self.archetype in _STOCK:
            arch = _STOCK[self.archetype]
            ring = "ring-0" if arch is Archetype.COLLUDER else None
            return [
                SimAgent.make(f"{self.archetype}-{i}", arch, ring=ring)
                for i in range(self.count)
            ]
        if self.archetype == "borderline":
            agents = []
            means = (
                np.linspace(self.quality_min, self.quality_max, self.count)
                if self.count > 1
                else [0.5 * (self.quality_min + self.quality_max)]
            )
            for i, mean in enumerate(means):
                c = self.concentration
                agents.append(
                    SimAgent(f"borderline-{i}", Archetype.MEDIOCRE, mean * c, (1.0 - mean) * c)
                )
            return agents
        if self.archetype == "volume_forger":
            return [
                VolumeForger(f"forger-{i}", Archetype.DECEPTIVE, *_POOR, forge_volume=self.forge_volume)
                for i in range(self.count)
            ]
        if self.archetype == "reputation_farmer":
            return [
                ReputationFarmer(f"farmer-{i}", Archetype.DECEPTIVE, *_POOR, flip_epoch=self.flip_epoch)
                for i in range(self.count)
            ]
        raise ValueError(
            f"unknown archetype {self.archetype!r}; expected one of "
            f"{sorted(_STOCK)} or borderline / volume_forger / reputation_farmer"
        )


class LeverSpec(_Strict):
    """One governance lever. ``type`` selects the lever; the rest are its params."""

    type: Literal["staking", "circuit_breaker", "risk_tax"]
    # staking
    min_stake: float = 1.0
    slash_rate: float = 0.25
    # circuit_breaker
    max_tail_exposure: float = 0.5
    window: int = 6
    freeze_epochs: int = 3
    use_realized: bool = True
    # risk_tax
    rate: float = 1.0
    # shared
    tau: float = 0.4

    def build(self) -> GovernanceLever:
        if self.type == "staking":
            return StakingLever(min_stake=self.min_stake, slash_rate=self.slash_rate, tau=self.tau)
        if self.type == "circuit_breaker":
            return TailCircuitBreaker(
                max_tail_exposure=self.max_tail_exposure,
                window=self.window,
                freeze_epochs=self.freeze_epochs,
                tau=self.tau,
                use_realized=self.use_realized,
            )
        return RiskTaxLever(rate=self.rate, tau=self.tau)


class StackSpec(_Strict):
    levers: list[LeverSpec]
    initial_resources: float = 10.0

    def build(self) -> GovernanceStack:
        return GovernanceStack([s.build() for s in self.levers], initial_resources=self.initial_resources)


class GovernorSpec(_Strict):
    tau: float = 0.25
    max_tail_mass: float = 0.2
    min_mean: float | None = None
    defer_below_concentration: float | None = None

    def build(self) -> TailMassGovernor:
        return TailMassGovernor(
            tau=self.tau,
            max_tail_mass=self.max_tail_mass,
            min_mean=self.min_mean,
            defer_below_concentration=self.defer_below_concentration,
        )


class ProxySpec(_Strict):
    w_progress: float = 0.4
    w_rework: float = 0.2
    w_verifier: float = 0.2
    w_engagement: float = 0.2
    sigmoid_k: float = 2.0
    base_concentration: float = 2.0
    evidence_scale: float = 1.5
    max_concentration: float | None = None

    def build(self) -> BetaProxyComputer:
        return BetaProxyComputer(**self.model_dump())


class SimSpec(_Strict):
    n_epochs: int = 20
    steps_per_epoch: int = 5
    audit_strength: float = 16.0
    audit_noise: float = 0.10
    audit_cost: float = 0.05
    reputation_decay: float = 0.9
    ring_affinity: float = 0.8
    random_audit_prob: float = 0.0
    audit_scales_with_concentration: bool = False
    audit_concentration_multiple: float = 2.0

    def build(self, seed: int) -> SimulationConfig:
        return SimulationConfig(seed=seed, **self.model_dump())


class AdaptiveSpec(_Strict):
    target_toxicity: float = 0.34
    band: float = 0.02
    parameter: str = "max_tail_mass"
    step: float = 0.05
    contemplation_interval: int = 4
    evaluation_window: int = 4
    warmup_epochs: int = 5
    min_evidence_epochs: int = 4
    welfare_weight: float = 0.0

    def build(self) -> AdaptiveController:
        return AdaptiveController(**self.model_dump())


class ScenarioConfig(_Strict):
    """A complete, reproducible experiment specification."""

    scenario_id: str
    description: str = ""
    seed: int = 0
    population: list[AgentSpec] = Field(min_length=1)
    simulation: SimSpec = Field(default_factory=SimSpec)
    proxy: ProxySpec = Field(default_factory=ProxySpec)
    payoff: PayoffConfig = Field(default_factory=PayoffConfig)
    governor: GovernorSpec = Field(default_factory=GovernorSpec)
    governance_stack: StackSpec | None = None
    adaptive: AdaptiveSpec | None = None

    def build_population(self) -> list[SimAgent]:
        agents: list[SimAgent] = []
        for spec in self.population:
            agents.extend(spec.build())
        return agents

    def build_simulation(self) -> Simulation:
        return Simulation(
            population=self.build_population(),
            governor=self.governor.build(),
            config=self.simulation.build(self.seed),
            engine=DistributionalPayoffEngine(self.payoff),
            proxy=self.proxy.build(),
            levers=self.governance_stack.build() if self.governance_stack else None,
            controller=self.adaptive.build() if self.adaptive else None,
        )

    def run(self):
        """Build and run; returns a :class:`~beta_swarm.simulation.SimulationResult`."""
        return self.build_simulation().run()


def load_scenario(path: str | Path) -> ScenarioConfig:
    """Parse and validate a YAML scenario file (unknown keys are rejected)."""
    data: dict[str, Any] = yaml.safe_load(Path(path).read_text()) or {}
    return ScenarioConfig.model_validate(data)


def run_scenario(path: str | Path):
    """Load, build, and run a scenario file in one call."""
    return load_scenario(path).run()
