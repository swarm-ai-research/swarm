"""Pydantic schema for scenario YAML files."""

from __future__ import annotations

from typing import Any, cast

from pydantic import BaseModel, ConfigDict


class _SectionModel(BaseModel):
    """Base class for typed scenario sections."""

    model_config = ConfigDict(extra="forbid")


class AgentSpec(_SectionModel):
    """Agent entry in the top-level ``agents`` list."""

    type: str | None = None
    count: int = 1
    name: str | None = None
    id: str | None = None
    role: str | None = None
    policy: str | None = None
    strategy: str | None = None
    config: dict[str, Any] | None = None
    params: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    letta: dict[str, Any] | None = None
    concordia: dict[str, Any] | None = None
    council: dict[str, Any] | None = None
    capabilities: list[str] | None = None
    trust_group: str | None = None
    hands_off_to: list[str] | None = None
    scooter_priority: int | None = None
    skill_gather: float | None = None
    skill_build: float | None = None
    shift_fraction: float | None = None
    underreport_fraction: float | None = None
    coalition_id: str | None = None


class SimulationConfig(_SectionModel):
    """Simulation timing and scheduling configuration."""

    n_epochs: int | None = None
    epochs: int | None = None
    steps_per_epoch: int | None = None
    seed: int | None = None
    schedule_mode: str | None = None
    max_actions_per_step: int | None = None
    observation_noise_probability: float | None = None
    observation_noise_std: float | None = None


class RateLimitsConfig(_SectionModel):
    """Per-epoch/per-step rate limits."""

    posts_per_epoch: int | None = None
    interactions_per_step: int | None = None
    votes_per_epoch: int | None = None
    tasks_per_epoch: int | None = None
    bounties_per_epoch: int | None = None
    bids_per_epoch: int | None = None


class PayoffConfig(_SectionModel):
    """Soft payoff configuration fields accepted by the loader."""

    s_plus: float | None = None
    s_minus: float | None = None
    h: float | None = None
    theta: float | None = None
    rho_a: float | None = None
    rho_b: float | None = None
    w_rep: float | None = None


class OutputsConfig(_SectionModel):
    """Scenario output path configuration."""

    event_log: str | None = None
    metrics_csv: str | None = None
    graph_memory_path: str | None = None
    results_dir: str | None = None
    plots_dir: str | None = None
    csv: str | None = None
    provenance: str | None = None


class NetworkConfig(_SectionModel):
    """Network section shape used by the scenario loader."""

    enabled: bool | None = None
    topology: str | None = None
    params: dict[str, Any] | None = None
    dynamic: bool | None = None
    edge_probability: float | None = None
    k_neighbors: int | None = None
    rewire_probability: float | None = None
    m_edges: int | None = None
    edge_strengthen_rate: float | None = None
    edge_decay_rate: float | None = None
    min_edge_weight: float | None = None
    max_edge_weight: float | None = None
    reputation_disconnect_threshold: float | None = None


class ScenarioConfig(BaseModel):
    """Validated structure of a scenario YAML file.

    This validates shape and field names, not simulation semantics. Bridge and
    domain-specific sections remain dictionaries so their own loaders keep
    detailed validation ownership.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    description: str
    motif: str | None = None

    agents: list[AgentSpec] | None = None
    simulation: SimulationConfig | None = None
    governance: dict[str, Any] | None = None
    payoff: PayoffConfig | None = None
    rate_limits: RateLimitsConfig | None = None
    outputs: OutputsConfig | None = None
    success_criteria: dict[str, Any] | None = None
    network: NetworkConfig | None = None

    awm: dict[str, Any] | None = None
    boundaries: dict[str, Any] | None = None
    bridge_config: dict[str, Any] | None = None
    community: dict[str, Any] | None = None
    composite_tasks: dict[str, Any] | None = None
    contracts: dict[str, Any] | None = None
    csm: dict[str, Any] | None = None
    delivery: dict[str, Any] | None = None
    dilemma: dict[str, Any] | None = None
    docker: dict[str, Any] | None = None
    domain: dict[str, Any] | None = None
    drift_detection: dict[str, Any] | None = None
    dynamic_toxicity: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    evo_game: dict[str, Any] | None = None
    evoskill: dict[str, Any] | None = None
    flash_crash: dict[str, Any] | None = None
    horizon_eval: dict[str, Any] | None = None
    letta: dict[str, Any] | None = None
    llm: dict[str, Any] | None = None
    marketplace: dict[str, Any] | None = None
    memory_tiers: dict[str, Any] | None = None
    metrics: dict[str, Any] | list[str] | None = None
    metrics_to_track: list[str] | None = None
    moltbook: dict[str, Any] | None = None
    moltipedia: dict[str, Any] | None = None
    multiverse: dict[str, Any] | None = None
    persona_expansion: dict[str, Any] | None = None
    perturbations: dict[str, Any] | None = None
    prime_intellect: dict[str, Any] | None = None
    resource_negotiation: dict[str, Any] | None = None
    rivals: dict[str, Any] | None = None
    scholar: dict[str, Any] | None = None
    seed_personas: list[dict[str, Any]] | None = None
    spawn: dict[str, Any] | None = None
    sweep: dict[str, Any] | None = None
    task_prompt: str | None = None
    thread_generation: dict[str, Any] | None = None
    tierra: dict[str, Any] | None = None
    whatif_injections: list[dict[str, Any]] | None = None


def validate_scenario_config(data: Any) -> ScenarioConfig:
    """Validate a parsed YAML payload and return its schema model."""
    return cast(ScenarioConfig, ScenarioConfig.model_validate(data))
