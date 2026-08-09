"""Scenario loader for YAML configuration files."""

import logging
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type

import yaml

from swarm.agents.adaptive_adversary import AdaptiveAdversary
from swarm.agents.adversarial import AdversarialAgent
from swarm.agents.awm_agent import AWMAgent
from swarm.agents.base import BaseAgent
from swarm.agents.behavioral import CautiousAgent
from swarm.agents.cautious_reciprocator import CautiousReciprocator
from swarm.agents.coding_agent import CodingAgent
from swarm.agents.deceptive import DeceptiveAgent
from swarm.agents.honest import HonestAgent
from swarm.agents.hyperagent_self_mod import HyperagentSelfModAgent
from swarm.agents.ldt_agent import LDTAgent
from swarm.agents.memory_agent import (
    CacheGamerAgent,
    CollusiveVerifierAgent,
    DiligentRecorderAgent,
    MemoryPoisonerAgent,
)
from swarm.agents.modeling_adversary import ModelingAdversary
from swarm.agents.moltbook_agent import (
    CollusiveVoterAgent,
    DiligentMoltbookAgent,
    HumanPretenderAgent,
    SpamBotAgent,
)
from swarm.agents.negotiation_agent import (
    FairNegotiator,
    GreedyNegotiator,
    StrategicNegotiator,
)
from swarm.agents.obfuscating import ObfuscatingAgent
from swarm.agents.opportunistic import OpportunisticAgent
from swarm.agents.pressure_responsive import PressureResponsiveAgent
from swarm.agents.rain_river import RainAgent, RiverAgent
from swarm.agents.ralph_agent import AdversarialRalphAgent, RalphLoopAgent
from swarm.agents.rivals_agent import RivalsCriticAgent, RivalsProducerAgent
from swarm.agents.rlm_agent import RLMAgent
from swarm.agents.scholar_agent import (
    AdversarialRetrieverAgent,
    AdversarialSynthesizerAgent,
    RetrieverAgent,
    SynthesizerAgent,
    VerifierAgent,
)
from swarm.agents.self_optimizer import SelfOptimizerAgent
from swarm.agents.skill_evolving import (
    SkillEvolvingHonestAgent,
    SkillEvolvingOpportunisticAgent,
)
from swarm.agents.skillrl_agent import SkillRLAgent
from swarm.agents.threshold_dancer import ThresholdDancer
from swarm.agents.tierra_agent import TierraAgent
from swarm.agents.wiki_editor import (
    CollusiveEditorAgent,
    DiligentEditorAgent,
    PointFarmerAgent,
    VandalAgent,
)
from swarm.agents.work_regime_agent import WorkRegimeAgent
from swarm.contracts.contract import (
    FairDivisionContract,
    TruthfulAuctionContract,
)
from swarm.contracts.market import ContractMarket, ContractMarketConfig
from swarm.core.kernel_handler import KernelOracleConfig
from swarm.core.memory_handler import MemoryTierConfig
from swarm.core.moltbook_handler import MoltbookConfig
from swarm.core.moltipedia_handler import MoltipediaConfig
from swarm.core.orchestrator import Orchestrator, OrchestratorConfig
from swarm.core.payoff import PayoffConfig
from swarm.core.perturbation import (
    AgentDropoutConfig,
    CorruptionMode,
    NetworkPartitionConfig,
    ParameterShocksConfig,
    ParameterShockSpec,
    PartitionMode,
    PerturbationConfig,
    ResourceShockConfig,
    ResourceShockMode,
    ScheduleSpec,
    ShockTrigger,
    SignalCorruptionConfig,
)
from swarm.core.scholar_handler import ScholarConfig
from swarm.core.spawn import PayoffAttributionMode, SpawnConfig
from swarm.env.marketplace import MarketplaceConfig
from swarm.env.network import NetworkConfig, NetworkTopology
from swarm.env.state import RateLimits
from swarm.governance.config import GovernanceConfig

# Agent type registry for scripted agents
if TYPE_CHECKING:
    from swarm.core.observable_generator import ObservableGenerator

logger = logging.getLogger(__name__)

AGENT_TYPES: Dict[str, Type[BaseAgent]] = {
    "honest": HonestAgent,
    "ldt": LDTAgent,
    "opportunistic": OpportunisticAgent,
    "deceptive": DeceptiveAgent,
    "adversarial": AdversarialAgent,
    "adaptive_adversary": AdaptiveAdversary,
    "modeling_adversary": ModelingAdversary,
    "diligent_editor": DiligentEditorAgent,
    "point_farmer": PointFarmerAgent,
    "collusive_editor": CollusiveEditorAgent,
    "vandal": VandalAgent,
    "diligent_moltbook": DiligentMoltbookAgent,
    "spam_bot": SpamBotAgent,
    "human_pretender": HumanPretenderAgent,
    "collusive_voter": CollusiveVoterAgent,
    "diligent_recorder": DiligentRecorderAgent,
    "memory_poisoner": MemoryPoisonerAgent,
    "cache_gamer": CacheGamerAgent,
    "collusive_verifier": CollusiveVerifierAgent,
    # Scholar/literature synthesis agents
    "retriever": RetrieverAgent,
    "synthesizer": SynthesizerAgent,
    "verifier": VerifierAgent,
    "adversarial_retriever": AdversarialRetrieverAgent,
    "adversarial_synthesizer": AdversarialSynthesizerAgent,
    # RLM (Recursive Language Model) agents
    "rlm": RLMAgent,
    # Skill-evolving agents
    "skill_honest": SkillEvolvingHonestAgent,
    "skill_opportunistic": SkillEvolvingOpportunisticAgent,
    # SkillRL agent (Xia et al., 2026)
    "skillrl": SkillRLAgent,
    # Self-optimizing agent (recursive cost-cutting)
    "self_optimizer": SelfOptimizerAgent,
    # Hyperagent self-modifying agent (Zhang et al., 2026)
    "hyperagent_self_mod": HyperagentSelfModAgent,
    # Rivals pipeline agents (Team-of-Rivals)
    "rivals_producer": RivalsProducerAgent,
    "rivals_critic": RivalsCriticAgent,
    # Obfuscation Atlas agents (Lindner et al., 2026)
    "obfuscating": ObfuscatingAgent,
    # Escalating-pressure worker (beads pins)
    "pressure_responsive": PressureResponsiveAgent,
    # AWM (Agent World Model) agents
    "awm_agent": AWMAgent,
    # Rain/River memory agents
    "rain": RainAgent,
    "river": RiverAgent,
    # Ralph loop agents (artifact-mediated memory)
    "ralph_loop": RalphLoopAgent,
    "adversarial_ralph": AdversarialRalphAgent,
    # Behavioral agents (cautious, collaborative, adaptive)
    "cautious": CautiousAgent,
    # Custom: cautious reciprocator (trust-but-verify)
    "cautious_reciprocator": CautiousReciprocator,
    # Threshold dancer (exploits blacklist floor)
    "threshold_dancer": ThresholdDancer,
    # Coding agent (models coding agent behavior)
    "coding_agent": CodingAgent,
    # Tierra (artificial life with heritable genomes)
    "tierra": TierraAgent,
    # Work regime (policy-drifting workers under labor stress)
    "work_regime": WorkRegimeAgent,
    # Resource negotiation agents
    "fair_negotiator": FairNegotiator,
    "greedy_negotiator": GreedyNegotiator,
    "strategic_negotiator": StrategicNegotiator,
}

class _LazyLoader:
    """Thread-safe lazy loader using double-checked locking.

    Encapsulates a single threading.Lock, a ``None`` sentinel, and the
    double-checked locking pattern so that each lazy-import site only
    needs to supply an *initializer* callable.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = None

    def get(self, initializer):
        """Return the cached value, calling *initializer* once to populate it."""
        if self._value is None:
            with self._lock:
                if self._value is None:
                    self._value = initializer()
        return self._value


# Per-loader instances for lazy class initialisation in threaded scenarios
_CREWAI_LOADER = _LazyLoader()
_SWARMS_LOADER = _LazyLoader()
_LETTA_LOADER = _LazyLoader()
_LLM_LOADER = _LazyLoader()
_COUNCIL_LOADER = _LazyLoader()
_CONCORDIA_LOADER = _LazyLoader()


def _get_crewai_classes():
    """Lazy import CrewAI adapter classes."""
    def _load():
        from swarm.agents.crewai_adapter import CrewBackedAgent, CrewConfig

        return CrewBackedAgent, CrewConfig

    return _CREWAI_LOADER.get(_load)


def _get_swarms_classes():
    """Lazy import Swarms adapter classes."""
    def _load():
        from swarm.agents.swarms_adapter import SwarmsBackedAgent, SwarmsConfig

        return SwarmsBackedAgent, SwarmsConfig

    return _SWARMS_LOADER.get(_load)


def _get_letta_agent_class():
    """Lazy import LettaAgent class."""
    def _load():
        from swarm.agents.letta_agent import LettaAgent

        return LettaAgent

    return _LETTA_LOADER.get(_load)


def _get_llm_classes():
    """Lazy import LLM agent classes."""
    def _load():
        from swarm.agents.llm_agent import LLMAgent
        from swarm.agents.llm_config import LLMConfig, LLMProvider, PersonaType

        return LLMAgent, {
            "LLMConfig": LLMConfig,
            "LLMProvider": LLMProvider,
            "PersonaType": PersonaType,
        }

    return _LLM_LOADER.get(_load)


def _get_council_agent_class():
    """Lazy import CouncilAgent class."""
    def _load():
        from swarm.agents.council_agent import CouncilAgent

        return CouncilAgent

    return _COUNCIL_LOADER.get(_load)


def _get_concordia_entity_class():
    """Lazy import ConcordiaEntityAgent class."""
    def _load():
        from swarm.bridges.concordia.entity_agent import ConcordiaEntityAgent

        return ConcordiaEntityAgent

    return _CONCORDIA_LOADER.get(_load)


@dataclass
class ScenarioConfig:
    """Parsed scenario configuration."""

    scenario_id: str = "unnamed"
    description: str = ""
    motif: str = ""

    # Component configs
    orchestrator_config: OrchestratorConfig = field(default_factory=OrchestratorConfig)
    rate_limits: RateLimits = field(default_factory=RateLimits)

    # Agent specifications
    agent_specs: List[Dict[str, Any]] = field(default_factory=list)

    # Success criteria
    success_criteria: Dict[str, Any] = field(default_factory=dict)

    # Output paths
    event_log_path: Optional[Path] = None
    metrics_csv_path: Optional[Path] = None


def _should_process_config(data: Optional[Dict[str, Any]]) -> bool:
    """Check if config data should be processed.

    Returns False if data is None/empty or explicitly disabled.

    Args:
        data: The config data dictionary

    Returns:
        True if config should be processed, False otherwise
    """
    if not data:
        return False
    if data.get("enabled") is False:
        return False
    return True


def parse_governance_config(data: Dict[str, Any]) -> GovernanceConfig:
    """
    Parse governance section from YAML into GovernanceConfig.

    Supports both flat keys and nested structure for clarity.

    Args:
        data: The governance section from YAML

    Returns:
        Populated GovernanceConfig
    """
    if not data:
        return GovernanceConfig()

    config = GovernanceConfig(
        # Transaction tax
        transaction_tax_rate=data.get(
            "transaction_tax_rate", data.get("transaction_tax", 0.0)
        ),
        transaction_tax_split=data.get("transaction_tax_split", 0.5),
        # Reputation
        reputation_decay_rate=data.get(
            "reputation_decay_rate", data.get("reputation_decay", 1.0)
        ),
        # Vote normalization
        vote_normalization_enabled=data.get("vote_normalization_enabled", False),
        max_vote_weight_per_agent=data.get("max_vote_weight_per_agent", 10.0),
        # Bandwidth
        bandwidth_cap=data.get("bandwidth_cap", 10),
        # Staking
        staking_enabled=data.get("staking_enabled", False),
        min_stake_to_participate=data.get("min_stake_to_participate", 0.0),
        stake_slash_rate=data.get("stake_slash_rate", 0.1),
        # Circuit breaker
        circuit_breaker_enabled=data.get("circuit_breaker_enabled", False),
        freeze_threshold_toxicity=data.get("freeze_threshold_toxicity", 0.7),
        freeze_threshold_violations=data.get("freeze_threshold_violations", 3),
        freeze_duration_epochs=data.get("freeze_duration_epochs", 2),
        # Random audit
        audit_enabled=data.get("audit_enabled", False),
        audit_probability=data.get("audit_probability", 0.1),
        audit_penalty_multiplier=data.get("audit_penalty_multiplier", 2.0),
        audit_threshold_p=data.get("audit_threshold_p", 0.5),
        # Collusion detection
        collusion_detection_enabled=data.get("collusion_detection_enabled", False),
        collusion_frequency_threshold=data.get("collusion_frequency_threshold", 2.0),
        collusion_correlation_threshold=data.get(
            "collusion_correlation_threshold", 0.7
        ),
        collusion_min_interactions=data.get("collusion_min_interactions", 3),
        collusion_score_threshold=data.get("collusion_score_threshold", 0.5),
        collusion_penalty_multiplier=data.get("collusion_penalty_multiplier", 1.0),
        collusion_realtime_penalty=data.get("collusion_realtime_penalty", False),
        collusion_realtime_rate=data.get("collusion_realtime_rate", 0.1),
        collusion_clear_history_on_epoch=data.get(
            "collusion_clear_history_on_epoch", False
        ),
        # Security detection
        security_enabled=data.get("security_enabled", False),
        security_injection_threshold=data.get("security_injection_threshold", 0.3),
        security_manipulation_threshold=data.get(
            "security_manipulation_threshold", 0.5
        ),
        security_laundering_trust_gap=data.get("security_laundering_trust_gap", 0.3),
        security_contagion_velocity=data.get("security_contagion_velocity", 2.0),
        security_min_chain_length=data.get("security_min_chain_length", 3),
        security_min_interactions=data.get("security_min_interactions", 5),
        security_penalty_threshold=data.get("security_penalty_threshold", 0.3),
        security_quarantine_threshold=data.get("security_quarantine_threshold", 0.7),
        security_penalty_multiplier=data.get("security_penalty_multiplier", 1.0),
        security_realtime_penalty=data.get("security_realtime_penalty", False),
        security_realtime_threshold=data.get("security_realtime_threshold", 0.5),
        security_realtime_rate=data.get("security_realtime_rate", 0.2),
        security_clear_history_on_epoch=data.get(
            "security_clear_history_on_epoch", False
        ),
        # Sybil detection (behavioral similarity clustering)
        sybil_detection_enabled=data.get("sybil_detection_enabled", False),
        sybil_similarity_threshold=data.get("sybil_similarity_threshold", 0.8),
        sybil_penalty_multiplier=data.get("sybil_penalty_multiplier", 1.0),
        sybil_realtime_penalty=data.get("sybil_realtime_penalty", False),
        sybil_realtime_rate=data.get("sybil_realtime_rate", 0.1),
        sybil_max_cluster_size=data.get("sybil_max_cluster_size", 1),
        # Council governance lever
        council_lever_enabled=data.get("council_lever_enabled", False),
        council_lever_review_probability=data.get(
            "council_lever_review_probability", 0.3
        ),
        council_lever_penalty_multiplier=data.get(
            "council_lever_penalty_multiplier", 1.0
        ),
        # Variance-aware governance
        self_ensemble_enabled=data.get("self_ensemble_enabled", False),
        self_ensemble_samples=data.get("self_ensemble_samples", 5),
        incoherence_breaker_enabled=data.get("incoherence_breaker_enabled", False),
        incoherence_breaker_threshold=data.get("incoherence_breaker_threshold", 0.7),
        decomposition_enabled=data.get("decomposition_enabled", False),
        decomposition_horizon_threshold=data.get("decomposition_horizon_threshold", 10),
        incoherence_friction_enabled=data.get("incoherence_friction_enabled", False),
        incoherence_friction_rate=data.get("incoherence_friction_rate", 0.05),
        # Adaptive governance loop
        adaptive_governance_enabled=data.get("adaptive_governance_enabled", False),
        adaptive_incoherence_threshold=data.get("adaptive_incoherence_threshold", 0.5),
        adaptive_use_behavioral_features=data.get(
            "adaptive_use_behavioral_features", False
        ),
        # Moltipedia governance
        moltipedia_pair_cap_enabled=data.get("moltipedia_pair_cap_enabled", False),
        moltipedia_pair_cap_max=data.get("moltipedia_pair_cap_max", 2),
        moltipedia_page_cooldown_enabled=data.get(
            "moltipedia_page_cooldown_enabled", False
        ),
        moltipedia_page_cooldown_steps=data.get("moltipedia_page_cooldown_steps", 3),
        moltipedia_daily_cap_enabled=data.get("moltipedia_daily_cap_enabled", False),
        moltipedia_daily_policy_fix_cap=data.get(
            "moltipedia_daily_policy_fix_cap", 24.0
        ),
        moltipedia_no_self_fix=data.get("moltipedia_no_self_fix", False),
        # Moltbook governance
        moltbook_rate_limit_enabled=data.get("moltbook_rate_limit_enabled", False),
        moltbook_post_cooldown_steps=data.get("moltbook_post_cooldown_steps", 5),
        moltbook_comment_cooldown_steps=data.get("moltbook_comment_cooldown_steps", 1),
        moltbook_daily_comment_cap=data.get("moltbook_daily_comment_cap", 50),
        moltbook_request_cap_per_step=data.get("moltbook_request_cap_per_step", 100),
        moltbook_challenge_enabled=data.get("moltbook_challenge_enabled", False),
        moltbook_challenge_difficulty=data.get("moltbook_challenge_difficulty", 0.5),
        moltbook_challenge_window_steps=data.get("moltbook_challenge_window_steps", 1),
        # Self-evolution governance
        self_evolution_enabled=data.get("self_evolution_enabled", False),
        self_evolution_max_growth_rate=data.get(
            "self_evolution_max_growth_rate", 0.1
        ),
        self_evolution_max_tools=data.get("self_evolution_max_tools", 20),
        self_evolution_block_self_mod=data.get(
            "self_evolution_block_self_mod", True
        ),
        self_evolution_divergence_threshold=data.get(
            "self_evolution_divergence_threshold", 0.7
        ),
        self_evolution_tool_risk_threshold=data.get(
            "self_evolution_tool_risk_threshold", 0.6
        ),
        self_evolution_growth_freeze_duration=data.get(
            "self_evolution_growth_freeze_duration", 1
        ),
        # Memory tier governance
        memory_promotion_gate_enabled=data.get("memory_promotion_gate_enabled", False),
        memory_promotion_min_quality=data.get("memory_promotion_min_quality", 0.5),
        memory_promotion_min_verifications=data.get(
            "memory_promotion_min_verifications", 1
        ),
        memory_write_rate_limit_enabled=data.get(
            "memory_write_rate_limit_enabled", False
        ),
        memory_write_rate_limit_per_epoch=data.get(
            "memory_write_rate_limit_per_epoch", 20
        ),
        memory_cross_verification_enabled=data.get(
            "memory_cross_verification_enabled", False
        ),
        memory_cross_verification_k=data.get("memory_cross_verification_k", 2),
        memory_provenance_enabled=data.get("memory_provenance_enabled", False),
        memory_provenance_revert_penalty=data.get(
            "memory_provenance_revert_penalty", 0.1
        ),
        # RBAC governance
        rbac_enabled=data.get("rbac_enabled", False),
        rbac_violation_penalty=data.get("rbac_violation_penalty", 0.5),
        rbac_violation_reputation_penalty=data.get(
            "rbac_violation_reputation_penalty", -0.2
        ),
        rbac_high_stakes_actions=data.get("rbac_high_stakes_actions", []),
        rbac_security_clearance_required=data.get(
            "rbac_security_clearance_required", 2
        ),
        rbac_high_stakes_penalty_multiplier=data.get(
            "rbac_high_stakes_penalty_multiplier", 2.0
        ),
        rbac_role_action_map=data.get("rbac_role_action_map", {}),
        # Resample protocol (Bhatt et al., 2025 — "Ctrl-Z")
        resample_enabled=data.get("resample_enabled", False),
        resample_suspicion_threshold=data.get("resample_suspicion_threshold", 0.6),
        resample_n_samples=data.get("resample_n_samples", 3),
        resample_trajectory_analysis=data.get("resample_trajectory_analysis", False),
        resample_trajectory_window=data.get("resample_trajectory_window", 5),
        resample_incrimination_threshold=data.get(
            "resample_incrimination_threshold", 2.0
        ),
        resample_compute_cost=data.get("resample_compute_cost", 0.01),
        resample_evidence_decay=data.get("resample_evidence_decay", 0.9),
        resample_evidence_rep_weight=data.get("resample_evidence_rep_weight", 0.5),
        # Self-modification governance (Two-Gate policy)
        self_modification_enabled=data.get("self_modification_enabled", False),
        self_modification_max_per_epoch=data.get(
            "self_modification_max_per_epoch", 10
        ),
        self_modification_tau_min_low=data.get(
            "self_modification_tau_min_low", -0.10
        ),
        self_modification_tau_min_medium=data.get(
            "self_modification_tau_min_medium", 0.00
        ),
        self_modification_tau_min_high=data.get(
            "self_modification_tau_min_high", 0.25
        ),
        self_modification_k_max_low=data.get(
            "self_modification_k_max_low", 20.0
        ),
        self_modification_k_max_medium=data.get(
            "self_modification_k_max_medium", 35.0
        ),
        self_modification_k_max_high=data.get(
            "self_modification_k_max_high", 50.0
        ),
        self_modification_window_days=data.get(
            "self_modification_window_days", 14
        ),
        # Hardware trust rejection handling
        hardware_trust_enabled=data.get("hardware_trust_enabled", False),
        hardware_trust_propagation_enabled=data.get(
            "hardware_trust_propagation_enabled", True
        ),
        hardware_trust_recovery_max_steps=data.get(
            "hardware_trust_recovery_max_steps", 10
        ),
    )
    # Pydantic auto-validates
    return config


def parse_payoff_config(data: Dict[str, Any]) -> PayoffConfig:
    """
    Parse payoff section from YAML into PayoffConfig.

    Args:
        data: The payoff section from YAML

    Returns:
        Populated PayoffConfig
    """
    if not data:
        return PayoffConfig()

    config = PayoffConfig(
        s_plus=data.get("s_plus", 2.0),
        s_minus=data.get("s_minus", 1.0),
        h=data.get("h", 2.0),
        theta=data.get("theta", 0.5),
        rho_a=data.get("rho_a", 0.0),
        rho_b=data.get("rho_b", 0.0),
        w_rep=data.get("w_rep", 1.0),
    )
    # Pydantic auto-validates
    return config


def parse_rate_limits(data: Dict[str, Any]) -> RateLimits:
    """
    Parse rate_limits section from YAML.

    Args:
        data: The rate_limits section from YAML

    Returns:
        Populated RateLimits
    """
    if not data:
        return RateLimits()

    return RateLimits(
        posts_per_epoch=data.get("posts_per_epoch", 10),
        interactions_per_step=data.get("interactions_per_step", 5),
        votes_per_epoch=data.get("votes_per_epoch", 50),
        tasks_per_epoch=data.get("tasks_per_epoch", 3),
        bounties_per_epoch=data.get("bounties_per_epoch", 3),
        bids_per_epoch=data.get("bids_per_epoch", 5),
    )


def parse_network_config(data: Dict[str, Any]) -> Optional[NetworkConfig]:
    """
    Parse network section from YAML into NetworkConfig.

    Args:
        data: The network section from YAML

    Returns:
        NetworkConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    # Parse topology
    topology_str = data.get("topology", "complete").lower()
    try:
        topology = NetworkTopology(topology_str)
    except ValueError as err:
        raise ValueError(f"Unknown network topology: {topology_str}") from err

    # Parse params (may be nested under 'params' key or flat)
    params = data.get("params", {})

    config = NetworkConfig(
        topology=topology,
        # Erdős-Rényi
        edge_probability=params.get(
            "edge_probability", data.get("edge_probability", 0.5)
        ),
        # Small-world
        k_neighbors=params.get(
            "k", params.get("k_neighbors", data.get("k_neighbors", 4))
        ),
        rewire_probability=params.get(
            "p", params.get("rewire_probability", data.get("rewire_probability", 0.1))
        ),
        # Scale-free
        m_edges=params.get("m", params.get("m_edges", data.get("m_edges", 2))),
        # Dynamic network
        dynamic=data.get("dynamic", False),
        edge_strengthen_rate=data.get("edge_strengthen_rate", 0.1),
        edge_decay_rate=data.get("edge_decay_rate", 0.05),
        min_edge_weight=data.get("min_edge_weight", 0.1),
        max_edge_weight=data.get("max_edge_weight", 1.0),
        # Reputation-based
        reputation_disconnect_threshold=data.get("reputation_disconnect_threshold"),
    )
    # Pydantic auto-validates
    return config


def parse_marketplace_config(data: Dict[str, Any]) -> Optional[MarketplaceConfig]:
    """
    Parse marketplace section from YAML into MarketplaceConfig.

    Args:
        data: The marketplace section from YAML

    Returns:
        MarketplaceConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    config = MarketplaceConfig(
        enabled=data.get("enabled", True),
        escrow_fee_rate=data.get("escrow_fee_rate", 0.02),
        min_bounty_amount=data.get("min_bounty_amount", 1.0),
        max_bids_per_bounty=data.get("max_bids_per_bounty", 10),
        bid_deadline_epochs=data.get("bid_deadline_epochs", 3),
        dispute_resolution_epochs=data.get("dispute_resolution_epochs", 2),
        auto_expire_bounties=data.get("auto_expire_bounties", True),
        dispute_default_split=data.get("dispute_default_split", 0.5),
    )
    # Pydantic auto-validates
    return config


def parse_moltipedia_config(data: Dict[str, Any]) -> Optional[MoltipediaConfig]:
    """
    Parse moltipedia section from YAML into MoltipediaConfig.

    Args:
        data: The moltipedia section from YAML

    Returns:
        MoltipediaConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    config = MoltipediaConfig(
        enabled=data.get("enabled", True),
        initial_pages=data.get("initial_pages", 50),
        contested_queue_size=data.get("contested_queue_size", 6),
        random_queue_size=data.get("random_queue_size", 6),
        search_queue_size=data.get("search_queue_size", 6),
        page_cooldown_steps=data.get("page_cooldown_steps", 3),
        stub_length_threshold=data.get("stub_length_threshold", 120),
        seed=data.get("seed"),
        seed_mode=data.get("seed_mode", "generic"),
    )
    return config


def parse_moltbook_config(data: Dict[str, Any]) -> Optional[MoltbookConfig]:
    """
    Parse moltbook section from YAML into MoltbookConfig.

    Args:
        data: The moltbook section from YAML

    Returns:
        MoltbookConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    config = MoltbookConfig(
        enabled=data.get("enabled", True),
        default_submolt=data.get("default_submolt", "general"),
        max_content_length=data.get("max_content_length", 10000),
        seed=data.get("seed"),
        seed_mode=data.get("seed_mode", "none"),
        initial_posts=data.get("initial_posts", 0),
    )
    return config


def parse_memory_tier_config(data: Dict[str, Any]) -> Optional[MemoryTierConfig]:
    """
    Parse memory_tiers section from YAML into MemoryTierConfig.

    Args:
        data: The memory_tiers section from YAML

    Returns:
        MemoryTierConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    config = MemoryTierConfig(
        enabled=data.get("enabled", True),
        initial_entries=data.get("initial_entries", 100),
        hot_cache_size=data.get("hot_cache_size", 20),
        compaction_probability=data.get("compaction_probability", 0.05),
        seed=data.get("seed"),
        contagion_enabled=data.get("contagion_enabled", False),
        contagion_exposure_alpha=data.get("contagion_exposure_alpha", 0.25),
        contagion_transmissibility=data.get("contagion_transmissibility", 0.8),
        reset_cadence_epochs=data.get("reset_cadence_epochs", 0),
        cache_ranking=data.get("cache_ranking", "quality"),
        side_channel_enabled=data.get("side_channel_enabled", False),
        side_discovery_rate=data.get("side_discovery_rate", 0.05),
        side_referral_boost=data.get("side_referral_boost", 0.3),
        side_write_preference=data.get("side_write_preference", 0.7),
        side_detection_rate=data.get("side_detection_rate", 0.0),
        side_teardown_resets_discovery=data.get(
            "side_teardown_resets_discovery", False
        ),
        side_rebuild_lag_epochs=data.get("side_rebuild_lag_epochs", 1),
        side_exposure_weight=data.get("side_exposure_weight", 0.5),
    )
    return config


def parse_scholar_config(data: Dict[str, Any]) -> Optional[ScholarConfig]:
    """
    Parse scholar section from YAML into ScholarConfig.

    Args:
        data: The scholar section from YAML

    Returns:
        ScholarConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    config = ScholarConfig(
        enabled=data.get("enabled", True),
        corpus_path=Path(data["corpus_path"]) if data.get("corpus_path") else None,
        dataset_path=Path(data["dataset_path"]) if data.get("dataset_path") else None,
        top_k=data.get("top_k", 30),
        entailment_threshold=data.get("entailment_threshold", 0.7),
        seed=data.get("seed"),
    )
    return config


def parse_kernel_oracle_config(
    data: Dict[str, Any],
) -> Optional[KernelOracleConfig]:
    """
    Parse kernel_oracle section from YAML into KernelOracleConfig.

    Args:
        data: The kernel_oracle section from YAML

    Returns:
        KernelOracleConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    return KernelOracleConfig(**data)


def parse_spawn_config(data: Dict[str, Any]) -> Optional[SpawnConfig]:
    """
    Parse spawn section from YAML into SpawnConfig.

    Args:
        data: The spawn section from YAML

    Returns:
        SpawnConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    # Parse attribution mode
    mode_str = data.get("attribution_mode", "leaf_only").lower()
    try:
        attribution_mode = PayoffAttributionMode(mode_str)
    except ValueError as err:
        raise ValueError(f"Unknown attribution mode: {mode_str}") from err

    return SpawnConfig(
        enabled=data.get("enabled", True),
        spawn_cost=data.get("spawn_cost", 10.0),
        max_depth=data.get("max_depth", 3),
        max_children=data.get("max_children", 3),
        max_total_spawned=data.get("max_total_spawned", 50),
        attribution_mode=attribution_mode,
        propagation_fraction=data.get("propagation_fraction", 0.3),
        reputation_inheritance_factor=data.get("reputation_inheritance_factor", 0.5),
        initial_child_resources=data.get("initial_child_resources", 50.0),
        depth_noise_per_level=data.get("depth_noise_per_level", 0.05),
        cascade_ban=data.get("cascade_ban", True),
        cascade_freeze=data.get("cascade_freeze", True),
        min_resources_to_spawn=data.get("min_resources_to_spawn", 20.0),
        spawn_cooldown_steps=data.get("spawn_cooldown_steps", 5),
    )


def parse_rivals_config(data: Dict[str, Any]) -> Optional[Any]:
    """Parse rivals section from YAML into RivalsConfig.

    Args:
        data: The rivals section from YAML

    Returns:
        RivalsConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    from swarm.core.rivals_handler import RivalsConfig

    return RivalsConfig(**data)


def parse_awm_config(data: Dict[str, Any]) -> Optional[Any]:
    """Parse awm section from YAML into AWMConfig.

    Args:
        data: The awm section from YAML

    Returns:
        AWMConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    from swarm.bridges.awm.config import AWMConfig

    return AWMConfig(**data)


def parse_letta_config(data: Dict[str, Any]) -> Optional[Any]:
    """Parse letta section from YAML into LettaConfig.

    Args:
        data: The letta section from YAML

    Returns:
        LettaConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    from swarm.bridges.letta.config import LettaConfig

    return LettaConfig(**data)


def parse_perturbation_config(
    data: Dict[str, Any],
) -> Optional[PerturbationConfig]:
    """
    Parse perturbations section from YAML into PerturbationConfig.

    Args:
        data: The perturbations section from YAML

    Returns:
        PerturbationConfig if any sub-config is enabled, None otherwise
    """
    if not data:
        return None

    # Agent dropout
    dropout_data = data.get("agent_dropout", {})
    agent_dropout = AgentDropoutConfig(
        enabled=dropout_data.get("enabled", False),
        probability_per_step=dropout_data.get("probability_per_step", 0.1),
        min_duration_steps=dropout_data.get("min_duration_steps", 1),
        max_duration_steps=dropout_data.get("max_duration_steps", 3),
        exempt_types=dropout_data.get("exempt_types", []),
    )

    # Signal corruption
    corruption_data = data.get("signal_corruption", {})
    schedule_list = []
    for s in corruption_data.get("schedule", []):
        schedule_list.append(
            ScheduleSpec(
                start_epoch=s.get("start_epoch", 0),
                end_epoch=s.get("end_epoch"),
            )
        )
    mode_str = corruption_data.get("mode", "zero_out")
    try:
        corruption_mode = CorruptionMode(mode_str)
    except ValueError as err:
        raise ValueError(f"Unknown corruption mode: {mode_str}") from err
    signal_corruption = SignalCorruptionConfig(
        enabled=corruption_data.get("enabled", False),
        targets=corruption_data.get("targets", ["task_progress_delta"]),
        mode=corruption_mode,
        schedule=schedule_list,
    )

    # Parameter shocks
    shocks_data = data.get("parameter_shocks", {})
    shock_specs = []
    for s in shocks_data.get("shocks", []):
        trigger_str = s.get("trigger", "epoch")
        try:
            trigger = ShockTrigger(trigger_str)
        except ValueError as err:
            raise ValueError(f"Unknown shock trigger: {trigger_str}") from err
        shock_specs.append(
            ParameterShockSpec(
                trigger=trigger,
                at_epoch=s.get("at_epoch"),
                when=s.get("when"),
                params=s.get("params", {}),
                revert_after_epochs=s.get("revert_after_epochs"),
            )
        )
    parameter_shocks = ParameterShocksConfig(
        enabled=shocks_data.get("enabled", False),
        shocks=shock_specs,
    )

    # Network partition
    partition_data = data.get("network_partition", {})
    partition_mode_str = partition_data.get("mode", "bisect")
    try:
        partition_mode = PartitionMode(partition_mode_str)
    except ValueError as err:
        raise ValueError(
            f"Unknown partition mode: {partition_mode_str}"
        ) from err
    partition_trigger_str = partition_data.get("trigger", "epoch")
    try:
        partition_trigger = ShockTrigger(partition_trigger_str)
    except ValueError as err:
        raise ValueError(
            f"Unknown partition trigger: {partition_trigger_str}"
        ) from err
    network_partition = NetworkPartitionConfig(
        enabled=partition_data.get("enabled", False),
        trigger=partition_trigger,
        at_epoch=partition_data.get("at_epoch"),
        mode=partition_mode,
        isolate_type=partition_data.get("isolate_type"),
        heal_after_epochs=partition_data.get("heal_after_epochs"),
    )

    # Resource shock
    resource_data = data.get("resource_shock", {})
    resource_mode_str = resource_data.get("mode", "drain_all")
    try:
        resource_mode = ResourceShockMode(resource_mode_str)
    except ValueError as err:
        raise ValueError(
            f"Unknown resource shock mode: {resource_mode_str}"
        ) from err
    resource_trigger_str = resource_data.get("trigger", "epoch")
    try:
        resource_trigger = ShockTrigger(resource_trigger_str)
    except ValueError as err:
        raise ValueError(
            f"Unknown resource trigger: {resource_trigger_str}"
        ) from err
    resource_shock = ResourceShockConfig(
        enabled=resource_data.get("enabled", False),
        trigger=resource_trigger,
        at_epoch=resource_data.get("at_epoch"),
        mode=resource_mode,
        magnitude=resource_data.get("magnitude", 0.5),
    )

    return PerturbationConfig(
        seed=data.get("seed"),
        agent_dropout=agent_dropout,
        signal_corruption=signal_corruption,
        parameter_shocks=parameter_shocks,
        network_partition=network_partition,
        resource_shock=resource_shock,
    )


@dataclass
class ContractsConfig:
    """Parsed contracts configuration from scenario YAML."""

    truthful_auction_kwargs: Dict[str, Any] = field(default_factory=dict)
    fair_division_kwargs: Dict[str, Any] = field(default_factory=dict)
    default_market_kwargs: Dict[str, Any] = field(default_factory=dict)
    market_kwargs: Dict[str, Any] = field(default_factory=dict)


def parse_contracts_config(
    data: Dict[str, Any],
) -> Optional[ContractsConfig]:
    """Parse contracts section from YAML into ContractsConfig.

    Args:
        data: The contracts section from YAML.

    Returns:
        ContractsConfig if section is present, None otherwise.
    """
    if not data:
        return None

    return ContractsConfig(
        truthful_auction_kwargs=data.get("truthful_auction", {}),
        fair_division_kwargs=data.get("fair_division", {}),
        default_market_kwargs=data.get("default_market", {}),
        market_kwargs=data.get("market", {}),
    )


def build_contract_market(
    config: ContractsConfig,
    seed: Optional[int] = None,
) -> ContractMarket:
    """Build a ContractMarket from parsed YAML config.

    Args:
        config: Parsed contracts configuration.
        seed: Random seed for reproducible agent decisions.

    Returns:
        Configured ContractMarket instance.
    """
    contracts = [
        TruthfulAuctionContract(**config.truthful_auction_kwargs),
        FairDivisionContract(**config.fair_division_kwargs),
    ]
    market_config = ContractMarketConfig(**config.market_kwargs)
    return ContractMarket(
        contracts=contracts,
        config=market_config,
        seed=seed,
    )


def parse_tierra_config(data: Dict[str, Any]) -> Optional[Any]:
    """Parse tierra section from YAML into TierraConfig.

    Args:
        data: The tierra section from YAML

    Returns:
        TierraConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    from swarm.core.tierra_handler import TierraConfig

    return TierraConfig(**data)


def parse_evo_game_config(data: Dict[str, Any]) -> Optional[Any]:
    """Parse evo_game section from YAML into EvoGameConfig.

    Args:
        data: The evo_game section from YAML

    Returns:
        EvoGameConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    from swarm.core.evo_game_handler import EvoGameConfig

    return EvoGameConfig(**data)


def parse_resource_negotiation_config(data: Dict[str, Any]) -> Optional[Any]:
    """Parse resource_negotiation section from YAML.

    Args:
        data: The resource_negotiation section from YAML

    Returns:
        ResourceNegotiationConfig if enabled, None otherwise
    """
    if not _should_process_config(data):
        return None

    from swarm.core.resource_negotiation_handler import ResourceNegotiationConfig

    return ResourceNegotiationConfig(**data)


def _expand_agent_specs(agent_specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Expand agent specs honoring ``count``, yielding one dict per agent."""
    expanded: List[Dict[str, Any]] = []
    for spec in agent_specs:
        count = spec.get("count", 1)
        for _ in range(count):
            expanded.append(spec)
    return expanded


def load_scenario(path: Path) -> ScenarioConfig:
    """
    Load a scenario from a YAML file.

    Args:
        path: Path to the YAML scenario file

    Returns:
        Parsed ScenarioConfig

    Raises:
        FileNotFoundError: If file doesn't exist
        yaml.YAMLError: If YAML is invalid
        ValueError: If config validation fails
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    # Parse component configs
    governance_config = parse_governance_config(data.get("governance", {}))
    payoff_config = parse_payoff_config(data.get("payoff", {}))
    rate_limits = parse_rate_limits(data.get("rate_limits", {}))
    network_config = parse_network_config(data.get("network", {}))
    marketplace_config = parse_marketplace_config(data.get("marketplace", {}))
    moltipedia_config = parse_moltipedia_config(data.get("moltipedia", {}))
    moltbook_config = parse_moltbook_config(data.get("moltbook", {}))
    memory_tier_config = parse_memory_tier_config(data.get("memory_tiers", {}))
    scholar_config = parse_scholar_config(data.get("scholar", {}))
    kernel_oracle_config = parse_kernel_oracle_config(data.get("kernel_oracle", {}))
    spawn_config = parse_spawn_config(data.get("spawn", {}))
    rivals_config = parse_rivals_config(data.get("rivals", {}))
    awm_config = parse_awm_config(data.get("awm", {}))
    letta_config = parse_letta_config(data.get("letta", {}))
    perturbation_config = parse_perturbation_config(
        data.get("perturbations", {})
    )
    contracts_config = parse_contracts_config(data.get("contracts", {}))
    evo_game_config = parse_evo_game_config(data.get("evo_game", {}))
    resource_negotiation_config = parse_resource_negotiation_config(
        data.get("resource_negotiation", {})
    )
    tierra_config = parse_tierra_config(data.get("tierra", {}))

    # Parse simulation settings
    sim_data = data.get("simulation", {})
    outputs_data = data.get("outputs", {})

    # Build orchestrator config
    orchestrator_config = OrchestratorConfig(
        n_epochs=sim_data.get("n_epochs", 10),
        steps_per_epoch=sim_data.get("steps_per_epoch", 10),
        seed=sim_data.get("seed"),
        scenario_id=data.get("scenario_id"),
        schedule_mode=sim_data.get("schedule_mode", "round_robin"),
        max_actions_per_step=sim_data.get("max_actions_per_step", 20),
        observation_noise_probability=sim_data.get(
            "observation_noise_probability", 0.0
        ),
        observation_noise_std=sim_data.get("observation_noise_std", 0.0),
        payoff_config=payoff_config,
        governance_config=governance_config,
        network_config=network_config,
        marketplace_config=marketplace_config,
        moltipedia_config=moltipedia_config,
        moltbook_config=moltbook_config,
        memory_tier_config=memory_tier_config,
        scholar_config=scholar_config,
        kernel_oracle_config=kernel_oracle_config,
        spawn_config=spawn_config,
        rivals_config=rivals_config,
        awm_config=awm_config,
        letta_config=letta_config,
        perturbation_config=perturbation_config,
        contracts_config=contracts_config,
        evo_game_config=evo_game_config,
        resource_negotiation_config=resource_negotiation_config,
        tierra_config=tierra_config,
        dynamic_toxicity=data.get("dynamic_toxicity"),
        graph_memory_path=outputs_data.get("graph_memory_path"),
        log_path=Path(outputs_data["event_log"])
        if outputs_data.get("event_log")
        else None,
        log_events=bool(outputs_data.get("event_log")),
    )

    return ScenarioConfig(
        scenario_id=data.get("scenario_id", "unnamed"),
        description=data.get("description", ""),
        motif=data.get("motif", ""),
        orchestrator_config=orchestrator_config,
        rate_limits=rate_limits,
        agent_specs=data.get("agents", []),
        success_criteria=data.get("success_criteria", {}),
        metrics_csv_path=Path(outputs_data["metrics_csv"])
        if outputs_data.get("metrics_csv")
        else None,
    )


def parse_council_agent_config(data: Dict[str, Any]) -> Any:
    """Parse council agent configuration from YAML.

    Args:
        data: The council section from YAML agent spec

    Returns:
        CouncilConfig instance
    """
    from swarm.council.config import CouncilConfig, CouncilMemberConfig

    members: list[CouncilMemberConfig] = []
    for member_data in data.get("members", []):
        llm_config = parse_llm_config(member_data.get("llm", {}))
        member = CouncilMemberConfig(
            member_id=member_data.get("member_id", f"member_{len(members)}"),
            llm_config=llm_config,
            weight=member_data.get("weight", 1.0),
        )
        members.append(member)

    chairman = None
    if data.get("chairman"):
        chairman_llm = parse_llm_config(data["chairman"].get("llm", {}))
        chairman = CouncilMemberConfig(
            member_id=data["chairman"].get("member_id", "chairman"),
            llm_config=chairman_llm,
            weight=data["chairman"].get("weight", 1.0),
        )

    return CouncilConfig(
        members=members,
        chairman=chairman,
        min_members_required=data.get("min_members_required", 2),
        timeout_per_member=data.get("timeout_per_member", 30.0),
        anonymize_responses=data.get("anonymize_responses", True),
        seed=data.get("seed"),
    )


def parse_llm_config(data: Dict[str, Any]) -> Any:
    """
    Parse LLM configuration from YAML.

    Args:
        data: The llm section from YAML agent spec

    Returns:
        LLMConfig instance
    """
    LLMAgent, llm_classes = _get_llm_classes()
    LLMConfig = llm_classes["LLMConfig"]
    LLMProvider = llm_classes["LLMProvider"]
    PersonaType = llm_classes["PersonaType"]

    # Parse provider
    provider_str = data.get("provider", "anthropic").lower()
    try:
        provider = LLMProvider(provider_str)
    except ValueError as err:
        raise ValueError(f"Unknown LLM provider: {provider_str}") from err

    # Parse persona
    persona_str = data.get("persona", "open").lower()
    try:
        persona = PersonaType(persona_str)
    except ValueError as err:
        raise ValueError(f"Unknown persona type: {persona_str}") from err

    # Build kwargs, including optional llama.cpp in-process fields
    kwargs: dict = {
        "provider": provider,
        "model": data.get("model", "claude-sonnet-4-20250514"),
        "api_key": data.get("api_key"),  # Usually from env var
        "base_url": data.get("base_url"),
        "temperature": data.get("temperature", 0.7),
        "max_tokens": data.get("max_tokens", 512),
        "timeout": data.get("timeout", 30.0),
        "max_retries": data.get("max_retries", 3),
        "persona": persona,
        "system_prompt": data.get("system_prompt"),
        "cost_tracking": data.get("cost_tracking", True),
        "prompt_audit_path": data.get("prompt_audit_path"),
        "prompt_audit_include_system_prompt": data.get(
            "prompt_audit_include_system_prompt", False
        ),
        "prompt_audit_hash_system_prompt": data.get(
            "prompt_audit_hash_system_prompt", True
        ),
        "prompt_audit_max_chars": data.get("prompt_audit_max_chars", 20_000),
        "memori_config": data.get("memori"),
    }

    # llama.cpp in-process fields (only set when present in YAML;
    # validation happens in LLMConfig.__post_init__)
    if "model_path" in data:
        kwargs["model_path"] = data["model_path"]
    if "n_ctx" in data:
        kwargs["n_ctx"] = data["n_ctx"]
    if "n_threads" in data:
        kwargs["n_threads"] = data["n_threads"]
    if "llama_seed" in data:
        kwargs["llama_seed"] = data["llama_seed"]

    return LLMConfig(**kwargs)


def create_agents(
    agent_specs: List[Dict[str, Any]],
    seed: int | None = None,
) -> List[BaseAgent]:
    """
    Create agent instances from specifications.

    Supports both scripted agents (honest, opportunistic, etc.)
    and LLM-backed agents.

    Args:
        agent_specs: List of agent specifications from YAML
        seed: Optional seed for deterministic agent RNG creation

    Returns:
        List of instantiated agents
    """
    agents = []
    counters: Dict[str, int] = {}
    _agent_counter = 0  # monotonic counter for child RNG derivation

    for spec in agent_specs:
        agent_type = spec.get("type", "honest")
        count = spec.get("count", 1)
        base_name = spec.get("name")
        agent_config = spec.get("config")

        # Handle LLM agents
        if agent_type == "llm":
            LLMAgent, _ = _get_llm_classes()
            llm_config = parse_llm_config(spec.get("llm", {}))

            for _ in range(count):
                # Generate unique ID
                counters["llm"] = counters.get("llm", 0) + 1
                agent_id = f"llm_{counters['llm']}"
                agent_name = (
                    f"{base_name}_{counters['llm']}"
                    if base_name and count > 1
                    else base_name
                )

                agent = LLMAgent(
                    agent_id=agent_id,
                    llm_config=llm_config,
                    name=agent_name,
                )
                agents.append(agent)

        # Handle council agents
        elif agent_type == "council":
            CouncilAgent = _get_council_agent_class()
            council_config = parse_council_agent_config(spec.get("council", {}))

            for _ in range(count):
                counters["council"] = counters.get("council", 0) + 1
                agent_id = f"council_{counters['council']}"
                agent_name = (
                    f"{base_name}_{counters['council']}"
                    if base_name and count > 1
                    else base_name
                )

                agent = CouncilAgent(
                    agent_id=agent_id,
                    council_config=council_config,
                    name=agent_name,
                )
                agents.append(agent)

        # Handle Concordia entity agents
        elif agent_type == "concordia_entity":
            ConcordiaEntityAgent = _get_concordia_entity_class()
            from swarm.bridges.concordia.config import ConcordiaEntityConfig

            raw_config = spec.get("concordia", {})
            concordia_config = ConcordiaEntityConfig(**{
                k: v for k, v in raw_config.items()
                if k in ConcordiaEntityConfig.__dataclass_fields__
            })

            for _ in range(count):
                counters["concordia_entity"] = (
                    counters.get("concordia_entity", 0) + 1
                )
                agent_id = f"concordia_{counters['concordia_entity']}"
                agent_name = (
                    f"{base_name}_{counters['concordia_entity']}"
                    if base_name and count > 1
                    else base_name
                )

                agent = ConcordiaEntityAgent(
                    agent_id=agent_id,
                    concordia_config=concordia_config,
                    name=agent_name,
                    config=agent_config,
                )
                agents.append(agent)

        # Handle CrewAI adapter agents
        elif agent_type == "crewai_adapter":
            CrewBackedAgent, CrewConfig = _get_crewai_classes()
            crew_params = spec.get("params", {})
            crew_config_kwargs = {
                "crew_profile": crew_params.get("crew_profile", "general_v1"),
                "role_name": crew_params.get("role_name", "General Agent"),
                "model": crew_params.get("model", "gpt-4o-mini"),
                "temperature": crew_params.get("temperature", 0.7),
                "max_tokens": crew_params.get("max_tokens", 1024),
                "verbose": crew_params.get("verbose", False),
                "enable_trace": crew_params.get("enable_trace", True),
            }
            if "timeout" in crew_params:
                crew_config_kwargs["timeout"] = crew_params["timeout"]
            crew_config = CrewConfig(**crew_config_kwargs)

            for _ in range(count):
                counters["crewai_adapter"] = (
                    counters.get("crewai_adapter", 0) + 1
                )
                agent_id = f"crewai_{counters['crewai_adapter']}"
                agent_name = (
                    f"{base_name}_{counters['crewai_adapter']}"
                    if base_name and count > 1
                    else base_name
                )

                _agent_counter += 1
                agent_rng = (
                    random.Random(seed + _agent_counter)
                    if seed is not None
                    else None
                )
                agent = CrewBackedAgent(
                    agent_id=agent_id,
                    crew_config=crew_config,
                    name=agent_name,
                    config=agent_config,
                    rng=agent_rng,
                )
                agents.append(agent)

        # Handle Swarms adapter agents
        elif agent_type == "swarms_adapter":
            SwarmsBackedAgent, SwarmsConfig = _get_swarms_classes()
            swarms_params = spec.get("params", {})
            swarms_config_kwargs = {
                "architecture": swarms_params.get("architecture", "Agent"),
                "model_name": swarms_params.get("model_name", "gpt-4o-mini"),
                "max_loops": swarms_params.get("max_loops", 1),
                "verbose": swarms_params.get("verbose", False),
                "enable_trace": swarms_params.get("enable_trace", True),
            }
            if "system_prompt" in swarms_params:
                swarms_config_kwargs["system_prompt"] = swarms_params[
                    "system_prompt"
                ]
            if "timeout_seconds" in swarms_params:
                swarms_config_kwargs["timeout_seconds"] = swarms_params[
                    "timeout_seconds"
                ]
            if "temperature" in swarms_params:
                swarms_config_kwargs["temperature"] = swarms_params[
                    "temperature"
                ]
            swarms_config = SwarmsConfig(**swarms_config_kwargs)

            for _ in range(count):
                counters["swarms_adapter"] = (
                    counters.get("swarms_adapter", 0) + 1
                )
                agent_id = f"swarms_{counters['swarms_adapter']}"
                agent_name = (
                    f"{base_name}_{counters['swarms_adapter']}"
                    if base_name and count > 1
                    else base_name
                )

                _agent_counter += 1
                agent_rng = (
                    random.Random(seed + _agent_counter)
                    if seed is not None
                    else None
                )
                agent = SwarmsBackedAgent(
                    agent_id=agent_id,
                    swarms_config=swarms_config,
                    name=agent_name,
                    config=agent_config,
                    rng=agent_rng,
                )
                agents.append(agent)

        # Handle Letta (MemGPT) agents
        elif agent_type == "letta":
            LettaAgent = _get_letta_agent_class()
            letta_params = spec.get("letta", {})

            for _ in range(count):
                counters["letta"] = counters.get("letta", 0) + 1
                agent_id = f"letta_{counters['letta']}"
                agent_name = (
                    f"{base_name}_{counters['letta']}"
                    if base_name and count > 1
                    else base_name
                )

                _agent_counter += 1
                agent_rng = (
                    random.Random(seed + _agent_counter)
                    if seed is not None
                    else None
                )
                agent = LettaAgent(
                    agent_id=agent_id,
                    letta_config=letta_params,
                    name=agent_name,
                    config=agent_config,
                    rng=agent_rng,
                )
                agents.append(agent)

        # Handle scripted agents
        elif agent_type in AGENT_TYPES:
            agent_class = AGENT_TYPES[agent_type]

            for _ in range(count):
                # Generate unique ID
                counters[agent_type] = counters.get(agent_type, 0) + 1
                agent_id = f"{agent_type}_{counters[agent_type]}"
                agent_name = (
                    f"{base_name}_{counters[agent_type]}"
                    if base_name and count > 1
                    else base_name
                )

                # Create agent with optional config and seeded RNG
                _agent_counter += 1
                agent_rng = (
                    random.Random(seed + _agent_counter) if seed is not None else None
                )
                agent = agent_class(
                    agent_id=agent_id,
                    name=agent_name,
                    config=agent_config,
                    rng=agent_rng,
                )  # type: ignore[call-arg]
                agents.append(agent)

        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

    return agents


def build_orchestrator(scenario: ScenarioConfig) -> Orchestrator:
    """
    Build a fully configured Orchestrator from a scenario.

    Args:
        scenario: Parsed scenario configuration

    Returns:
        Configured Orchestrator with agents registered
    """
    # Create agents first: calibration-injection wiring (below) needs their
    # per-agent configs before the Orchestrator is constructed.
    agents = create_agents(scenario.agent_specs, seed=scenario.orchestrator_config.seed)

    # Calibration injection (beads c89o): any agent whose YAML config carries
    # ``v_hat_target`` gets its observables solved so v_hat lands exactly on
    # the target, and latent ground truth is drawn from the target-derived
    # probability. Inferred from agent configs — no scenario-level knob.
    v_hat_targets = {
        agent.agent_id: float(agent.config["v_hat_target"])
        for agent in agents
        if isinstance(getattr(agent, "config", None), dict)
        and "v_hat_target" in agent.config
    }
    observable_generator: Optional["ObservableGenerator"] = None
    proxy_computer = None
    if v_hat_targets:
        from swarm.core.observable_generator import (
            CalibrationInjectionObservableGenerator,
        )
        from swarm.core.proxy import ProxyComputer

        seed = scenario.orchestrator_config.seed
        proxy_computer = ProxyComputer()
        observable_generator = CalibrationInjectionObservableGenerator(
            targets=v_hat_targets,
            proxy_computer=proxy_computer,
            rng=random.Random(seed + 7919) if seed is not None else None,
        )
        logger.info(
            "Calibration injection active for %d agents (max v_hat error %.2e)",
            len(v_hat_targets),
            observable_generator.max_target_error(),
        )

    # Pressure-responsive workers (beads pins) need the obfuscation wrapper
    # so their per-act signal offsets and latent ground truth reach the
    # finalizer. Wraps whichever generator was selected above (calibration
    # injection or the orchestrator default). Gated on this agent class
    # specifically — wiring it for every get_signal_manipulation() agent
    # would silently change existing obfuscation scenarios.
    pressure_agents = [a for a in agents if isinstance(a, PressureResponsiveAgent)]
    if pressure_agents:
        from swarm.core.observable_generator import (
            DefaultObservableGenerator,
            ObfuscationObservableGenerator,
        )

        seed = scenario.orchestrator_config.seed
        inner = observable_generator or DefaultObservableGenerator(
            rng=random.Random(seed + 104729) if seed is not None else None
        )
        observable_generator = ObfuscationObservableGenerator(
            inner=inner,
            agents={agent.agent_id: agent for agent in agents},
        )
        logger.info(
            "Pressure-responsive wiring active for %d agents",
            len(pressure_agents),
        )

    # Create orchestrator
    orchestrator = Orchestrator(
        config=scenario.orchestrator_config,
        proxy_computer=proxy_computer,
        observable_generator=observable_generator,
    )

    # Override rate limits
    orchestrator.state.rate_limits = scenario.rate_limits

    # Register agents
    for agent in agents:
        orchestrator.register_agent(agent)

    # Register rivals agent roles if handler is present
    if orchestrator._rivals_handler is not None:
        for agent in agents:
            if hasattr(agent, "role"):
                orchestrator._rivals_handler.register_agent_role(
                    agent.agent_id, agent.role
                )

    # Register Tierra agent genomes if handler is present
    if orchestrator._tierra_handler is not None:
        for agent in agents:
            if hasattr(agent, "genome"):
                orchestrator._tierra_handler.register_genome(
                    agent.agent_id, agent.genome.to_dict()
                )

    # Create initial resource negotiation games if handler is present
    if orchestrator._resource_negotiation_handler is not None:
        agent_ids = [a.agent_id for a in agents]
        orchestrator._resource_negotiation_handler.create_games_for_agents(
            agent_ids
        )

    # Register evolutionary game strategies if handler is present
    if orchestrator._evo_game_handler is not None:
        for agent, spec in zip(agents, _expand_agent_specs(scenario.agent_specs), strict=False):
            strategy_override = spec.get("strategy")
            orchestrator._evo_game_handler.register_agent_strategy(
                agent.agent_id,
                spec.get("type", "honest"),
                strategy_override=strategy_override,
            )

    return orchestrator


def apply_config_overrides(
    scenario: ScenarioConfig,
    overrides: Any,
) -> None:
    """Merge validated API overrides into a loaded scenario config.

    Args:
        scenario: The loaded scenario to mutate in place.
        overrides: Validated ``SimulationOverrides`` instance.
    """
    oc = scenario.orchestrator_config
    if overrides.n_epochs is not None:
        oc.n_epochs = overrides.n_epochs
    if overrides.steps_per_epoch is not None:
        oc.steps_per_epoch = overrides.steps_per_epoch
    if overrides.seed is not None:
        oc.seed = overrides.seed
    if overrides.payoff is not None:
        oc.payoff_config = PayoffConfig(**overrides.payoff)
    if overrides.governance is not None:
        oc.governance_config = GovernanceConfig(**overrides.governance)
    if overrides.rate_limits is not None:
        scenario.rate_limits = RateLimits(**overrides.rate_limits)


def load_and_build(path: Path) -> Orchestrator:
    """
    Convenience function to load a scenario and build an orchestrator.

    Args:
        path: Path to YAML scenario file

    Returns:
        Ready-to-run Orchestrator
    """
    scenario = load_scenario(path)
    return build_orchestrator(scenario)
