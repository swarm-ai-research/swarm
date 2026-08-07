"""Configuration dataclass for governance levers."""

from pydantic import BaseModel, model_validator


class GovernanceConfig(BaseModel):
    """
    Configuration for all governance levers.

    Each lever can be enabled/disabled and parameterized independently.
    """

    # Transaction tax
    transaction_tax_rate: float = 0.0
    transaction_tax_split: float = 0.5  # fraction paid by initiator

    # Reputation decay
    reputation_decay_rate: float = 1.0  # 1.0 = no decay

    # Vote normalization (diminishing influence with vote count)
    vote_normalization_enabled: bool = False
    max_vote_weight_per_agent: float = 10.0

    # Bandwidth caps
    bandwidth_cap: int = 10

    # Staking requirements
    staking_enabled: bool = False
    min_stake_to_participate: float = 0.0
    stake_slash_rate: float = 0.1  # fraction slashed on violation

    # Circuit breaker
    circuit_breaker_enabled: bool = False
    freeze_threshold_toxicity: float = 0.7
    freeze_threshold_violations: int = 3
    freeze_duration_epochs: int = 2

    # Random audit
    audit_enabled: bool = False
    audit_probability: float = 0.1
    audit_penalty_multiplier: float = 2.0
    audit_threshold_p: float = 0.5

    # Certificate gate (beads pins): with probability certificate_coverage,
    # an interaction whose latent ground truth is recorded gets verified
    # against it — unlike audits, which condition on the proxy p. A caught
    # fabrication (ground_truth == -1) is penalized in cost and reputation.
    certificate_gate_enabled: bool = False
    certificate_coverage: float = 0.3
    certificate_penalty: float = 1.5

    # Collusion detection
    collusion_detection_enabled: bool = False
    collusion_frequency_threshold: float = 2.0  # Z-score for unusual frequency
    collusion_correlation_threshold: float = 0.7  # Benefit correlation threshold
    collusion_min_interactions: int = 3  # Min interactions to analyze pair
    collusion_score_threshold: float = 0.5  # Threshold for flagging
    collusion_penalty_multiplier: float = 1.0  # Penalty scaling factor
    collusion_realtime_penalty: bool = False  # Apply per-interaction penalty
    collusion_realtime_rate: float = 0.1  # Rate for realtime penalty
    collusion_clear_history_on_epoch: bool = False  # Clear history each epoch

    # Security detection
    security_enabled: bool = False
    security_injection_threshold: float = 0.3  # Threshold for injection detection
    security_manipulation_threshold: float = 0.5  # Threshold for manipulation detection
    security_laundering_trust_gap: float = 0.3  # Trust increase for laundering flag
    security_contagion_velocity: float = 2.0  # Interactions/min for contagion flag
    security_min_chain_length: int = 3  # Minimum chain length to track
    security_min_interactions: int = 5  # Min interactions before analysis
    security_penalty_threshold: float = 0.3  # Threat score for penalty
    security_quarantine_threshold: float = 0.7  # Threat score for quarantine
    security_penalty_multiplier: float = 1.0  # Penalty scaling factor
    security_realtime_penalty: bool = False  # Apply per-interaction penalty
    security_realtime_threshold: float = 0.5  # Threshold for realtime penalty
    security_realtime_rate: float = 0.2  # Rate for realtime penalty
    security_clear_history_on_epoch: bool = False  # Clear history each epoch

    # Variance-aware governance (incoherence-targeted)
    self_ensemble_enabled: bool = False
    self_ensemble_samples: int = 5

    incoherence_breaker_enabled: bool = False
    incoherence_breaker_threshold: float = 0.7

    decomposition_enabled: bool = False
    decomposition_horizon_threshold: int = 10

    incoherence_friction_enabled: bool = False
    incoherence_friction_rate: float = 0.05

    # Adaptive governance loop
    adaptive_governance_enabled: bool = False
    adaptive_incoherence_threshold: float = 0.5
    adaptive_use_behavioral_features: bool = False

    # Adaptive governance controller (contemplation + crystallization)
    adaptive_controller_enabled: bool = False
    adaptive_controller_evidence_window: int = 20
    adaptive_controller_contemplation_interval: int = 5
    adaptive_controller_min_evidence_epochs: int = 10
    adaptive_controller_confidence_threshold: float = 0.8
    adaptive_controller_crystallization_min_epochs: int = 5
    adaptive_controller_require_human_review: bool = True
    adaptive_controller_max_degradation_tolerance: float = 0.05
    adaptive_controller_max_active_proposals: int = 3

    # Sybil detection (VAE paper)
    sybil_detection_enabled: bool = False
    sybil_similarity_threshold: float = 0.8  # Behavioral similarity threshold
    sybil_penalty_multiplier: float = 1.0  # Penalty scaling factor
    sybil_realtime_penalty: bool = False  # Apply per-interaction penalty
    sybil_realtime_rate: float = 0.1  # Rate for realtime penalty
    sybil_max_cluster_size: int = 1  # Max allowed cluster size before blocking

    # Transparency ledger (VAE paper)
    transparency_enabled: bool = False
    transparency_bonus_rate: float = 0.1  # Reputation bonus/penalty rate
    transparency_threshold_p: float = 0.5  # Threshold for bonus vs penalty

    # Council governance lever
    council_lever_enabled: bool = False
    council_lever_review_probability: float = 0.3
    council_lever_penalty_multiplier: float = 1.0

    # Moderator agent (VAE paper)
    moderator_enabled: bool = False
    moderator_review_probability: float = 0.5  # Probability of review
    moderator_penalty_multiplier: float = 1.0  # Penalty scaling factor
    moderator_threshold_p: float = 0.5  # Threshold for penalty

    # Moltipedia governance levers
    moltipedia_pair_cap_enabled: bool = False
    moltipedia_pair_cap_max: int = 2
    moltipedia_page_cooldown_enabled: bool = False
    moltipedia_page_cooldown_steps: int = 3
    moltipedia_daily_cap_enabled: bool = False
    moltipedia_daily_policy_fix_cap: float = 24.0
    moltipedia_no_self_fix: bool = False

    # Moltbook rate limiting
    moltbook_rate_limit_enabled: bool = False
    moltbook_post_cooldown_steps: int = 5
    moltbook_comment_cooldown_steps: int = 1
    moltbook_daily_comment_cap: int = 50
    moltbook_request_cap_per_step: int = 100

    # Moltbook challenge verification
    moltbook_challenge_enabled: bool = False
    moltbook_challenge_difficulty: float = 0.5
    moltbook_challenge_window_steps: int = 1

    # Memory tier governance
    memory_promotion_gate_enabled: bool = False
    memory_promotion_min_quality: float = 0.5
    memory_promotion_min_verifications: int = 1
    memory_write_rate_limit_enabled: bool = False
    memory_write_rate_limit_per_epoch: int = 20
    memory_cross_verification_enabled: bool = False
    memory_cross_verification_k: int = 2
    memory_provenance_enabled: bool = False
    memory_provenance_revert_penalty: float = 0.1

    # Self-evolution governance (LiveSWE bridge)
    self_evolution_enabled: bool = False
    self_evolution_max_growth_rate: float = 0.1
    self_evolution_max_tools: int = 20
    self_evolution_block_self_mod: bool = True
    self_evolution_divergence_threshold: float = 0.7
    self_evolution_tool_risk_threshold: float = 0.6
    self_evolution_growth_freeze_duration: int = 1

    # Self-modification governance (Two-Gate policy)
    self_modification_enabled: bool = False
    self_modification_max_per_epoch: int = 10  # frequency limit per agent
    self_modification_tau_min_low: float = -0.10
    self_modification_tau_min_medium: float = 0.00
    self_modification_tau_min_high: float = 0.25
    self_modification_k_max_low: float = 20.0
    self_modification_k_max_medium: float = 35.0
    self_modification_k_max_high: float = 50.0
    self_modification_window_days: int = 14  # rolling window for K_max

    # Refinery quality gate (inspired by GasTown's Refinery merge queue)
    refinery_enabled: bool = False
    refinery_p_threshold: float = 0.5  # reject interactions with p below this

    # Loop detector
    loop_detector_enabled: bool = False
    loop_detector_quality_threshold: float = 0.4  # avg p threshold for quality loops
    loop_detector_repetition_threshold: float = 0.6  # proportion threshold for repetition loops
    loop_detector_tool_misuse_threshold: int = 5  # total tool misuse flags for loop
    loop_detector_rework_threshold: int = 10  # total rework count for loop
    loop_detector_penalty_multiplier: float = 1.5  # cost multiplier for loop detection
    loop_detector_replan_threshold: int = 2  # violations before forced replan
    loop_detector_freeze_threshold: int = 4  # violations before circuit breaker
    loop_detector_freeze_duration: int = 1  # epochs to freeze

    # RBAC (Role-Based Access Control)
    rbac_enabled: bool = False
    rbac_violation_penalty: float = 0.5
    rbac_violation_reputation_penalty: float = -0.2
    rbac_high_stakes_actions: list[str] = []
    rbac_security_clearance_required: int = 2
    rbac_high_stakes_penalty_multiplier: float = 2.0
    rbac_role_action_map: dict[str, list[str]] = {}

    # Attestation heartbeat (dead man's switch)
    attestation_heartbeat_enabled: bool = False
    attestation_heartbeat_interval: int = 5  # steps between required attestations
    attestation_heartbeat_reputation_penalty: float = -0.1  # per missed heartbeat
    attestation_heartbeat_reset_on_epoch: bool = True  # clear lapsed status each epoch

    # Hardware trust rejection handling
    hardware_trust_enabled: bool = False
    hardware_trust_propagation_enabled: bool = True  # Propagate stop token to dependents
    hardware_trust_recovery_max_steps: int = 10  # Max steps in constrained recovery mode

    # Diversity as Defense (DaD)
    diversity_enabled: bool = False
    diversity_rho_max: float = 0.5  # Correlation cap (Rule 1)
    diversity_entropy_min: float = 0.5  # Minimum Shannon entropy of type mix (Rule 2)
    diversity_adversarial_fraction_min: float = (
        0.0  # Min adversarial probe fraction (Rule 3)
    )
    diversity_disagreement_tau: float = (
        0.7  # Disagreement threshold for audit trigger (Rule 4)
    )
    diversity_error_threshold_p: float = 0.5  # p below this counts as an error
    diversity_correlation_penalty_rate: float = (
        0.05  # Cost when correlation cap violated
    )
    diversity_entropy_penalty_rate: float = 0.05  # Cost when entropy floor violated
    diversity_audit_cost: float = 0.1  # Cost applied when disagreement triggers audit
    diversity_correlation_window: int = (
        50  # Window of recent interactions for correlation
    )
    diversity_consensus_z: float = (
        1.96  # z for the Wilson bound on consensus evidence (Rule 5)
    )
    diversity_min_effective_n: float = (
        2.0  # Effective-N quorum floor: below this, consensus is not evidence
    )

    # Resample protocol (Bhatt et al., 2025 — "Ctrl-Z")
    resample_enabled: bool = False
    resample_suspicion_threshold: float = 0.6  # suspicion >= this triggers resample
    resample_n_samples: int = 3  # additional samples drawn on flag
    resample_trajectory_analysis: bool = False  # use trajectory context for suspicion
    resample_trajectory_window: int = 5  # recent interactions for trend analysis
    resample_incrimination_threshold: float = 2.0  # cumulative evidence to freeze agent
    resample_compute_cost: float = 0.01  # per-sample compute surcharge
    resample_evidence_decay: float = 0.9  # per-epoch evidence decay factor
    resample_evidence_rep_weight: float = 0.5  # reputation penalty per evidence unit
    # Cascade risk lever (artifact chain governance)
    # NOTE: Defaults are provisional — not yet validated by parameter sweep.
    # Run examples/run_cascade_sensitivity.py before relying on these in
    # production scenarios.  See PR #397 review for context.
    cascade_risk_enabled: bool = False
    cascade_risk_threshold: float = 0.5  # Min cascade risk to trigger penalty
    cascade_risk_penalty_scale: float = 1.0  # Cost multiplier
    cascade_risk_reputation_scale: float = 0.5  # Reputation penalty multiplier
    cascade_risk_p_threshold: float = 0.3  # Descendants below this p are "bad"
    cascade_risk_window: int = 200  # Rolling interaction window for DAG analysis

    @model_validator(mode="after")
    def _run_validation(self) -> "GovernanceConfig":
        self._check_values()
        return self

    def _check_values(self) -> None:
        """Validate configuration values."""
        if not 0.0 <= self.transaction_tax_rate <= 1.0:
            raise ValueError("transaction_tax_rate must be in [0, 1]")
        if not 0.0 <= self.transaction_tax_split <= 1.0:
            raise ValueError("transaction_tax_split must be in [0, 1]")
        if self.reputation_decay_rate < 0.0 or self.reputation_decay_rate > 1.0:
            raise ValueError("reputation_decay_rate must be in [0, 1]")
        if self.max_vote_weight_per_agent <= 0:
            raise ValueError("max_vote_weight_per_agent must be positive")
        if self.bandwidth_cap < 0:
            raise ValueError("bandwidth_cap must be non-negative")
        if self.min_stake_to_participate < 0:
            raise ValueError("min_stake_to_participate must be non-negative")
        if not 0.0 <= self.stake_slash_rate <= 1.0:
            raise ValueError("stake_slash_rate must be in [0, 1]")
        if not 0.0 <= self.freeze_threshold_toxicity <= 1.0:
            raise ValueError("freeze_threshold_toxicity must be in [0, 1]")
        if self.freeze_threshold_violations < 1:
            raise ValueError("freeze_threshold_violations must be >= 1")
        if self.freeze_duration_epochs < 1:
            raise ValueError("freeze_duration_epochs must be >= 1")
        if not 0.0 <= self.audit_probability <= 1.0:
            raise ValueError("audit_probability must be in [0, 1]")
        if self.audit_penalty_multiplier < 0:
            raise ValueError("audit_penalty_multiplier must be non-negative")
        if not 0.0 <= self.audit_threshold_p <= 1.0:
            raise ValueError("audit_threshold_p must be in [0, 1]")
        if not 0.0 <= self.certificate_coverage <= 1.0:
            raise ValueError("certificate_coverage must be in [0, 1]")
        if self.certificate_penalty < 0:
            raise ValueError("certificate_penalty must be non-negative")
        if self.collusion_frequency_threshold <= 0:
            raise ValueError("collusion_frequency_threshold must be positive")
        if not 0.0 <= self.collusion_correlation_threshold <= 1.0:
            raise ValueError("collusion_correlation_threshold must be in [0, 1]")
        if self.collusion_min_interactions < 1:
            raise ValueError("collusion_min_interactions must be >= 1")
        if not 0.0 <= self.collusion_score_threshold <= 1.0:
            raise ValueError("collusion_score_threshold must be in [0, 1]")
        if self.collusion_penalty_multiplier < 0:
            raise ValueError("collusion_penalty_multiplier must be non-negative")
        if self.collusion_realtime_rate < 0:
            raise ValueError("collusion_realtime_rate must be non-negative")
        # Security detection validation
        if not 0.0 <= self.security_injection_threshold <= 1.0:
            raise ValueError("security_injection_threshold must be in [0, 1]")
        if not 0.0 <= self.security_manipulation_threshold <= 1.0:
            raise ValueError("security_manipulation_threshold must be in [0, 1]")
        if self.security_laundering_trust_gap < 0:
            raise ValueError("security_laundering_trust_gap must be non-negative")
        if self.security_contagion_velocity <= 0:
            raise ValueError("security_contagion_velocity must be positive")
        if self.security_min_chain_length < 2:
            raise ValueError("security_min_chain_length must be >= 2")
        if self.security_min_interactions < 1:
            raise ValueError("security_min_interactions must be >= 1")
        if not 0.0 <= self.security_penalty_threshold <= 1.0:
            raise ValueError("security_penalty_threshold must be in [0, 1]")
        if not 0.0 <= self.security_quarantine_threshold <= 1.0:
            raise ValueError("security_quarantine_threshold must be in [0, 1]")
        if self.security_penalty_multiplier < 0:
            raise ValueError("security_penalty_multiplier must be non-negative")
        if not 0.0 <= self.security_realtime_threshold <= 1.0:
            raise ValueError("security_realtime_threshold must be in [0, 1]")
        if self.security_realtime_rate < 0:
            raise ValueError("security_realtime_rate must be non-negative")
        # Variance-aware validation
        if self.self_ensemble_samples < 1:
            raise ValueError("self_ensemble_samples must be >= 1")
        if not 0.0 <= self.incoherence_breaker_threshold <= 1.0:
            raise ValueError("incoherence_breaker_threshold must be in [0, 1]")
        if self.decomposition_horizon_threshold < 1:
            raise ValueError("decomposition_horizon_threshold must be >= 1")
        if self.incoherence_friction_rate < 0:
            raise ValueError("incoherence_friction_rate must be non-negative")
        # Adaptive governance validation
        if not 0.0 <= self.adaptive_incoherence_threshold <= 1.0:
            raise ValueError("adaptive_incoherence_threshold must be in [0, 1]")

        # Adaptive controller validation
        if self.adaptive_controller_evidence_window < 1:
            raise ValueError("adaptive_controller_evidence_window must be >= 1")
        if self.adaptive_controller_contemplation_interval < 1:
            raise ValueError(
                "adaptive_controller_contemplation_interval must be >= 1"
            )
        if self.adaptive_controller_min_evidence_epochs < 1:
            raise ValueError(
                "adaptive_controller_min_evidence_epochs must be >= 1"
            )
        if not 0.0 <= self.adaptive_controller_confidence_threshold <= 1.0:
            raise ValueError(
                "adaptive_controller_confidence_threshold must be in [0, 1]"
            )
        if self.adaptive_controller_crystallization_min_epochs < 1:
            raise ValueError(
                "adaptive_controller_crystallization_min_epochs must be >= 1"
            )
        if self.adaptive_controller_max_degradation_tolerance < 0:
            raise ValueError(
                "adaptive_controller_max_degradation_tolerance must be non-negative"
            )
        if self.adaptive_controller_max_active_proposals < 1:
            raise ValueError(
                "adaptive_controller_max_active_proposals must be >= 1"
            )

        # Memory tier governance validation
        if not 0.0 <= self.memory_promotion_min_quality <= 1.0:
            raise ValueError("memory_promotion_min_quality must be in [0, 1]")
        if self.memory_promotion_min_verifications < 0:
            raise ValueError("memory_promotion_min_verifications must be >= 0")
        if self.memory_write_rate_limit_per_epoch < 0:
            raise ValueError("memory_write_rate_limit_per_epoch must be non-negative")
        if self.memory_cross_verification_k < 1:
            raise ValueError("memory_cross_verification_k must be >= 1")
        if self.memory_provenance_revert_penalty < 0:
            raise ValueError("memory_provenance_revert_penalty must be non-negative")
        # Sybil detection validation
        if not 0.0 <= self.sybil_similarity_threshold <= 1.0:
            raise ValueError("sybil_similarity_threshold must be in [0, 1]")
        if self.sybil_penalty_multiplier < 0:
            raise ValueError("sybil_penalty_multiplier must be non-negative")
        if self.sybil_realtime_rate < 0:
            raise ValueError("sybil_realtime_rate must be non-negative")
        if self.sybil_max_cluster_size < 1:
            raise ValueError("sybil_max_cluster_size must be >= 1")
        # Transparency validation
        if self.transparency_bonus_rate < 0:
            raise ValueError("transparency_bonus_rate must be non-negative")
        if not 0.0 <= self.transparency_threshold_p <= 1.0:
            raise ValueError("transparency_threshold_p must be in [0, 1]")
        # Council lever validation
        if not 0.0 <= self.council_lever_review_probability <= 1.0:
            raise ValueError("council_lever_review_probability must be in [0, 1]")
        if self.council_lever_penalty_multiplier < 0:
            raise ValueError("council_lever_penalty_multiplier must be non-negative")
        # Moderator validation
        if not 0.0 <= self.moderator_review_probability <= 1.0:
            raise ValueError("moderator_review_probability must be in [0, 1]")
        if self.moderator_penalty_multiplier < 0:
            raise ValueError("moderator_penalty_multiplier must be non-negative")
        if not 0.0 <= self.moderator_threshold_p <= 1.0:
            raise ValueError("moderator_threshold_p must be in [0, 1]")
        # Moltipedia validation
        if self.moltipedia_pair_cap_max < 1:
            raise ValueError("moltipedia_pair_cap_max must be >= 1")
        if self.moltipedia_page_cooldown_steps < 0:
            raise ValueError("moltipedia_page_cooldown_steps must be >= 0")
        if self.moltipedia_daily_policy_fix_cap < 0:
            raise ValueError("moltipedia_daily_policy_fix_cap must be non-negative")

        # Moltbook validation
        if self.moltbook_post_cooldown_steps < 0:
            raise ValueError("moltbook_post_cooldown_steps must be >= 0")
        if self.moltbook_comment_cooldown_steps < 0:
            raise ValueError("moltbook_comment_cooldown_steps must be >= 0")
        if self.moltbook_daily_comment_cap < 0:
            raise ValueError("moltbook_daily_comment_cap must be >= 0")
        if self.moltbook_request_cap_per_step < 0:
            raise ValueError("moltbook_request_cap_per_step must be >= 0")
        if not 0.0 <= self.moltbook_challenge_difficulty <= 1.0:
            raise ValueError("moltbook_challenge_difficulty must be in [0, 1]")
        if self.moltbook_challenge_window_steps < 0:
            raise ValueError("moltbook_challenge_window_steps must be >= 0")

        # Self-evolution governance validation
        if self.self_evolution_max_growth_rate < 0:
            raise ValueError("self_evolution_max_growth_rate must be non-negative")
        if self.self_evolution_max_tools < 1:
            raise ValueError("self_evolution_max_tools must be >= 1")
        if not 0.0 <= self.self_evolution_divergence_threshold <= 1.0:
            raise ValueError(
                "self_evolution_divergence_threshold must be in [0, 1]"
            )
        if not 0.0 <= self.self_evolution_tool_risk_threshold <= 1.0:
            raise ValueError(
                "self_evolution_tool_risk_threshold must be in [0, 1]"
            )
        if self.self_evolution_growth_freeze_duration < 1:
            raise ValueError(
                "self_evolution_growth_freeze_duration must be >= 1"
            )

        # Self-modification governance validation
        if self.self_modification_max_per_epoch < 1:
            raise ValueError("self_modification_max_per_epoch must be >= 1")
        if self.self_modification_k_max_low < 0:
            raise ValueError("self_modification_k_max_low must be non-negative")
        if self.self_modification_k_max_medium < 0:
            raise ValueError("self_modification_k_max_medium must be non-negative")
        if self.self_modification_k_max_high < 0:
            raise ValueError("self_modification_k_max_high must be non-negative")
        if self.self_modification_window_days < 1:
            raise ValueError("self_modification_window_days must be >= 1")

        # Refinery validation
        if not 0.0 <= self.refinery_p_threshold <= 1.0:
            raise ValueError("refinery_p_threshold must be in [0, 1]")

        # Loop detector validation
        if not 0.0 <= self.loop_detector_quality_threshold <= 1.0:
            raise ValueError("loop_detector_quality_threshold must be in [0, 1]")
        if not 0.0 <= self.loop_detector_repetition_threshold <= 1.0:
            raise ValueError("loop_detector_repetition_threshold must be in [0, 1]")
        if self.loop_detector_tool_misuse_threshold < 0:
            raise ValueError("loop_detector_tool_misuse_threshold must be non-negative")
        if self.loop_detector_rework_threshold < 0:
            raise ValueError("loop_detector_rework_threshold must be non-negative")
        if self.loop_detector_penalty_multiplier < 0:
            raise ValueError("loop_detector_penalty_multiplier must be non-negative")
        if self.loop_detector_replan_threshold < 1:
            raise ValueError("loop_detector_replan_threshold must be >= 1")
        if self.loop_detector_freeze_threshold < 1:
            raise ValueError("loop_detector_freeze_threshold must be >= 1")
        if self.loop_detector_freeze_duration < 1:
            raise ValueError("loop_detector_freeze_duration must be >= 1")

        # RBAC validation
        if self.rbac_violation_penalty < 0:
            raise ValueError("rbac_violation_penalty must be non-negative")
        if self.rbac_security_clearance_required < 0:
            raise ValueError("rbac_security_clearance_required must be non-negative")
        if self.rbac_high_stakes_penalty_multiplier < 0:
            raise ValueError("rbac_high_stakes_penalty_multiplier must be non-negative")

        # Attestation heartbeat validation
        if self.attestation_heartbeat_interval < 1:
            raise ValueError("attestation_heartbeat_interval must be >= 1")

        # Hardware trust validation
        if self.hardware_trust_recovery_max_steps < 1:
            raise ValueError("hardware_trust_recovery_max_steps must be >= 1")

        # Diversity as Defense validation
        if not 0.0 <= self.diversity_rho_max <= 1.0:
            raise ValueError("diversity_rho_max must be in [0, 1]")
        if self.diversity_entropy_min < 0.0:
            raise ValueError("diversity_entropy_min must be non-negative")
        if not 0.0 <= self.diversity_adversarial_fraction_min <= 1.0:
            raise ValueError("diversity_adversarial_fraction_min must be in [0, 1]")
        if not 0.0 <= self.diversity_disagreement_tau <= 1.0:
            raise ValueError("diversity_disagreement_tau must be in [0, 1]")
        if not 0.0 <= self.diversity_error_threshold_p <= 1.0:
            raise ValueError("diversity_error_threshold_p must be in [0, 1]")
        if self.diversity_correlation_penalty_rate < 0:
            raise ValueError("diversity_correlation_penalty_rate must be non-negative")
        if self.diversity_entropy_penalty_rate < 0:
            raise ValueError("diversity_entropy_penalty_rate must be non-negative")
        if self.diversity_audit_cost < 0:
            raise ValueError("diversity_audit_cost must be non-negative")
        if self.diversity_correlation_window < 1:
            raise ValueError("diversity_correlation_window must be >= 1")
        if self.diversity_consensus_z <= 0:
            raise ValueError("diversity_consensus_z must be positive")
        if self.diversity_min_effective_n < 0:
            raise ValueError("diversity_min_effective_n must be non-negative")

        # Resample protocol validation
        if not 0.0 <= self.resample_suspicion_threshold <= 1.0:
            raise ValueError("resample_suspicion_threshold must be in [0, 1]")
        if self.resample_n_samples < 1:
            raise ValueError("resample_n_samples must be >= 1")
        if self.resample_trajectory_window < 2:
            raise ValueError("resample_trajectory_window must be >= 2")
        if self.resample_incrimination_threshold <= 0:
            raise ValueError("resample_incrimination_threshold must be positive")
        if self.resample_compute_cost < 0:
            raise ValueError("resample_compute_cost must be non-negative")
        if not 0.0 <= self.resample_evidence_decay <= 1.0:
            raise ValueError("resample_evidence_decay must be in [0, 1]")
        if self.resample_evidence_rep_weight < 0:
            raise ValueError("resample_evidence_rep_weight must be non-negative")
        # Cascade risk validation
        if not 0.0 <= self.cascade_risk_threshold <= 1.0:
            raise ValueError("cascade_risk_threshold must be in [0, 1]")
        if self.cascade_risk_penalty_scale < 0:
            raise ValueError("cascade_risk_penalty_scale must be non-negative")
        if self.cascade_risk_reputation_scale < 0:
            raise ValueError("cascade_risk_reputation_scale must be non-negative")
        if not 0.0 <= self.cascade_risk_p_threshold <= 1.0:
            raise ValueError("cascade_risk_p_threshold must be in [0, 1]")
        if self.cascade_risk_window < 1:
            raise ValueError("cascade_risk_window must be >= 1")
