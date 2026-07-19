"""Governance policies for the SWARM–AgentVeil bridge.

Implements the write-back gate (C4 uncertain-band dampening, C5 per-pair rate limiting,
C6 negative-attestation quota, global attestation rate limit) plus the admission gate
(tier/allowed check) and a circuit-breaker stub (A1 failure-mode mitigation).

All gates are stateless except for explicit counters managed via `reset_epoch()`.
Failure modes addressed:

- **A1 (Registry unavailable)**: `circuit_breaker_state()` stub with configurable fail_mode.
- **C4 (Attestation amplification loop)**: Uncertain-band suppression when p ∈ [negative_threshold, positive_threshold).
- **C5 (Collusion via per-pair attestations)**: Rate-limit to max 1 attestation per (initiator_did, counterparty_did) ordered pair per epoch.
- **C6 (Dispute flooding via negative attestations)**: Cap negative attestations to max 5 per initiator per epoch.
- **D3 (Circuit breaker conflict / redemption path)**: TODO v2 — implement downgrade-to-warning after N epochs.
- **D6 (Non-deterministic replay)**: Policy is pure stateless except counters; no datetime/random calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from swarm.bridges.agentveil.client import AVPClient
from swarm.bridges.agentveil.config import AgentVeilConfig

logger = logging.getLogger(__name__)


class PolicyDecision(Enum):
    """Policy decision outcomes."""

    ALLOW = "allow"
    DENY = "deny"


@dataclass
class PolicyResult:
    """Result of a policy decision gate.

    Fields:
        decision: ALLOW or DENY.
        reason: Human-readable gate name and rationale (e.g. "uncertain_band",
            "per_pair_rate_limit", "circuit_open_fail_closed").
        policy: Which policy method produced this result.
        metadata: Optional context (e.g. outcome_sign for write-back decisions).
    """

    decision: PolicyDecision
    reason: str = ""
    policy: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


class AVPPolicy:
    """Stateful governance policies for the AgentVeil bridge.

    Tracks per-epoch counters (global attestation count, per-actor negative count,
    per-pair attestation count). Call `reset_epoch()` at epoch boundary to rotate
    counters.
    """

    def __init__(self, config: AgentVeilConfig | None = None) -> None:
        """Initialize policy with configuration.

        Args:
            config: AgentVeilConfig. If None, uses defaults.
        """
        self._config = config or AgentVeilConfig()

        # Circuit breaker state
        self._consecutive_failures: int = 0

        # Per-epoch counters
        self._global_attestation_count: int = 0
        self._per_actor_negative_count: dict[str, int] = {}
        self._per_pair_attestation_epoch: dict[tuple[str, str], int] = {}

    def admission_decision(
        self, *, tier: str, allowed: bool
    ) -> PolicyResult:
        """Admission gate: check tier and allowed flag.

        Implements the admission policy from failure-mode A1/C1. Maps tier to an
        ordinal (newcomer=0, basic=1, trusted=2, elite=3) and denies if:
        - tier ordinal < min_tier ordinal, OR
        - allowed is False

        Args:
            tier: Agent's AVP tier (one of "newcomer", "basic", "trusted", "elite").
            allowed: Whether AVP registry explicitly allowed the agent.

        Returns:
            PolicyResult with ALLOW or DENY.
        """
        min_tier_ordinal = AVPClient.TIER_ORDINALS.get(self._config.min_tier, 1)
        agent_tier_ordinal = AVPClient.TIER_ORDINALS.get(tier, 0)

        if not allowed:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="admission_gate_registry_denied",
                policy="admission_decision",
            )

        if agent_tier_ordinal < min_tier_ordinal:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=f"tier_{tier}_below_min_{self._config.min_tier}",
                policy="admission_decision",
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason=f"tier_{tier}_meets_min",
            policy="admission_decision",
        )

    def writeback_decision(
        self,
        *,
        terminal_p: float,
        initiator_did: str,
        counterparty_did: str,
        current_epoch: int,
    ) -> PolicyResult:
        """Write-back gate: decide whether to submit an AVP attestation.

        Applies gates in order:
        1. **Disabled check**: if writeback_enabled is False, deny.
        2. **C4 (uncertain-band)**: if negative_threshold ≤ p < positive_threshold, deny.
        3. **C5 (per-pair rate limit)**: if this (initiator, counterparty) pair has
           already attested in this epoch, deny.
        4. **C6 (negative-attestation quota)**: if p < negative_threshold, check the
           per-actor negative count; deny if at or above 5.
        5. **Global attestation quota**: check total attestations this epoch; deny if
           at or above max_attestations_per_epoch.

        If all gates pass, increment counters and return ALLOW with outcome_sign
        in metadata.

        Args:
            terminal_p: Final p value from the interaction (∈ [0, 1]).
            initiator_did: DIDs of initiating agent.
            counterparty_did: DIDs of counterparty.
            current_epoch: Current epoch number (used for per-pair tracking).

        Returns:
            PolicyResult with ALLOW or DENY and metadata["outcome_sign"] = +1 or -1
            if ALLOW.
        """
        # Gate 1: Disabled check
        if not self._config.writeback_enabled:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="writeback_disabled",
                policy="writeback_decision",
            )

        # Gate 2: C4 uncertain-band dampening
        pos_threshold = self._config.writeback_positive_threshold
        neg_threshold = self._config.writeback_negative_threshold
        if neg_threshold <= terminal_p < pos_threshold:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="uncertain_band",
                policy="writeback_decision",
            )

        # Gate 3: C5 per-pair rate limit (max 1 attestation per (init, counterparty) per epoch)
        pair_key = (initiator_did, counterparty_did)
        last_epoch = self._per_pair_attestation_epoch.get(pair_key, -1)
        if last_epoch == current_epoch:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="per_pair_rate_limit",
                policy="writeback_decision",
            )

        # Determine outcome sign
        outcome_sign = 1 if terminal_p >= pos_threshold else -1

        # Gate 4: C6 negative-attestation quota (max 5 per initiator per epoch)
        if outcome_sign == -1:
            neg_count = self._per_actor_negative_count.get(initiator_did, 0)
            if neg_count >= 5:
                return PolicyResult(
                    decision=PolicyDecision.DENY,
                    reason="negative_attestation_quota",
                    policy="writeback_decision",
                )

        # Gate 5: Global attestation quota
        if self._global_attestation_count >= self._config.max_attestations_per_epoch:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="global_attestation_quota",
                policy="writeback_decision",
            )

        # All gates passed: increment counters
        self._global_attestation_count += 1
        self._per_pair_attestation_epoch[pair_key] = current_epoch
        if outcome_sign == -1:
            self._per_actor_negative_count[initiator_did] = (
                self._per_actor_negative_count.get(initiator_did, 0) + 1
            )

        return PolicyResult(
            decision=PolicyDecision.ALLOW,
            reason="writeback_gates_passed",
            policy="writeback_decision",
            metadata={"outcome_sign": outcome_sign},
        )

    def circuit_breaker_state(
        self, *, consecutive_failures: int, threshold: int = 5
    ) -> PolicyResult:
        """Circuit breaker stub: fail-open or fail-closed on registry unavailability.

        Implements failure-mode A1 mitigation. When the registry becomes unavailable
        (e.g., N consecutive HTTP failures), the bridge switches behavior based on
        `fail_mode`:
        - `"closed"`: Deny all interactions (fail-safe).
        - `"open"`: Allow with a logged warning (fail-operational).

        **TODO (D3)**: Implement redemption path — after N epochs of lockout,
        automatically downgrade denial to warning, allowing re-entry at newcomer tier
        with elevated scrutiny.

        Args:
            consecutive_failures: Number of consecutive registry call failures.
            threshold: Trip the breaker when consecutive_failures >= threshold.

        Returns:
            PolicyResult with ALLOW or DENY depending on fail_mode.
        """
        if consecutive_failures < threshold:
            return PolicyResult(
                decision=PolicyDecision.ALLOW,
                reason="circuit_healthy",
                policy="circuit_breaker_state",
            )

        # Circuit breaker is tripped
        if self._config.fail_mode == "closed":
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason="circuit_open_fail_closed",
                policy="circuit_breaker_state",
            )
        else:  # "open"
            logger.warning(
                "Circuit breaker open (fail-open mode): allowing interactions "
                "with logged warning. Registry may be unavailable."
            )
            return PolicyResult(
                decision=PolicyDecision.ALLOW,
                reason="circuit_open_fail_open_with_warning",
                policy="circuit_breaker_state",
                metadata={"warning": "registry_unavailable"},
            )

    def reset_epoch(self, *, epoch: int) -> None:
        """Reset per-epoch counters at epoch boundary.

        Call this method between epochs to clear:
        - Global attestation count
        - Per-actor negative-attestation counts
        - Per-pair attestation tracking

        Args:
            epoch: The new epoch number (used for logging/debugging).
        """
        logger.debug(
            f"AVPPolicy.reset_epoch: epoch={epoch}, "
            f"global_count={self._global_attestation_count}, "
            f"per_pair_pairs={len(self._per_pair_attestation_epoch)}"
        )
        self._global_attestation_count = 0
        self._per_actor_negative_count.clear()
        self._per_pair_attestation_epoch.clear()
