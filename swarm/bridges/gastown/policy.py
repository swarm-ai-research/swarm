"""GasTownPolicy — map SWARM governance decisions to ``gt`` CLI actions.

Translates SWARM governance levers (circuit breaker, audit, staking,
transaction tax) into concrete GasTown operations executed via the
``gt`` CLI.
"""

import logging
import subprocess
from typing import List, Optional

from swarm.bridges.claude_code.policy import PolicyDecision, PolicyResult
from swarm.governance.config import GovernanceConfig

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT = 10  # seconds


class GasTownPolicy:
    """Maps SWARM governance config to GasTown ``gt`` CLI actions.

    Governance mapping:

    =================== ========================= ================================
    SWARM Lever         GasTown Action            Implementation
    =================== ========================= ================================
    Transaction tax     Budget deduction          Track per-agent budgets, warn
    Circuit breaker     Agent suspension           ``gt stop <agent>``
    Random audit        Witness review assignment  ``gt sling <bead> --to witness``
    Staking requirement Bead hold                  Flag bead for approval
    =================== ========================= ================================
    """

    def __init__(
        self,
        config: Optional[GovernanceConfig] = None,
        gt_cli_path: str = "gt",
    ) -> None:
        self._config = config or GovernanceConfig()
        self._gt_cli_path = gt_cli_path

    # --- Evaluation ---

    def evaluate_bead_action(
        self,
        agent_id: str,
        bead_id: str,
        reputation: float,
    ) -> PolicyResult:
        """Decide whether *agent_id* should work on *bead_id*.

        Checks:
        1. Circuit breaker — deny if reputation below freeze threshold.
        2. Staking — require stake for low-reputation agents.
        3. Transaction tax — attach governance cost.
        """
        # Circuit breaker
        if self._config.circuit_breaker_enabled and reputation < -0.5:
            return PolicyResult(
                decision=PolicyDecision.DENY,
                reason=(
                    f"Circuit breaker: agent reputation {reputation:.2f} "
                    f"below threshold"
                ),
            )

        # Staking requirement
        if self._config.staking_enabled and reputation < 0.0:
            return PolicyResult(
                decision=PolicyDecision.REQUIRE_STAKE,
                reason="Low-reputation agent requires stake",
                stake_required=self._config.min_stake_to_participate,
            )

        # Transaction tax
        cost = self._config.transaction_tax_rate

        return PolicyResult(
            decision=PolicyDecision.APPROVE,
            reason="Bead action approved",
            cost=cost,
        )

    # --- Enforcement actions ---

    def execute_circuit_breaker(self, agent_name: str) -> bool:
        """Suspend an agent via ``gt stop <agent>``."""
        result = self._run_gt(["stop", agent_name])
        if result and result.returncode == 0:
            logger.info("Circuit breaker: stopped agent %s", agent_name)
            return True
        logger.warning("Circuit breaker: failed to stop agent %s", agent_name)
        return False

    def assign_audit(self, bead_id: str) -> bool:
        """Assign a bead for witness review via ``gt sling``."""
        result = self._run_gt(["sling", bead_id, "--to", "witness"])
        if result and result.returncode == 0:
            logger.info("Audit: assigned bead %s to witness", bead_id)
            return True
        logger.warning("Audit: failed to assign bead %s to witness", bead_id)
        return False

    # --- Internal ---

    def _run_gt(
        self, args: List[str]
    ) -> Optional[subprocess.CompletedProcess[str]]:
        """Run a ``gt`` CLI command with timeout."""
        try:
            return subprocess.run(
                [self._gt_cli_path, *args],
                capture_output=True,
                text=True,
                timeout=_SUBPROCESS_TIMEOUT,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.warning("gt %s failed: %s", " ".join(args), exc)
            return None
