"""Role-Based Access Control (RBAC) governance lever.

Implements Section 3.1.8 "Roles, Obligations, and Access Controls":
agents register roles that map to permitted actions, with high-stakes
actions requiring security certification.
"""

from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Set

from swarm.agents.base import ActionType, Role
from swarm.governance.levers import GovernanceLever, LeverEffect

if TYPE_CHECKING:
    from swarm.env.state import EnvState
    from swarm.governance.config import GovernanceConfig
    from swarm.models.interaction import SoftInteraction  # noqa: F401

# Default mapping from Role -> permitted ActionType values
DEFAULT_ROLE_ACTION_MAP: Dict[str, List[str]] = {
    Role.WORKER.value: [
        ActionType.CLAIM_TASK.value,
        ActionType.SUBMIT_OUTPUT.value,
        ActionType.PROPOSE_INTERACTION.value,
        ActionType.ACCEPT_INTERACTION.value,
        ActionType.REJECT_INTERACTION.value,
        ActionType.WRITE_MEMORY.value,
        ActionType.SEARCH_MEMORY.value,
    ],
    Role.VERIFIER.value: [
        ActionType.VERIFY_OUTPUT.value,
        ActionType.VERIFY_MEMORY.value,
        ActionType.VERIFY_KERNEL.value,
        ActionType.MOLTBOOK_VERIFY.value,
        ActionType.PROPOSE_INTERACTION.value,
        ActionType.ACCEPT_INTERACTION.value,
        ActionType.REJECT_INTERACTION.value,
    ],
    Role.PLANNER.value: [
        ActionType.POST_BOUNTY.value,
        ActionType.ACCEPT_BID.value,
        ActionType.REJECT_BID.value,
        ActionType.CLAIM_TASK.value,
        ActionType.SUBMIT_OUTPUT.value,
        ActionType.VERIFY_OUTPUT.value,
        ActionType.PROPOSE_INTERACTION.value,
        ActionType.ACCEPT_INTERACTION.value,
        ActionType.REJECT_INTERACTION.value,
        ActionType.SPAWN_SUBAGENT.value,
    ],
    Role.POSTER.value: [
        ActionType.POST.value,
        ActionType.REPLY.value,
        ActionType.VOTE.value,
        ActionType.MOLTBOOK_POST.value,
        ActionType.MOLTBOOK_COMMENT.value,
        ActionType.MOLTBOOK_VOTE.value,
    ],
    Role.MODERATOR.value: [
        ActionType.POLICY_FLAG.value,
        ActionType.FILE_OBJECTION.value,
        ActionType.FILE_DISPUTE.value,
        ActionType.VERIFY_OUTPUT.value,
        ActionType.PROPOSE_INTERACTION.value,
        ActionType.ACCEPT_INTERACTION.value,
        ActionType.REJECT_INTERACTION.value,
    ],
}


class RBACLever(GovernanceLever):
    """Governance lever enforcing role-based access control.

    Checks each interaction's ``action_type`` metadata against the
    initiator's registered roles.  Violations incur cost and reputation
    penalties.  High-stakes actions additionally require a minimum
    security clearance level.
    """

    def __init__(self, config: "GovernanceConfig"):
        super().__init__(config)

        # agent_id -> list of role strings
        self._agent_roles: Dict[str, List[str]] = {}
        # agent_id -> clearance level (int)
        self._security_clearances: Dict[str, int] = {}

        # Build the role->actions map (allow config override)
        raw_map = getattr(config, "rbac_role_action_map", {})
        if raw_map:
            self._role_action_map: Dict[str, Set[str]] = {
                role: set(actions) for role, actions in raw_map.items()
            }
        else:
            self._role_action_map = {
                role: set(actions)
                for role, actions in DEFAULT_ROLE_ACTION_MAP.items()
            }

        # Tracking
        self._epoch_violations: Dict[str, int] = defaultdict(int)
        self._lifetime_violations: Dict[str, int] = defaultdict(int)

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "rbac"

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def set_agent_roles(self, agent_roles: Dict[str, List[str]]) -> None:
        """Populate role mappings for agents."""
        self._agent_roles = dict(agent_roles)

    def set_security_clearances(self, clearances: Dict[str, int]) -> None:
        """Populate clearance levels for agents."""
        self._security_clearances = dict(clearances)

    def get_permitted_actions(self, agent_id: str) -> Set[str]:
        """Return union of all role permissions for *agent_id*."""
        roles = self._agent_roles.get(agent_id, [])
        permitted: Set[str] = set()
        for role in roles:
            permitted |= self._role_action_map.get(role, set())
        return permitted

    def get_epoch_violations(self) -> Dict[str, int]:
        return dict(self._epoch_violations)

    def get_lifetime_violations(self) -> Dict[str, int]:
        return dict(self._lifetime_violations)

    # ------------------------------------------------------------------
    # Lever hooks
    # ------------------------------------------------------------------
    def on_epoch_start(
        self,
        state: "EnvState",
        epoch: int,
    ) -> LeverEffect:
        """Clear per-epoch violations; auto-populate roles from state."""
        self._epoch_violations.clear()

        # Auto-populate roles from state agent registry if not set
        if not self._agent_roles and hasattr(state, "agents"):
            for aid, agent_state in state.agents.items():
                if hasattr(agent_state, "role") and agent_state.role:
                    role_val = (
                        agent_state.role.value
                        if hasattr(agent_state.role, "value")
                        else str(agent_state.role)
                    )
                    self._agent_roles[aid] = [role_val]

        return LeverEffect(lever_name=self.name)

    def on_interaction(
        self,
        interaction: "SoftInteraction",
        state: "EnvState",
    ) -> LeverEffect:
        if not self.config.rbac_enabled:
            return LeverEffect(lever_name=self.name)

        metadata = interaction.metadata or {}
        action_type = metadata.get("action_type")
        if action_type is None:
            return LeverEffect(lever_name=self.name)

        agent_id = interaction.initiator
        permitted = self.get_permitted_actions(agent_id)

        # Normalise to string value
        action_str = action_type.value if hasattr(action_type, "value") else str(action_type)

        # Fetch high-stakes actions once for reuse
        high_stakes = getattr(self.config, "rbac_high_stakes_actions", [])

        if action_str in permitted:
            # Permitted — but still check high-stakes clearance
            if action_str in high_stakes:
                clearance = self._security_clearances.get(agent_id, 0)
                required = getattr(
                    self.config, "rbac_security_clearance_required", 2
                )
                if clearance >= required:
                    return LeverEffect(lever_name=self.name)
                # Insufficient clearance for high-stakes action
                return self._apply_violation(
                    agent_id, action_str, high_stakes=True
                )
            return LeverEffect(lever_name=self.name)

        # Not in permitted set — role violation
        is_high_stakes = action_str in high_stakes
        return self._apply_violation(agent_id, action_str, high_stakes=is_high_stakes)

    # ------------------------------------------------------------------
    def _apply_violation(
        self,
        agent_id: str,
        action_str: str,
        *,
        high_stakes: bool = False,
    ) -> LeverEffect:
        self._epoch_violations[agent_id] += 1
        self._lifetime_violations[agent_id] += 1

        penalty = getattr(self.config, "rbac_violation_penalty", 0.5)
        rep_penalty = getattr(
            self.config, "rbac_violation_reputation_penalty", -0.2
        )
        if high_stakes:
            multiplier = getattr(
                self.config, "rbac_high_stakes_penalty_multiplier", 2.0
            )
            penalty *= multiplier
            rep_penalty *= multiplier

        return LeverEffect(
            cost_a=penalty,
            reputation_deltas={agent_id: rep_penalty},
            lever_name=self.name,
            details={
                "violation": True,
                "agent_id": agent_id,
                "action_type": action_str,
                "high_stakes": high_stakes,
                "penalty": penalty,
                "reputation_penalty": rep_penalty,
            },
        )
