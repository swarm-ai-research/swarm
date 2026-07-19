"""CrewAI-backed SWARM agent adapter.

Runs a CrewAI crew as the policy module inside a SWARM agent.
SWARM remains the environment + governance + metrics + logging layer;
CrewAI provides role-based internal collaboration that produces
an action/utterance.

Each SWARM agent step becomes:
  (observe SWARM state) -> (CrewAI crew deliberates) -> (emit Action to SWARM)

CrewAI is an *optional* dependency.  Import this module only when
``crewai`` is installed (the loader uses lazy importing).

Security notes
--------------
* Crew output is **untrusted**.  All fields are validated and clamped
  before being converted to SWARM ``Action`` objects.
* ``target_id`` and ``counterparty_id`` are validated against the
  current observation's visible IDs.
* Crew-supplied metadata is namespaced under ``"crew_metadata"`` to
  prevent collision with SWARM-internal keys.
* ``crew.kickoff()`` is wrapped in a timeout to prevent DoS.
* Deliberation memory is capped to prevent unbounded growth.
* Observation content from other agents may contain adversarial
  payloads (prompt injection).  This is an **accepted risk** in the
  simulation threat model -- the adapter's job is to contain the
  *output* side, not to sanitize every LLM input.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import math
import random
import re
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field, field_validator

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Limits (security hardening)
# ---------------------------------------------------------------------------

MAX_CONTENT_LENGTH = 10_000
MAX_RATIONALE_LENGTH = 2_000
MAX_ID_LENGTH = 200
MAX_METADATA_KEYS = 20
MAX_METADATA_KEY_LENGTH = 200
MAX_METADATA_VALUE_LENGTH = 1_000
MAX_RAW_TRACE_LENGTH = 2_000
MAX_DELIBERATION_MEMORY = 200
MAX_STAGED_ACTIONS = 10
DEFAULT_CREW_TIMEOUT_SECONDS = 120.0

# ---------------------------------------------------------------------------
# Action schema (Pydantic) that CrewAI outputs must conform to
# ---------------------------------------------------------------------------

# Map from friendly string names to ActionType enum values.
# CrewAI crews produce one of these string keys; the adapter
# resolves it to the real ActionType.
_ACTION_KIND_MAP: Dict[str, ActionType] = {
    "post": ActionType.POST,
    "reply": ActionType.REPLY,
    "vote": ActionType.VOTE,
    "propose_interaction": ActionType.PROPOSE_INTERACTION,
    "accept_interaction": ActionType.ACCEPT_INTERACTION,
    "reject_interaction": ActionType.REJECT_INTERACTION,
    "claim_task": ActionType.CLAIM_TASK,
    "submit_output": ActionType.SUBMIT_OUTPUT,
    "post_bounty": ActionType.POST_BOUNTY,
    "place_bid": ActionType.PLACE_BID,
    "accept_bid": ActionType.ACCEPT_BID,
    "noop": ActionType.NOOP,
}

_VALID_KINDS = sorted(_ACTION_KIND_MAP.keys())


class SwarmActionSchema(BaseModel):
    """Schema that a CrewAI crew must output.

    This is the "action API" boundary between CrewAI and SWARM.
    All string fields are length-clamped for safety.
    """

    kind: str = Field(
        ...,
        description=(
            "Action type. One of: " + ", ".join(_VALID_KINDS)
        ),
    )
    content: str = Field(
        default="",
        max_length=MAX_CONTENT_LENGTH,
        description="Text payload (post content, reply text, output, etc.)",
    )
    target_id: str = Field(
        default="",
        max_length=MAX_ID_LENGTH,
        description="Target entity ID (post_id, task_id, bounty_id, etc.)",
    )
    counterparty_id: str = Field(
        default="",
        max_length=MAX_ID_LENGTH,
        description="Counterparty agent ID for interactions.",
    )
    confidence: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Crew's confidence in this action [0, 1].",
    )
    rationale: str = Field(
        default="",
        max_length=MAX_RATIONALE_LENGTH,
        description="Free-text explanation of why this action was chosen.",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Extra key-value metadata.",
    )

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, v: str) -> str:
        """Reject unknown action kinds at parse time."""
        if v not in _ACTION_KIND_MAP:
            raise ValueError(
                f"Invalid action kind: {v!r}. Valid: {_VALID_KINDS}"
            )
        return v


# ---------------------------------------------------------------------------
# Tool adapter: read-only + write-intent tools exposed to CrewAI
# ---------------------------------------------------------------------------


class CrewAIToolAdapter:
    """Provides SWARM-safe tool functions to CrewAI agents.

    Tools fall into two categories:
      1. **Read-only** -- derived from the current observation.
      2. **Write-intent** -- stage candidate actions that the crew's
         Executor can select among.  SWARM is the one that commits.

    Parameters
    ----------
    observation : Observation | None
        The current SWARM observation.  Updated each tick via
        ``update_observation()``.
    """

    def __init__(self) -> None:
        self._obs: Optional[Observation] = None
        self._staged_actions: List[Dict[str, Any]] = []

    def update_observation(self, obs: Observation) -> None:
        """Refresh the observation for the current tick."""
        self._obs = obs
        self._staged_actions.clear()

    # -- read-only tools --------------------------------------------------

    def get_agent_state(self) -> Dict[str, Any]:
        """Return the agent's own state snapshot."""
        if self._obs is None:
            return {}
        return dict(self._obs.agent_state.to_dict())

    def get_visible_posts(self, limit: int = 10) -> List[Dict]:
        """Return the most recent visible posts."""
        if self._obs is None:
            return []
        return list(self._obs.visible_posts[:limit])

    def get_available_tasks(self) -> List[Dict]:
        """Return tasks available for claiming."""
        if self._obs is None:
            return []
        return list(self._obs.available_tasks)

    def get_pending_proposals(self) -> List[Dict]:
        """Return interaction proposals pending acceptance."""
        if self._obs is None:
            return []
        return list(self._obs.pending_proposals)

    def get_visible_agents(self) -> List[Dict]:
        """Return visible agent summaries."""
        if self._obs is None:
            return []
        return list(self._obs.visible_agents)

    def get_ecosystem_metrics(self) -> Dict:
        """Return ecosystem health metrics."""
        if self._obs is None:
            return {}
        return dict(self._obs.ecosystem_metrics)

    def get_available_bounties(self) -> List[Dict]:
        """Return marketplace bounties open for bidding."""
        if self._obs is None:
            return []
        return list(self._obs.available_bounties)

    # -- write-intent tools (staging) -------------------------------------

    def stage_action(self, action_dict: Dict[str, Any]) -> str:
        """Stage a candidate action for the Executor to pick.

        Capped at ``MAX_STAGED_ACTIONS`` to prevent memory abuse.
        Returns a short confirmation string.
        """
        if len(self._staged_actions) >= MAX_STAGED_ACTIONS:
            return f"rejected (limit {MAX_STAGED_ACTIONS} reached)"
        self._staged_actions.append(action_dict)
        return f"staged ({len(self._staged_actions)} total)"

    def get_staged_actions(self) -> List[Dict[str, Any]]:
        """Return all staged candidate actions."""
        return list(self._staged_actions)


# ---------------------------------------------------------------------------
# Crew configuration
# ---------------------------------------------------------------------------


class CrewProfile(str, Enum):
    """Built-in crew profiles.

    Each profile defines the internal sub-agents that form the crew.
    Custom profiles can be registered via ``register_crew_profile()``.
    """

    MARKET_TEAM = "market_team_v1"
    AUDIT_TEAM = "audit_team_v1"
    GENERAL = "general_v1"


class CrewAgentRole(BaseModel):
    """Configuration for one sub-agent inside the crew."""

    role: str = Field(..., description="Role name (e.g. 'Strategist')")
    goal: str = Field(..., description="Goal description for this sub-agent")
    backstory: str = Field(default="", description="Backstory/context")


class CrewConfig(BaseModel):
    """Configuration for a CrewAI crew that backs a SWARM agent."""

    crew_profile: str = Field(
        default="general_v1",
        description="Crew profile identifier.",
    )
    role_name: str = Field(
        default="General Agent",
        description="High-level role name for the SWARM agent.",
    )
    model: str = Field(
        default="gpt-4o-mini",
        description="LLM model for the CrewAI agents.",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, gt=0)
    verbose: bool = Field(
        default=False,
        description="Enable verbose CrewAI output.",
    )
    enable_trace: bool = Field(
        default=True,
        description="Attach crew deliberation trace to action metadata.",
    )
    timeout: float = Field(
        default=DEFAULT_CREW_TIMEOUT_SECONDS,
        gt=0.0,
        description="Max seconds for a single crew.kickoff() call.",
    )
    sub_agents: List[CrewAgentRole] = Field(
        default_factory=list,
        description=(
            "Custom sub-agent definitions.  If empty, the adapter uses "
            "the built-in defaults for the chosen crew_profile."
        ),
    )


# ---------------------------------------------------------------------------
# Built-in crew profiles
# ---------------------------------------------------------------------------

_DEFAULT_PROFILES: Dict[str, List[CrewAgentRole]] = {
    "market_team_v1": [
        CrewAgentRole(
            role="Perception Summarizer",
            goal="Compress the observation into the key facts the team needs.",
            backstory="Expert at distilling complex state into actionable summaries.",
        ),
        CrewAgentRole(
            role="Strategist",
            goal="Choose the best plan given incentives, budget, and objectives.",
            backstory="Experienced decision-maker who maximizes long-term value.",
        ),
        CrewAgentRole(
            role="Compliance Monitor",
            goal="Check rules, sanctions risk, and governance constraints.",
            backstory="Regulatory expert who prevents policy violations.",
        ),
        CrewAgentRole(
            role="Executor",
            goal="Format the final SwarmAction JSON that satisfies the schema.",
            backstory="Reliable formatter who produces valid, well-structured output.",
        ),
    ],
    "audit_team_v1": [
        CrewAgentRole(
            role="Perception Summarizer",
            goal="Summarize the current ecosystem state and anomalies.",
            backstory="Anomaly detection specialist.",
        ),
        CrewAgentRole(
            role="Risk Assessor",
            goal="Identify which agents or interactions pose the highest risk.",
            backstory="Seasoned auditor who can spot irregularities.",
        ),
        CrewAgentRole(
            role="Executor",
            goal="Produce the final audit/action JSON for SWARM.",
            backstory="Reliable formatter for audit reports.",
        ),
    ],
    "general_v1": [
        CrewAgentRole(
            role="Analyst",
            goal="Analyze the observation and recommend an action.",
            backstory="Generalist analyst who balances multiple objectives.",
        ),
        CrewAgentRole(
            role="Executor",
            goal="Format the recommended action as valid SwarmAction JSON.",
            backstory="Reliable formatter who produces valid output.",
        ),
    ],
}

# Names that cannot be overwritten by register_crew_profile().
_BUILTIN_PROFILE_NAMES: frozenset = frozenset(_DEFAULT_PROFILES.keys())


def get_profile_agents(profile_name: str) -> List[CrewAgentRole]:
    """Get the sub-agent definitions for a crew profile."""
    if profile_name in _DEFAULT_PROFILES:
        return list(_DEFAULT_PROFILES[profile_name])
    raise ValueError(
        f"Unknown crew profile: {profile_name!r}. "
        f"Available: {sorted(_DEFAULT_PROFILES.keys())}"
    )


def register_crew_profile(name: str, agents: List[CrewAgentRole]) -> None:
    """Register a custom crew profile.

    Raises ``ValueError`` if *name* collides with a built-in profile.
    """
    if name in _BUILTIN_PROFILE_NAMES:
        raise ValueError(
            f"Cannot overwrite built-in profile: {name!r}. "
            f"Built-in profiles: {sorted(_BUILTIN_PROFILE_NAMES)}"
        )
    _DEFAULT_PROFILES[name] = list(agents)


# ---------------------------------------------------------------------------
# Crew factory: build a CrewAI Crew from config
# ---------------------------------------------------------------------------


def build_crew(
    config: CrewConfig,
    tool_adapter: CrewAIToolAdapter,
):
    """Build a CrewAI ``Crew`` instance from *config*.

    Returns
    -------
    crew : crewai.Crew
        A ready-to-kickoff crew.

    Raises
    ------
    ImportError
        If ``crewai`` is not installed.
    """
    try:
        from crewai import Agent as CrewAgent
        from crewai import Crew, Task
    except ImportError as exc:
        raise ImportError(
            "CrewAI adapter requires the 'crewai' package.  "
            "Install it with:  pip install crewai"
        ) from exc

    # Resolve sub-agent definitions
    sub_agent_defs = config.sub_agents or get_profile_agents(config.crew_profile)

    # Build CrewAI Tool wrappers from the tool adapter
    crewai_tools = _build_crewai_tools(tool_adapter)

    # Create CrewAI agents
    crew_agents: list = []
    for agent_def in sub_agent_defs:
        agent = CrewAgent(
            role=agent_def.role,
            goal=agent_def.goal,
            backstory=agent_def.backstory,
            verbose=config.verbose,
            tools=crewai_tools,
        )
        crew_agents.append(agent)

    # Create a single sequential pipeline task:
    # each sub-agent contributes to producing the final SwarmAction.
    tasks: list = []
    for i, (agent_def, crew_agent) in enumerate(
        zip(sub_agent_defs, crew_agents, strict=True)
    ):
        is_last = i == len(sub_agent_defs) - 1
        description = (
            f"As the {agent_def.role}, {agent_def.goal}  "
            "Use the context provided to contribute to the final decision."
        )
        expected_output = (
            "A valid JSON object matching the SwarmActionSchema."
            if is_last
            else f"A summary/recommendation from the {agent_def.role} perspective."
        )
        task = Task(
            description=description,
            expected_output=expected_output,
            agent=crew_agent,
        )
        tasks.append(task)

    crew = Crew(
        agents=crew_agents,
        tasks=tasks,
        verbose=config.verbose,
    )
    return crew


def _make_crewai_tool(
    tool_name: str,
    description: str,
    tool_adapter: CrewAIToolAdapter,
    method_name: str,
) -> Any:
    """Factory function to create a CrewAI tool class.

    Args:
        tool_name: Snake_case name of the tool (e.g., "get_agent_state")
        description: Human-readable description
        tool_adapter: The CrewAIToolAdapter instance
        method_name: Name of the adapter method to call (e.g., "get_agent_state")
    """
    from crewai.tools import BaseTool as CrewBaseTool

    class DynamicCrewAITool(CrewBaseTool):
        name: str = tool_name
        description: str = description

        def _run(self) -> str:
            method = getattr(tool_adapter, method_name)
            return json.dumps(method())

    return DynamicCrewAITool()


def _build_crewai_tools(tool_adapter: CrewAIToolAdapter) -> list:
    """Create CrewAI-compatible tool wrappers from the tool adapter.

    If ``crewai`` provides a ``Tool`` class we use it; otherwise we
    return plain callables (some CrewAI versions accept both).
    """
    try:
        # Define tool specifications as (name, description, adapter_method_name)
        tool_specs = [
            ("get_agent_state", "Get the agent's own state (reputation, resources, etc.)", "get_agent_state"),
            ("get_visible_posts", "Get recent visible posts in the feed", "get_visible_posts"),
            ("get_available_tasks", "Get tasks available for claiming", "get_available_tasks"),
            ("get_pending_proposals", "Get interaction proposals pending acceptance", "get_pending_proposals"),
            ("get_visible_agents", "Get visible agent summaries", "get_visible_agents"),
            ("get_ecosystem_metrics", "Get ecosystem health metrics", "get_ecosystem_metrics"),
            ("get_available_bounties", "Get marketplace bounties open for bidding", "get_available_bounties"),
        ]

        return [
            _make_crewai_tool(name, desc, tool_adapter, method)
            for name, desc, method in tool_specs
        ]
    except ImportError:
        # Fallback: return empty tools list if crewai.tools is unavailable
        logger.warning("crewai.tools not available; crew will run without tools")
        return []


# ---------------------------------------------------------------------------
# Metadata sanitization
# ---------------------------------------------------------------------------


def _sanitize_key(key: str) -> str:
    """Sanitize a metadata key: truncate and strip non-printable chars."""
    # Strip non-printable/control characters (keep printable ASCII + unicode)
    cleaned = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", key)
    return cleaned[:MAX_METADATA_KEY_LENGTH]


def _sanitize_crew_metadata(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize crew-supplied metadata (parity with the Swarms adapter).

    * Limits number of keys to ``MAX_METADATA_KEYS``.
    * Sanitizes key names (truncation, control char stripping).
    * Only allows str, int, float, bool values.
    * Rejects non-finite floats (NaN, Inf).
    * Truncates string values to ``MAX_METADATA_VALUE_LENGTH``.
    """
    sanitized: Dict[str, Any] = {}
    for key, value in list(raw.items())[:MAX_METADATA_KEYS]:
        clean_key = _sanitize_key(str(key))
        if not clean_key:
            continue
        if isinstance(value, str):
            sanitized[clean_key] = value[:MAX_METADATA_VALUE_LENGTH]
        elif isinstance(value, bool):
            # Check bool before int/float since bool is subclass of int
            sanitized[clean_key] = value
        elif isinstance(value, (int, float)):
            if isinstance(value, float) and not math.isfinite(value):
                continue  # Drop NaN, Inf, -Inf
            sanitized[clean_key] = value
        # Drop complex/nested values silently
    return sanitized


# ---------------------------------------------------------------------------
# CrewBackedAgent: the SWARM agent adapter
# ---------------------------------------------------------------------------


class CrewBackedAgent(BaseAgent):
    """A SWARM agent whose policy is driven by a CrewAI crew.

    The adapter follows a strict separation of concerns:

    * **SWARM** owns the environment, governance, payoffs, and logging.
    * **CrewAI** owns the internal deliberation that produces each action.

    The ``act()`` method builds a context dict from the SWARM observation,
    passes it to the crew, parses the result into a SWARM ``Action``,
    and optionally attaches the deliberation trace.

    Security model
    ~~~~~~~~~~~~~~
    Crew output is untrusted.  The adapter validates and clamps all
    fields before converting them to ``Action`` objects:

    * ``agent_id`` is always pinned to ``self.agent_id`` (never from crew).
    * ``target_id`` / ``counterparty_id`` are checked against the current
      observation's visible IDs; unknown IDs are rejected to empty string.
    * Crew-supplied ``metadata`` is namespaced under ``"crew_metadata"``
      to prevent collision with SWARM-internal keys.
    * ``crew.kickoff()`` is wrapped in a configurable timeout.
    * Deliberation memory is capped at ``MAX_DELIBERATION_MEMORY``.

    Parameters
    ----------
    agent_id : str
        Unique SWARM agent identifier.
    crew_config : CrewConfig
        Configuration for the backing CrewAI crew.
    roles : list[Role] | None
        SWARM roles this agent can fulfil.
    config : dict | None
        Extra agent-level config (passed to ``BaseAgent``).
    name : str | None
        Human-readable label.
    rng : random.Random | None
        Seeded RNG for deterministic fallback behaviour.
    """

    def __init__(
        self,
        agent_id: str,
        crew_config: Optional[CrewConfig] = None,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.CREWAI,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )
        self.crew_config = crew_config or CrewConfig()
        self.tool_adapter = CrewAIToolAdapter()
        self._crew = None  # lazy-built on first act()
        self._deliberation_memory: List[Dict[str, Any]] = []
        # Populated each tick from the observation for ID validation.
        self._valid_target_ids: Set[str] = set()
        self._valid_counterparty_ids: Set[str] = set()
        # Reuse a single thread pool across ticks to avoid per-call overhead.
        self._executor_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None

    # -- lazy crew construction -------------------------------------------

    def _ensure_crew(self):
        """Build the CrewAI crew on first use (lazy import)."""
        if self._crew is None:
            self._crew = build_crew(self.crew_config, self.tool_adapter)
        return self._crew

    # -- SWARM BaseAgent interface ----------------------------------------

    def act(self, observation: Observation) -> Action:
        """Produce a SWARM Action by running the CrewAI crew.

        1. Build a context dict from the observation.
        2. Pass context to the crew (with timeout).
        3. Parse the crew's output into ``SwarmActionSchema``.
        4. Validate IDs against the observation.
        5. Convert to a SWARM ``Action``.
        6. Optionally attach trace to ``action.metadata``.
        """
        self.tool_adapter.update_observation(observation)
        self._collect_valid_ids(observation)

        context = self._build_context(observation)

        try:
            crew = self._ensure_crew()
            crew_result = self._kickoff_with_timeout(crew, context)
            raw_output = self._extract_crew_output(crew_result)
            action_schema = self._parse_action(raw_output)
        except concurrent.futures.TimeoutError:
            logger.warning(
                "CrewAI crew timed out after %.1fs for %s; falling back to noop",
                self.crew_config.timeout,
                self.agent_id,
            )
            return self.create_noop_action()
        except Exception:
            logger.exception(
                "CrewAI deliberation failed for %s; falling back to noop",
                self.agent_id,
            )
            return self.create_noop_action()

        action = self._schema_to_action(action_schema)

        # Attach trace
        if self.crew_config.enable_trace:
            raw_str = (
                raw_output if isinstance(raw_output, str) else str(raw_output)
            )
            trace = {
                "crew_profile": self.crew_config.crew_profile,
                "role_name": self.crew_config.role_name,
                "confidence": action_schema.confidence,
                "rationale": action_schema.rationale[:MAX_RATIONALE_LENGTH],
                "raw_output": raw_str[:MAX_RAW_TRACE_LENGTH],
            }
            action.metadata["_crew_trace"] = trace

        # Record to deliberation memory (capped)
        self._deliberation_memory.append(
            {
                "epoch": observation.current_epoch,
                "step": observation.current_step,
                "action_kind": action_schema.kind,
                "confidence": action_schema.confidence,
                "rationale": action_schema.rationale[:500],
            }
        )
        if len(self._deliberation_memory) > MAX_DELIBERATION_MEMORY:
            self._deliberation_memory = self._deliberation_memory[
                -MAX_DELIBERATION_MEMORY:
            ]

        return action

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Decide whether to accept a proposed interaction.

        Uses counterparty trust as a simple heuristic; a full CrewAI
        deliberation on every proposal would be too expensive.
        """
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        threshold = self.config.get("acceptance_threshold", 0.4)
        return bool(trust >= threshold)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose an interaction to a counterparty.

        Delegates to a simple trust-based heuristic.
        """
        trust = self.compute_counterparty_trust(counterparty_id)
        if trust < 0.3:
            return None
        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="CrewAI agent proposing collaboration.",
        )

    # -- timeout wrapper --------------------------------------------------

    def _get_executor_pool(self) -> concurrent.futures.ThreadPoolExecutor:
        """Return (and lazily create) a reusable single-thread pool."""
        if self._executor_pool is None or getattr(
            self._executor_pool, "_shutdown", False
        ):
            self._executor_pool = concurrent.futures.ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix=f"crewai-{self.agent_id}",
            )
        return self._executor_pool

    def _kickoff_with_timeout(self, crew: Any, context: Dict[str, Any]) -> Any:
        """Run ``crew.kickoff()`` with a timeout.

        Uses ``shutdown(wait=False, cancel_futures=True)`` on timeout so
        the simulation is not blocked waiting for the abandoned HTTP call
        to complete.  A fresh pool is created on the next tick.

        Raises
        ------
        concurrent.futures.TimeoutError
            If the crew exceeds ``self.crew_config.timeout``.
        """
        pool = self._get_executor_pool()
        future = pool.submit(crew.kickoff, inputs=context)
        try:
            return future.result(timeout=self.crew_config.timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            pool.shutdown(wait=False, cancel_futures=True)
            self._executor_pool = None  # force fresh pool next tick
            raise

    # -- observation ID collection ----------------------------------------

    def _collect_valid_ids(self, obs: Observation) -> None:
        """Build the sets of valid target and counterparty IDs."""
        self._valid_target_ids = set()
        self._valid_counterparty_ids = set()

        for post in obs.visible_posts:
            pid = post.get("post_id", "")
            if pid:
                self._valid_target_ids.add(pid)
        for task in obs.available_tasks:
            tid = task.get("task_id", "")
            if tid:
                self._valid_target_ids.add(tid)
        for task in obs.active_tasks:
            tid = task.get("task_id", "")
            if tid:
                self._valid_target_ids.add(tid)
        for proposal in obs.pending_proposals:
            pid = proposal.get("proposal_id", "")
            if pid:
                self._valid_target_ids.add(pid)
        for bounty in obs.available_bounties:
            bid = bounty.get("bounty_id", "")
            if bid:
                self._valid_target_ids.add(bid)
        for bid_decision in obs.pending_bid_decisions:
            for key in ("bounty_id", "bid_id"):
                val = bid_decision.get(key, "")
                if val:
                    self._valid_target_ids.add(val)

        for agent in obs.visible_agents:
            aid = agent.get("agent_id", "")
            if aid:
                self._valid_counterparty_ids.add(aid)

    # -- context building -------------------------------------------------

    def _build_context(self, obs: Observation) -> Dict[str, Any]:
        """Build the context dict passed to the CrewAI crew.

        .. warning::

           Observation content (posts, proposals, etc.) originates from
           other agents and may contain adversarial prompt-injection
           payloads.  This is an accepted risk in the simulation threat
           model.  The adapter hardens the *output* side instead.
        """
        return {
            "epoch": obs.current_epoch,
            "step": obs.current_step,
            "agent_id": self.agent_id,
            "role_name": self.crew_config.role_name,
            "budget": obs.agent_state.resources,
            "reputation": obs.agent_state.reputation,
            "can_post": obs.can_post,
            "can_interact": obs.can_interact,
            "can_vote": obs.can_vote,
            "can_claim_task": obs.can_claim_task,
            "pending_proposals": list(obs.pending_proposals[:5]),
            "visible_posts": list(obs.visible_posts[:10]),
            "available_tasks": list(obs.available_tasks[:5]),
            "visible_agents": list(obs.visible_agents[:10]),
            "ecosystem_metrics": dict(obs.ecosystem_metrics),
            "available_bounties": list(obs.available_bounties[:5]),
            "recent_deliberations": self._deliberation_memory[-5:],
            "action_schema": SwarmActionSchema.model_json_schema(),
            "valid_action_kinds": _VALID_KINDS,
        }

    # -- output parsing ---------------------------------------------------

    def _extract_crew_output(self, crew_result: Any) -> Any:
        """Extract usable output from the CrewAI result object."""
        # CrewAI returns different types depending on version
        if hasattr(crew_result, "raw"):
            return crew_result.raw
        if isinstance(crew_result, str):
            return crew_result
        if isinstance(crew_result, dict):
            return crew_result
        return str(crew_result)

    def _parse_action(self, raw: Any) -> SwarmActionSchema:
        """Parse raw crew output into a SwarmActionSchema.

        Tries JSON parsing first, then falls back to extracting
        JSON from a markdown code block.
        """
        text = raw if isinstance(raw, str) else json.dumps(raw)

        # Try direct JSON parse
        try:
            data = json.loads(text)
            return SwarmActionSchema(**data)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

        # Try extracting JSON from ```json ... ``` block
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return SwarmActionSchema(**data)
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        # Last resort: noop with rationale
        logger.warning(
            "Could not parse crew output for %s; defaulting to noop. "
            "Raw output: %.200s",
            self.agent_id,
            text,
        )
        return SwarmActionSchema(
            kind="noop",
            rationale=f"Parse failure. Raw: {text[:200]}",
        )

    def _schema_to_action(self, schema: SwarmActionSchema) -> Action:
        """Convert a parsed SwarmActionSchema to a SWARM Action.

        Validates ``target_id`` and ``counterparty_id`` against the
        current observation.  Unknown IDs are rejected to empty string.
        Crew-supplied metadata is namespaced under ``"crew_metadata"``.
        """
        action_type = _ACTION_KIND_MAP.get(schema.kind, ActionType.NOOP)

        # Validate IDs against the observation whitelist
        target_id = (
            schema.target_id
            if schema.target_id in self._valid_target_ids
            else ""
        )
        counterparty_id = (
            schema.counterparty_id
            if schema.counterparty_id in self._valid_counterparty_ids
            else ""
        )

        if schema.target_id and not target_id:
            logger.debug(
                "Rejected crew target_id %r (not in observation) for %s",
                schema.target_id,
                self.agent_id,
            )
        if schema.counterparty_id and not counterparty_id:
            logger.debug(
                "Rejected crew counterparty_id %r (not in observation) for %s",
                schema.counterparty_id,
                self.agent_id,
            )

        return Action(
            action_type=action_type,
            agent_id=self.agent_id,
            content=schema.content,
            target_id=target_id,
            counterparty_id=counterparty_id,
            metadata={
                "crew_metadata": _sanitize_crew_metadata(schema.metadata),
                "confidence": schema.confidence,
                "rationale": schema.rationale[:MAX_RATIONALE_LENGTH],
            },
        )

    # -- deliberation trace access ----------------------------------------

    def get_deliberation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent deliberation trace entries."""
        return self._deliberation_memory[-limit:]

    def __repr__(self) -> str:
        return (
            f"CrewBackedAgent(id={self.agent_id}, "
            f"crew={self.crew_config.crew_profile})"
        )
