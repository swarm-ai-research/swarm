"""Concordia Entity as a SWARM agent.

Wraps a Concordia v2.x Entity so it participates directly in the
SWARM orchestrator loop.  Natural-language reasoning is scored by
ProxyComputer, testing whether governance mechanisms hold against
emergent LLM behavior rather than scripted patterns.
"""

import logging
import os
import re
import time
from typing import Callable, Dict, Mapping, Optional

import numpy as np

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.bridges.concordia.config import ConcordiaEntityConfig
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType, SoftInteraction

logger = logging.getLogger(__name__)

# Type alias: (system_prompt, user_prompt) -> response_text
LLMCallable = Callable[[str, str], str]

# Provider → (env var for API key, default base URL)
_PROVIDER_DEFAULTS: Dict[str, tuple] = {
    "groq": ("GROQ_API_KEY", "https://api.groq.com/openai/v1"),
    "openrouter": ("OPENROUTER_API_KEY", "https://openrouter.ai/api/v1"),
    "openai": ("OPENAI_API_KEY", "https://api.openai.com/v1"),
    "anthropic": ("ANTHROPIC_API_KEY", None),  # Uses native SDK, not OpenAI compat
    "ollama": (None, "http://localhost:11434/v1"),
    "together": ("TOGETHER_API_KEY", "https://api.together.xyz/v1"),
    "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/v1"),
    "google": ("GOOGLE_API_KEY", None),  # Uses native SDK
}


def _build_llm_fn(config: ConcordiaEntityConfig) -> Optional[LLMCallable]:
    """Build an LLM callable from config.

    Returns ``(system_prompt, user_prompt) -> response_text``,
    or *None* in any of these cases:
      - provider is ``"none"`` (canned-response mode).
      - provider is unknown/unrecognized.
      - required API key is missing.
      - required package cannot be imported (openai, anthropic, google-generativeai).

    Supports:
      - **OpenAI-compatible** (groq, openrouter, openai, ollama, together,
        deepseek) via the ``openai`` package.
      - **Anthropic** via the ``anthropic`` package (native messages API).
      - **Google** via the ``google-generativeai`` package.
      - **Custom** — pass ``llm_fn`` directly to ``ConcordiaEntityAgent``.
    """
    if config.provider == "none":
        return None

    provider_info = _PROVIDER_DEFAULTS.get(config.provider)
    if provider_info is None:
        logger.warning("Unknown provider %r — falling back to canned responses", config.provider)
        return None

    env_var, default_base_url = provider_info
    api_key = config.api_key or (os.environ.get(env_var) if env_var else "ollama")

    if not api_key and config.provider not in ("ollama",):
        logger.warning(
            "No API key for provider %r (set %s). Falling back to canned responses.",
            config.provider, env_var,
        )
        return None

    # ---- Anthropic native SDK ----
    if config.provider == "anthropic":
        return _build_anthropic_fn(config, api_key)

    # ---- Google native SDK ----
    if config.provider == "google":
        return _build_google_fn(config, api_key)

    # ---- OpenAI-compatible providers ----
    return _build_openai_compat_fn(config, api_key, default_base_url)


def _build_openai_compat_fn(
    config: ConcordiaEntityConfig,
    api_key: Optional[str],
    default_base_url: str,
) -> Optional[LLMCallable]:
    """OpenAI-compatible chat completions (Groq, OpenRouter, Together, etc.)."""
    try:
        from openai import OpenAI
    except ImportError:
        logger.warning("openai package required for provider %r. pip install openai", config.provider)
        return None

    base_url = config.base_url or default_base_url
    client = OpenAI(api_key=api_key or "ollama", base_url=base_url, timeout=config.timeout)

    def _call(system_prompt: str, user_prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1 + config.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                )
                return resp.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc
                if attempt < config.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    logger.debug("Retry %d/%d for %s: %s", attempt + 1, config.max_retries, config.provider, exc)
        logger.error("LLM call failed after %d attempts (%s): %s", 1 + config.max_retries, config.provider, last_err)
        return ""

    return _call


def _build_anthropic_fn(
    config: ConcordiaEntityConfig,
    api_key: Optional[str],
) -> Optional[LLMCallable]:
    """Anthropic Messages API."""
    try:
        from anthropic import Anthropic
    except ImportError:
        logger.warning("anthropic package required. pip install anthropic")
        return None

    client = Anthropic(api_key=api_key, timeout=config.timeout)

    def _call(system_prompt: str, user_prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(1 + config.max_retries):
            try:
                resp = client.messages.create(
                    model=config.model,
                    max_tokens=config.max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=config.temperature,
                )
                block = resp.content[0] if resp.content else None
                return block.text if block is not None and hasattr(block, "text") else ""
            except Exception as exc:
                last_err = exc
                if attempt < config.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
        logger.error("Anthropic call failed after %d attempts: %s", 1 + config.max_retries, last_err)
        return ""

    return _call


def _build_google_fn(
    config: ConcordiaEntityConfig,
    api_key: Optional[str],
) -> Optional[LLMCallable]:
    """Google Generative AI (Gemini)."""
    try:
        import google.generativeai as genai
    except ImportError:
        logger.warning("google-generativeai package required. pip install google-generativeai")
        return None

    genai.configure(api_key=api_key)

    def _call(system_prompt: str, user_prompt: str) -> str:
        last_err: Optional[Exception] = None
        m = genai.GenerativeModel(config.model, system_instruction=system_prompt)
        for attempt in range(1 + config.max_retries):
            try:
                resp = m.generate_content(
                    user_prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=config.temperature,
                        max_output_tokens=config.max_tokens,
                    ),
                )
                return resp.text or ""
            except Exception as exc:
                last_err = exc
                if attempt < config.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
        logger.error("Google AI call failed after %d attempts: %s", 1 + config.max_retries, last_err)
        return ""

    return _call


# ---------------------------------------------------------------------------
# Concordia integration (guarded import)
# ---------------------------------------------------------------------------

try:
    from concordia.agents.entity_agent import EntityAgent as _ConcordiaEntity
    from concordia.associative_memory.basic_associative_memory import (
        AssociativeMemoryBank,
    )
    from concordia.typing.entity import ActionSpec, OutputType
    from concordia.typing.entity_component import ActingComponent

    class _SwarmActComponent(ActingComponent):
        """ActingComponent backed by a configurable LLM or canned fallback.

        When *llm_fn* is provided every ``get_action_attempt`` routes
        through the LLM.  Otherwise falls back to ``_fallback``.
        """

        def __init__(
            self,
            agent_name: str = "",
            goal: str = "",
            llm_fn: Optional[LLMCallable] = None,
            system_prompt_template: str = "",
            observation_memory_depth: int = 10,
            fallback: str = "noop",
        ) -> None:
            self._agent_name = agent_name
            self._goal = goal
            self._llm_fn = llm_fn
            self._system_template = system_prompt_template
            self._obs_depth = observation_memory_depth
            self._fallback = fallback
            self._next_response: str = ""
            self._state: Dict[str, str] = {}
            self._observations: list[str] = []

        def set_next_response(self, response: str) -> None:
            """Pre-set a canned response (used by tests and no-LLM mode)."""
            self._next_response = response

        def add_observation(self, observation: str) -> None:
            """Record an observation for LLM context."""
            self._observations.append(observation)
            # Keep bounded to 2× depth so we never allocate unbounded
            cap = self._obs_depth * 2
            if len(self._observations) > cap:
                self._observations = self._observations[-cap:]

        def _render_system_prompt(self) -> str:
            recent = self._observations[-self._obs_depth:]
            obs_text = "\n".join(f"- {o}" for o in recent) if recent else "(none yet)"
            return self._system_template.format(
                name=self._agent_name,
                goal=self._goal,
                observations=obs_text,
            )

        def get_action_attempt(
            self,
            context: Mapping[str, str],
            action_spec: ActionSpec,
        ) -> str:
            if self._llm_fn is not None:
                return self._call_llm(action_spec)
            return self._next_response or self._fallback

        def _call_llm(self, action_spec: ActionSpec) -> str:
            system = self._render_system_prompt()
            user = action_spec.call_to_action
            if action_spec.options:
                user += f"\n\nYou MUST respond with exactly one of: {', '.join(action_spec.options)}"
            try:
                result = self._llm_fn(system, user)  # type: ignore[misc]
                return result if result else self._fallback
            except Exception:
                logger.exception("LLM call failed for %s", self._agent_name)
                return self._next_response or self._fallback

        def get_state(self) -> Mapping[str, object]:
            return dict(self._state)

        def set_state(self, state: Mapping[str, object]) -> None:
            self._state = dict(state)  # type: ignore[arg-type]

    _HAS_CONCORDIA = True
except ImportError:
    _HAS_CONCORDIA = False

# ---------------------------------------------------------------------------
# Standalone helpers (no Concordia dependency)
# ---------------------------------------------------------------------------

# Section size defaults (chars) when rendering observations
_SECTION_DEFAULTS = {
    "posts": 800,
    "agents": 400,
    "tasks": 400,
    "proposals": 400,
    "interactions": 400,
    "ecosystem": 300,
}


def render_observation(obs: Observation, config: ConcordiaEntityConfig) -> str:
    """Convert a structured Observation into a natural-language situation prompt.

    Bounded by *config.max_prompt_chars*; individual sections are capped
    so that one giant list cannot starve the rest.
    """
    sections: list[str] = []
    budget = config.max_prompt_chars

    def _add(header: str, body: str, cap: int) -> None:
        nonlocal budget
        if not body or budget <= 0:
            return
        text = f"## {header}\n{body[:cap]}"
        sections.append(text)
        budget -= len(text)

    # Agent's own state
    state = obs.agent_state
    own = (
        f"You are agent {state.agent_id or '(unknown)'}. "
        f"Reputation: {state.reputation:.2f}. "
        f"Epoch {obs.current_epoch}, step {obs.current_step}."
    )
    _add("Your status", own, 300)

    # Visible posts
    if obs.visible_posts:
        posts_text = "\n".join(
            f"- [{p.get('post_id', '?')}] by {p.get('author', '?')}: "
            f"{str(p.get('content', ''))[:120]}"
            for p in obs.visible_posts[: config.max_visible_posts]
        )
        _add("Recent posts", posts_text, _SECTION_DEFAULTS["posts"])

    # Visible agents
    if obs.visible_agents:
        agents_text = "\n".join(
            f"- {a.get('agent_id', '?')} (rep {a.get('reputation', '?')})"
            for a in obs.visible_agents[: config.max_visible_agents]
        )
        _add("Other agents", agents_text, _SECTION_DEFAULTS["agents"])

    # Available tasks
    if obs.available_tasks:
        tasks_text = "\n".join(
            f"- [{t.get('task_id', '?')}] {str(t.get('description', ''))[:100]}"
            for t in obs.available_tasks[: config.max_available_tasks]
        )
        _add("Available tasks", tasks_text, _SECTION_DEFAULTS["tasks"])

    # Pending proposals
    if obs.pending_proposals:
        props_text = "\n".join(
            f"- from {p.get('initiator_id', '?')}: {str(p.get('content', ''))[:100]}"
            for p in obs.pending_proposals
        )
        _add("Pending proposals", props_text, _SECTION_DEFAULTS["proposals"])

    # Recent interactions
    if obs.recent_interactions:
        inter_text = "\n".join(
            f"- with {i.get('counterparty', '?')}: p={i.get('p', '?')}, "
            f"payoff={i.get('payoff', '?')}"
            for i in obs.recent_interactions[:5]
        )
        _add("Recent interactions", inter_text, _SECTION_DEFAULTS["interactions"])

    # Ecosystem metrics
    if obs.ecosystem_metrics:
        eco_text = ", ".join(
            f"{k}={v}" for k, v in list(obs.ecosystem_metrics.items())[:6]
        )
        _add("Ecosystem", eco_text, _SECTION_DEFAULTS["ecosystem"])

    # Capabilities
    caps = []
    if obs.can_post:
        caps.append("post")
    if obs.can_interact:
        caps.append("interact")
    if obs.can_vote:
        caps.append("vote")
    if obs.can_claim_task and obs.available_tasks:
        caps.append("claim_task")
    if caps:
        _add("You can", ", ".join(caps), 200)

    result = "\n\n".join(sections)
    return result[: config.max_prompt_chars]


# Regex patterns for response parsing
_ACTION_PATTERNS: Dict[str, ActionType] = {
    r"\bpost\b": ActionType.POST,
    r"\bvote\b": ActionType.VOTE,
    r"\bpropose\b": ActionType.PROPOSE_INTERACTION,
    r"\bclaim[_ ]?task\b": ActionType.CLAIM_TASK,
    r"\bnoop\b": ActionType.NOOP,
    r"\bdo nothing\b": ActionType.NOOP,
    r"\bpass\b": ActionType.NOOP,
}


def parse_action_response(response: str, observation: Observation) -> Action:
    """Parse a free-text LLM response into a SWARM Action.

    Performs keyword + regex extraction for POST/VOTE/PROPOSE/CLAIM_TASK/NOOP.
    Falls back to NOOP on parse failure.
    """
    text = response.strip().lower()

    # Try each pattern
    for pattern, action_type in _ACTION_PATTERNS.items():
        if re.search(pattern, text, re.IGNORECASE):
            if action_type == ActionType.POST:
                content_match = re.search(r"post[:\s]+(.+)", response, re.IGNORECASE | re.DOTALL)
                content = content_match.group(1).strip() if content_match else response.strip()
                return Action(
                    action_type=ActionType.POST,
                    content=content[:500],
                )

            if action_type == ActionType.VOTE:
                target = ""
                direction = 1
                target_match = re.search(r"vote\s+(?:on\s+)?(\S+)", text)
                if target_match:
                    target = target_match.group(1)
                if "down" in text:
                    direction = -1
                return Action(
                    action_type=ActionType.VOTE,
                    target_id=target,
                    vote_direction=direction,
                )

            if action_type == ActionType.PROPOSE_INTERACTION:
                cp_match = re.search(r"propose\s+(?:to\s+)?(\S+)", text)
                counterparty = cp_match.group(1) if cp_match else ""
                content_match = re.search(
                    r"propose[^:]*:\s*(.+)", response, re.IGNORECASE | re.DOTALL
                )
                content = content_match.group(1).strip() if content_match else ""
                return Action(
                    action_type=ActionType.PROPOSE_INTERACTION,
                    counterparty_id=counterparty,
                    content=content[:500],
                    interaction_type=InteractionType.COLLABORATION,
                )

            if action_type == ActionType.CLAIM_TASK:
                task_match = re.search(r"claim[_ ]?task\s+(\S+)", text)
                task_id = task_match.group(1) if task_match else ""
                if not task_id and observation.available_tasks:
                    task_id = observation.available_tasks[0].get("task_id", "")
                return Action(
                    action_type=ActionType.CLAIM_TASK,
                    target_id=task_id,
                )

            # NOOP
            return Action(action_type=ActionType.NOOP)

    # Fallback: NOOP
    logger.debug("Could not parse action from response, falling back to NOOP: %s", text[:100])
    return Action(action_type=ActionType.NOOP)


# ---------------------------------------------------------------------------
# ConcordiaEntityAgent
# ---------------------------------------------------------------------------


class ConcordiaEntityAgent(BaseAgent):
    """SWARM agent wrapping a Concordia v2.x Entity.

    The entity's natural-language reasoning drives actions; SWARM's
    ProxyComputer scores those actions probabilistically.

    Dual memory:
      - Concordia's AssociativeMemoryBank for entity reasoning
      - SWARM's ``_memory`` for trust/decay bookkeeping

    LLM backends (set via ``ConcordiaEntityConfig.provider``):
      groq, openrouter, openai, anthropic, google, ollama, together,
      deepseek, or "none" for canned responses.

    You can also pass a pre-built ``llm_fn(system, user) -> str``
    to bypass provider resolution entirely.
    """

    def __init__(
        self,
        agent_id: str,
        concordia_config: Optional[ConcordiaEntityConfig] = None,
        roles: Optional[list[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        entity: object = None,
        llm_fn: Optional[LLMCallable] = None,
    ):
        if not _HAS_CONCORDIA:
            raise ImportError(
                "gdm-concordia is required for ConcordiaEntityAgent. "
                "Install with: pip install gdm-concordia"
            )

        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,  # Actual behavior from LLM
            roles=roles,
            config=config,
            name=name,
        )

        self.concordia_config = concordia_config or ConcordiaEntityConfig()
        self._entity = entity  # Pre-built entity (or None for lazy build)
        self._custom_llm_fn = llm_fn  # User-supplied callable overrides provider
        self._act_component: Optional[_SwarmActComponent] = None
        self._memory_bank: Optional[AssociativeMemoryBank] = None

    @property
    def entity(self):
        """Lazy-build the Concordia entity on first access."""
        if self._entity is None:
            self._entity = self._build_entity()
        return self._entity

    def _build_entity(self):
        """Build a Concordia entity from config."""
        cfg = self.concordia_config

        # Create associative memory
        if cfg.use_concordia_memory:
            self._memory_bank = AssociativeMemoryBank(
                sentence_embedder=lambda x: np.zeros(64, dtype=np.float32),
            )
            self._memory_bank.add(cfg.goal)

        # Resolve LLM callable: explicit > provider > None
        llm_fn = self._custom_llm_fn or _build_llm_fn(cfg)

        # Create act component
        self._act_component = _SwarmActComponent(
            agent_name=self.name,
            goal=cfg.goal,
            llm_fn=llm_fn,
            system_prompt_template=cfg.system_prompt,
            observation_memory_depth=cfg.observation_memory_depth,
            fallback=cfg.fallback_action,
        )

        # Build entity
        entity = _ConcordiaEntity(
            agent_name=self.name,
            act_component=self._act_component,
        )
        return entity

    def _observe(self, text: str) -> None:
        """Feed an observation to both Concordia entity and act component."""
        try:
            self.entity.observe(text)
        except Exception:
            logger.debug("Entity observe() failed")
        if self._act_component is not None:
            self._act_component.add_observation(text)
        if self._memory_bank is not None:
            try:
                self._memory_bank.add(text)
            except Exception:
                logger.debug("Memory bank add() failed")

    def act(self, observation: Observation) -> Action:
        """Decide on an action via Concordia entity reasoning."""
        cfg = self.concordia_config
        rendered = render_observation(observation, cfg)
        user_prompt = cfg.act_prompt.format(observation=rendered)

        spec = ActionSpec(
            call_to_action=user_prompt,
            output_type=OutputType.FREE,
        )

        # Canned fallback for no-LLM mode
        if self._act_component is not None and self._act_component._llm_fn is None:
            self._act_component.set_next_response(
                f"post: Pursuing my goal - {cfg.goal}"
            )

        try:
            response = self.entity.act(spec)
        except Exception:
            logger.exception("Concordia entity act() failed for %s", self.agent_id)
            return self.create_noop_action()

        action = parse_action_response(response, observation)
        action.agent_id = self.agent_id

        summary = f"{action.action_type.value}: {action.content[:100]}" if action.content else action.action_type.value
        self._observe(f"You decided to: {summary}")

        return action

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Decide whether to accept a proposed interaction via entity reasoning."""
        cfg = self.concordia_config
        user_prompt = cfg.accept_prompt.format(
            initiator=proposal.initiator_id,
            interaction_type=proposal.interaction_type.value,
            content=proposal.content[:200],
            transfer=proposal.offered_transfer,
            reputation=observation.agent_state.reputation,
        )

        spec = ActionSpec(
            call_to_action=user_prompt,
            output_type=OutputType.CHOICE,
            options=("accept", "reject"),
        )

        if self._act_component is not None and self._act_component._llm_fn is None:
            self._act_component.set_next_response("accept")

        try:
            response = self.entity.act(spec)
        except Exception:
            logger.exception("accept_interaction() failed for %s", self.agent_id)
            return False

        accepted = response.strip().lower() == "accept"
        decision = "accepted" if accepted else "rejected"
        self._observe(f"You {decision} a proposal from {proposal.initiator_id}.")

        return bool(accepted)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Propose an interaction via entity reasoning."""
        cfg = self.concordia_config
        trust = self.compute_counterparty_trust(counterparty_id)
        user_prompt = cfg.propose_prompt.format(
            counterparty=counterparty_id,
            trust=trust,
            reputation=observation.agent_state.reputation,
        )

        spec = ActionSpec(
            call_to_action=user_prompt,
            output_type=OutputType.FREE,
        )

        if self._act_component is not None and self._act_component._llm_fn is None:
            self._act_component.set_next_response(
                f"propose to {counterparty_id}: Let's collaborate!"
            )

        try:
            response = self.entity.act(spec)
        except Exception:
            logger.exception("propose_interaction() failed for %s", self.agent_id)
            return None

        text = response.strip().lower()
        if "decline" in text or text == "no" or "pass" in text:
            return None

        action = self.create_propose_action(
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content=response.strip()[:500],
        )

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content=action.content,
        )

    def update_from_outcome(
        self,
        interaction: SoftInteraction,
        payoff: float,
    ) -> None:
        """Update both SWARM bookkeeping and Concordia memory."""
        super().update_from_outcome(interaction, payoff)

        counterparty = (
            interaction.counterparty
            if interaction.initiator == self.agent_id
            else interaction.initiator
        )
        status = "accepted" if interaction.accepted else "rejected"
        narrative = (
            f"Interaction with {counterparty} was {status}. "
            f"Quality score: {interaction.p:.2f}, payoff: {payoff:.2f}."
        )
        self._observe(narrative)
