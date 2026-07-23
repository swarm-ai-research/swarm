"""SciAgentGym environment lifecycle management for SWARM bridges.

This module keeps SWARM decoupled from SciAgentGym internals by requiring only
factory callables that produce environment objects with a compatible interface.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .config import EnvironmentTopology


@dataclass(frozen=True)
class EnvAllocationKey:
    """Lookup key describing which environment instance should be used."""

    episode_id: str
    agent_id: str | None = None
    task_id: str | None = None


@dataclass
class SciEnvManager:
    """Create, cache, and close SciAgentGym environments for SWARM episodes.

    The manager accepts a factory function so SWARM can integrate with whichever
    SciAgentGym constructor/version is installed.
    """

    env_factory: Callable[..., Any]
    topology: EnvironmentTopology = EnvironmentTopology.SHARED_EPISODE
    _envs: dict[EnvAllocationKey, Any] = field(default_factory=dict)

    def get_or_create(
        self,
        *,
        episode_id: str,
        agent_id: str | None = None,
        task_id: str | None = None,
        **factory_kwargs: Any,
    ) -> Any:
        """Get an existing env for the topology or create one with env_factory."""
        key = self._resolve_key(
            episode_id=episode_id,
            agent_id=agent_id,
            task_id=task_id,
        )
        if key not in self._envs:
            self._envs[key] = self.env_factory(
                episode_id=episode_id,
                agent_id=agent_id,
                task_id=task_id,
                **factory_kwargs,
            )
        return self._envs[key]

    def close_all(self) -> None:
        """Close all managed environments and clear cache."""
        for env in self._envs.values():
            close_fn = getattr(env, "close", None)
            if callable(close_fn):
                close_fn()
        self._envs.clear()

    def _resolve_key(
        self,
        *,
        episode_id: str,
        agent_id: str | None,
        task_id: str | None,
    ) -> EnvAllocationKey:
        if self.topology == EnvironmentTopology.SHARED_EPISODE:
            return EnvAllocationKey(episode_id=episode_id)
        if self.topology == EnvironmentTopology.PER_AGENT:
            return EnvAllocationKey(episode_id=episode_id, agent_id=agent_id)
        return EnvAllocationKey(episode_id=episode_id, task_id=task_id)
