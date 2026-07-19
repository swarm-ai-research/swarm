"""Configuration for the Gitlawb SWARM bridge."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class GitlawbConfig:
    """Configuration for monitoring Gitlawb agent interactions via SWARM metrics."""

    # Gitlawb node endpoints
    node_url: str = "https://node.gitlawb.com"
    ws_url: str = "wss://node.gitlawb.com/graphql/ws"

    # Repos to subscribe to (empty = all)
    repos: list[str] = field(default_factory=list)

    # Identity of the monitoring agent
    watcher_did: str = ""

    # LLM judge for quality scoring
    use_llm_judge: bool = True
    judge_model: str = "gpt-4o-mini"
    judge_endpoint: Optional[str] = None
    judge_api_key_env: str = "OPENAI_API_KEY"
    heuristic_fallback: bool = True

    # Metrics computation interval
    metrics_interval_sec: float = 300.0

    # In-memory interaction ring buffer size
    history_max_size: int = 10_000

    # Optional JSONL persistence path (one interaction per line, append-only)
    persistence_path: Optional[str] = None

    # WebSocket reconnection
    reconnect_delay_sec: float = 5.0
    max_reconnect_attempts: int = 0  # 0 = infinite

    # Logging
    log_level: str = "INFO"

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if not self.ws_url:
            raise ValueError("ws_url must not be empty")
        if not self.node_url:
            raise ValueError("node_url must not be empty")

    @classmethod
    def from_toml(cls, path: str | Path) -> "GitlawbConfig":
        """Load configuration from a TOML file."""
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Try standard nested path [tool.swarm.gitlawb]; if missing, fall back to root.
        # This supports two TOML layouts: (1) standard [tool.swarm.gitlawb] section,
        # or (2) root-level config keys. Fallback is useful for minimal configs
        # that omit the nested structure.
        section = data.get("tool", {}).get("swarm", {}).get("gitlawb", data)
        return cls(**{k: v for k, v in section.items() if hasattr(cls, k)})

    def to_toml(self, path: str | Path) -> None:
        """Serialize current config to a TOML file."""
        import tomli_w

        data = {
            k: v for k, v in self.__dict__.items() if v is not None and v != ""
        }
        with open(path, "wb") as f:
            tomli_w.dump(data, f)
