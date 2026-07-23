"""Configuration for the Aeon SWARM bridge.

Aeon (https://github.com/aaronjmars/aeon) is an autonomous agent that runs on
GitHub Actions and records its multi-agent activity as append-only JSONL
ledgers under ``memory/agent-first/`` (tasks, proofs, reviews), conforming to
its ``schemas/agent-first/*.schema.json`` records. Unlike the Gitlawb bridge,
which subscribes to a live GraphQL node, this bridge reads those ledger files
from a local Aeon checkout, so it has no network transport and no async deps.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional


@dataclass
class AeonConfig:
    """Configuration for monitoring Aeon agent-first ledgers via SWARM metrics."""

    # Path to the Aeon checkout's agent-first ledger directory
    # (defaults to the conventional location inside an Aeon repo).
    ledger_dir: str = "memory/agent-first"

    # Ledger file names (one JSON record per line).
    tasks_file: str = "tasks.jsonl"
    proofs_file: str = "proofs.jsonl"
    reviews_file: str = "reviews.jsonl"

    # Restrict to specific repos (matches Task.repo). Empty = all.
    repos: list[str] = field(default_factory=list)

    # Identity of the monitoring agent (informational).
    watcher_did: str = ""

    # Optionally pull GitHub Actions skill-run conclusions via the `gh` CLI.
    # Each completed run becomes an agent<->fleet interaction.
    include_skill_runs: bool = False
    skill_runs_repo: str = ""  # owner/name; "" = current `gh` default repo
    skill_runs_limit: int = 100

    # LLM judge for quality scoring (mirrors the Gitlawb bridge).
    use_llm_judge: bool = False
    judge_model: str = "gpt-4o-mini"
    judge_endpoint: Optional[str] = None
    judge_api_key_env: str = "OPENAI_API_KEY"
    heuristic_fallback: bool = True

    # Watch-mode poll interval (seconds) for re-reading the ledgers.
    poll_interval_sec: float = 60.0

    # In-memory interaction ring buffer size.
    history_max_size: int = 10_000

    # Optional JSONL persistence path for emitted SoftInteractions.
    persistence_path: Optional[str] = None

    # Logging.
    log_level: str = "INFO"

    def validate(self) -> None:
        """Raise ValueError if configuration is invalid."""
        if not self.ledger_dir:
            raise ValueError("ledger_dir must not be empty")
        if self.poll_interval_sec <= 0:
            raise ValueError("poll_interval_sec must be positive")
        if self.history_max_size <= 0:
            raise ValueError("history_max_size must be positive")

    @property
    def tasks_path(self) -> Path:
        return Path(self.ledger_dir) / self.tasks_file

    @property
    def proofs_path(self) -> Path:
        return Path(self.ledger_dir) / self.proofs_file

    @property
    def reviews_path(self) -> Path:
        return Path(self.ledger_dir) / self.reviews_file

    @classmethod
    def from_toml(cls, path: str | Path) -> "AeonConfig":
        """Load configuration from a TOML file."""
        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib  # type: ignore[no-redef]

        with open(path, "rb") as f:
            data = tomllib.load(f)

        # Try standard nested path [tool.swarm.aeon]; if missing, fall back to root.
        # This supports two TOML layouts: (1) standard [tool.swarm.aeon] section,
        # or (2) root-level config keys. Fallback is useful for minimal configs
        # that omit the nested structure.
        section = data.get("tool", {}).get("swarm", {}).get("aeon", data)
        # Use the dataclass field names, not hasattr: fields declared with a
        # default_factory (e.g. ``repos``) are NOT class attributes, so a
        # hasattr filter would silently drop them from a TOML config.
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in section.items() if k in known})

    def to_toml(self, path: str | Path) -> None:
        """Serialize current config to a TOML file."""
        import tomli_w

        data = {k: v for k, v in self.__dict__.items() if v is not None and v != ""}
        with open(path, "wb") as f:
            tomli_w.dump(data, f)
