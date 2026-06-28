"""Sanitized, opt-out telemetry for SWARM runs.

Ported (not drop-in) from fabro's ``lib/crates/fabro-telemetry``. Mirrors the
Segment/Sentry-style ``Track`` event shape, the off/errors/all level gate, the
anonymous-id pattern, and CLI-argument sanitization — adapted to SWARM's long
experimental runs.

Privacy posture (acceptance criteria for issue ``y6to.2``):

  - **No PII in payloads.** The actor is a random, locally-persisted anonymous
    UUID (``~/.swarm/.telemetry_id``) — *not* derived from username, hostname,
    or MAC. The context block carries only OS / arch / Python+app version /
    locale. CLI arguments are reduced to a basename plus ``VALUE``-redacted
    operands, and every property dict passes through :func:`scrub_properties`,
    which strips path-like strings and sensitive-looking keys.
  - **Opt-out via env var.** ``SWARM_TELEMETRY=off`` disables emission
    entirely. ``errors`` emits only error events; ``all`` (the default) emits
    everything.
  - **No network egress (v1).** Events are appended best-effort to a local
    JSONL sink the operator fully controls (``SWARM_TELEMETRY_PATH``, default
    ``~/.swarm/telemetry/events.jsonl``). A Segment/Sentry HTTP sender is
    intentionally out of scope here; this layer establishes the sanitized
    event stream a sender would later consume.

All emission is best-effort: a failing sink or a malformed property must never
interrupt or slow a run, so :meth:`Telemetry.track` swallows its own errors.
"""

from __future__ import annotations

import json
import os
import platform
import re
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Env vars (follow the existing SWARM_* convention).
ENV_LEVEL = "SWARM_TELEMETRY"
ENV_SINK_PATH = "SWARM_TELEMETRY_PATH"
ENV_ID_PATH = "SWARM_TELEMETRY_ID_PATH"

_APP_NAME = "swarm"

_NUMERIC_RE = re.compile(r"^\d+(\.\d+)*$")  # ints and dotted numerics (semver)
_BOOLISH = {"true", "false", "yes", "no"}
_REDACTED = "REDACTED"
_VALUE = "VALUE"

# Property keys whose values are redacted outright (defence-in-depth on top of
# the value-level path heuristics below).
_SENSITIVE_KEY_RE = re.compile(
    r"(token|secret|password|passwd|api[_-]?key|auth|email|user(name)?|"
    r"home|cwd|path|dir|file|host(name)?|ip|addr|url|uri)",
    re.IGNORECASE,
)


class TelemetryLevel(Enum):
    """How much to emit. Controlled by ``SWARM_TELEMETRY``."""

    OFF = "off"
    ERRORS = "errors"
    ALL = "all"


def level_from_env(value: Optional[str]) -> TelemetryLevel:
    """Parse a telemetry level. Unset/unknown defaults to ``ALL`` (local sink).

    The default is on-but-local: events go to a JSONL file the operator owns,
    never the network. ``SWARM_TELEMETRY=off`` is the opt-out.
    """
    match (value or "").strip().lower():
        case "off":
            return TelemetryLevel.OFF
        case "errors":
            return TelemetryLevel.ERRORS
        case "all":
            return TelemetryLevel.ALL
        case _:
            return TelemetryLevel.ALL


def should_track(level: TelemetryLevel, is_error: bool) -> bool:
    """Gate one event against the configured level."""
    if level == TelemetryLevel.OFF:
        return False
    if level == TelemetryLevel.ERRORS:
        return is_error
    return True


# Sanitization


def sanitize_command(argv: List[str], subcommand: str = "") -> str:
    """Reduce an argv to a PII-free command string.

    ``argv[0]`` becomes its basename; the ``subcommand`` tokens are kept
    verbatim; flags (``-x`` / ``--flag``) are kept; ``key=value`` redacts the
    value; boolean/numeric/semver operands are kept; everything else (paths,
    free text) becomes ``VALUE``. Mirrors fabro's ``sanitize_command``.
    """
    parts: List[str] = []
    if argv:
        parts.append(Path(argv[0]).name)

    sub_tokens = subcommand.split()
    skip = 1 + len(sub_tokens)
    parts.extend(sub_tokens)

    for arg in argv[skip:]:
        parts.append(_sanitize_arg(arg))
    return " ".join(parts)


def _sanitize_arg(arg: str) -> str:
    if "=" in arg:
        key, _, val = arg.partition("=")
        return f"{key}={_sanitize_scalar(val)}"
    if arg.startswith("-"):
        return arg  # bare flag, no value attached
    return _sanitize_scalar(arg)


def _sanitize_scalar(val: str) -> str:
    lowered = val.lower()
    if lowered in _BOOLISH or _NUMERIC_RE.match(val):
        return val
    return _VALUE


def _looks_like_path(val: str) -> bool:
    return (
        "/" in val
        or "\\" in val
        or val.startswith("~")
        or val.startswith(".")
        and len(val) > 1
    )


# Strings longer than this are treated as opaque blobs and redacted, on the
# assumption that anything that large is free text or serialized payload rather
# than an identifier the dashboard wants to group on.
_BLOB_LEN = 256


def scrub_value(value: Any) -> Any:
    """Recursively strip likely-PII from a property value.

    Numbers and booleans pass through. Strings that look like filesystem
    paths/URLs are redacted to ``VALUE``, as are oversized blobs. Other strings
    are kept — emitters are expected to pre-sanitize command lines and free
    text via :func:`sanitize_command` (this scrub is defence-in-depth, not the
    primary sanitizer, so it must not mangle already-safe multi-token strings).
    Dicts/lists are scrubbed element-wise; sensitive *keys* are redacted by
    :func:`_scrub_keyed`.
    """
    if isinstance(value, bool) or isinstance(value, (int, float)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _scrub_keyed(k, v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub_value(v) for v in value]
    if isinstance(value, str):
        if _looks_like_path(value) or len(value) > _BLOB_LEN:
            return _VALUE
        return value
    return _VALUE


def _scrub_keyed(key: str, value: Any) -> Any:
    if isinstance(value, str) and _SENSITIVE_KEY_RE.search(key):
        return _REDACTED
    return scrub_value(value)


def scrub_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Return a PII-scrubbed copy of an event property dict."""
    return {k: _scrub_keyed(k, v) for k, v in properties.items()}


# Context (no PII)


def build_context() -> Dict[str, Any]:
    """OS / arch / runtime / app / locale — never hostname, user, or paths."""
    return {
        "os": {"name": platform.system() or "unknown"},
        "device": {"type": platform.machine() or "unknown"},
        "runtime": {"python": platform.python_version()},
        "app": {"name": _APP_NAME, "version": _app_version()},
        "locale": _parse_locale(os.environ.get("LANG", "")),
    }


def _app_version() -> str:
    try:
        from importlib.metadata import version

        return version("distributional-agi-safety")
    except Exception:
        return "0.0.0"


def _parse_locale(lang: str) -> str:
    if not lang or lang in ("C", "POSIX"):
        return "en-US"
    return lang.split(".")[0].replace("_", "-")


# Anonymous id (random, persisted; not derived from any machine identifier)


def _id_path() -> Path:
    override = os.environ.get(ENV_ID_PATH)
    if override:
        return Path(override)
    return Path.home() / ".swarm" / ".telemetry_id"


def anonymous_id() -> str:
    """Return a stable, locally-persisted random UUID.

    Generated once and cached on disk; deliberately *not* derived from MAC /
    hostname / username so it carries no machine-identifying information. Falls
    back to an ephemeral UUID if the cache file cannot be read or written.
    """
    path = _id_path()
    try:
        existing = path.read_text().strip()
        if existing:
            return existing
    except OSError:
        pass

    new_id = str(uuid.uuid4())
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(new_id)
    except OSError:
        pass  # ephemeral id this run; still anonymous
    return new_id


# Event


@dataclass
class TelemetryEvent:
    """A Segment-style ``Track`` event."""

    event: str
    anonymous_id: str
    properties: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    message_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anonymousId": self.anonymous_id,
            "event": self.event,
            "properties": self.properties,
            "context": self.context,
            "timestamp": self.timestamp,
            "messageId": self.message_id,
        }


# Sinks


class TelemetrySink(ABC):
    """Destination for telemetry events. Implementations must never raise."""

    @abstractmethod
    def emit(self, event: TelemetryEvent) -> None: ...


class NullSink(TelemetrySink):
    """Discards events (used when telemetry is off)."""

    def emit(self, event: TelemetryEvent) -> None:  # noqa: D401
        return


class FileSink(TelemetrySink):
    """Appends events as JSONL to a local file, best-effort.

    No network egress: the operator owns the file. Errors are swallowed so a
    full disk or permissions issue can never break a run.
    """

    def __init__(self, path: Path | str):
        self.path = Path(path)

    def emit(self, event: TelemetryEvent) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except (OSError, TypeError, ValueError):
            pass  # best-effort; telemetry must not disrupt the run


def _default_sink_path() -> Path:
    override = os.environ.get(ENV_SINK_PATH)
    if override:
        return Path(override)
    return Path.home() / ".swarm" / "telemetry" / "events.jsonl"


# Telemetry facade


class Telemetry:
    """Builds, gates, sanitizes, and dispatches telemetry events."""

    def __init__(
        self,
        *,
        level: Optional[TelemetryLevel] = None,
        sink: Optional[TelemetrySink] = None,
        anon_id: Optional[str] = None,
    ):
        self.level = (
            level if level is not None else level_from_env(os.environ.get(ENV_LEVEL))
        )
        if sink is not None:
            self.sink: TelemetrySink = sink
        elif self.level == TelemetryLevel.OFF:
            self.sink = NullSink()
        else:
            self.sink = FileSink(_default_sink_path())
        self._anon_id = anon_id
        self._context: Optional[Dict[str, Any]] = None

    @property
    def enabled(self) -> bool:
        return self.level != TelemetryLevel.OFF

    def track(
        self,
        event: str,
        properties: Optional[Dict[str, Any]] = None,
        *,
        is_error: bool = False,
    ) -> bool:
        """Emit one sanitized event. Returns True if it was dispatched.

        Best-effort: any failure (sink, serialization, id lookup) is caught so
        telemetry never disrupts the run.
        """
        try:
            if not should_track(self.level, is_error):
                return False
            built = TelemetryEvent(
                event=event,
                anonymous_id=self._get_anon_id(),
                properties=scrub_properties(properties or {}),
                context=self._get_context(),
                timestamp=datetime.now(timezone.utc).isoformat(),
                message_id=str(uuid.uuid4()),
            )
            self.sink.emit(built)
            return True
        except Exception:
            return False

    def _get_anon_id(self) -> str:
        if self._anon_id is None:
            self._anon_id = anonymous_id()
        return self._anon_id

    def _get_context(self) -> Dict[str, Any]:
        if self._context is None:
            self._context = build_context()
        return self._context


# Module-level singleton


_INSTANCE: Optional[Telemetry] = None


def get_telemetry() -> Telemetry:
    """Return the process-wide telemetry instance, creating it on first use."""
    global _INSTANCE
    if _INSTANCE is None:
        _INSTANCE = Telemetry()
    return _INSTANCE


def track(
    event: str,
    properties: Optional[Dict[str, Any]] = None,
    *,
    is_error: bool = False,
) -> bool:
    """Convenience wrapper around :meth:`Telemetry.track` on the singleton."""
    return get_telemetry().track(event, properties, is_error=is_error)


def reset_telemetry(instance: Optional[Telemetry] = None) -> None:
    """Replace (or clear) the singleton. Primarily for tests."""
    global _INSTANCE
    _INSTANCE = instance
