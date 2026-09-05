"""Wards: structural authority bounds for dispatched agents.

A *ward* is a runtime limit the dispatcher enforces on an agent, not a rule
the agent is asked to respect. The vocabulary and the two load-bearing rules
come from deepfates' Cantrip runtime (docs/research/cantrip-runtime-lessons.md,
section 1; bead distributional-agi-safety-illq):

1. **Bounded by construction.** A ward set without a turn limit, a depth
   limit, and a named done gate is invalid. You cannot dispatch an unbounded
   agent by leaving fields out.
2. **Children only narrow.** When a parent spawns a child, numeric wards take
   the ``min``, boolean wards take the ``or``, permissions take the
   intersection, and the negative spec accumulates. A child that declares
   more than its parent allows is *rejected*, never silently rewritten.

The rig already narrows permission *sets* in ``DelegationChain.verify``
(``swarm.agentgit.identity``). This module adds the numeric and boolean
dimensions and the declaration-time child constraints, and derives its
permission dimension from a verified chain, deny-by-default, the way
``swarm.agentgit.capabilities`` does.

Like ``DelegationChain.verify``, the checks here sit on a trust boundary over
data that may come from a bead comment or a bundle: they return structured
error lists and never raise on bad data. Only programming errors raise.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

from swarm.agentgit.identity import DelegationChain

#: Done gates the rig knows how to verify. ``artifact`` is the artifact-only
#: DONE protocol (commit hash, runs/ path, or artifact= in the closing
#: message, enforced by the ``done_requires_artifact`` trigger).
KNOWN_DONE_GATES: FrozenSet[str] = frozenset({"artifact", "test", "counterexample"})

_NUMERIC_FIELDS: Tuple[str, ...] = ("max_turns", "max_depth", "max_children", "timeout_s")
_BOOLEAN_FIELDS: Tuple[str, ...] = ("require_artifact_on_done",)


@dataclass(frozen=True)
class WardSet:
    """The bounds an agent runs under.

    Runtime wards (enforced on the agent itself):

    - ``max_turns``: dispatch rounds before truncation. Required.
    - ``max_depth``: how many further levels of children may be spawned
      beneath this agent. ``0`` means it may not spawn. Required.
    - ``max_children``: cumulative children this agent may spawn.
    - ``timeout_s``: wall-clock budget.
    - ``done_gate``: the mechanism that terminates the task. Required and
      must be one of :data:`KNOWN_DONE_GATES`.
    - ``require_artifact_on_done``: the DONE message must carry an artifact.
    - ``permissions``: capability tokens (see ``capabilities.py``).
    - ``negative_spec``: bead-specific insufficient outcomes the Auditor
      verifies against (the NEGATIVE SPEC from the dispatch rationale stamp).

    Declaration-time child wards (checked on the *child's declaration* before
    runtime composition; a failing child is rejected, not rewritten):

    - ``child_permission_allowlist``: if set, a child may declare only these.
    - ``child_permission_denylist``: a child may never declare these.
    - ``child_max_turns_ceiling`` / ``child_max_depth_ceiling``: a child must
      declare the corresponding ward at or below the ceiling.
    """

    max_turns: Optional[int] = None
    max_depth: Optional[int] = None
    max_children: Optional[int] = None
    timeout_s: Optional[float] = None
    done_gate: str = "artifact"
    require_artifact_on_done: bool = True
    permissions: FrozenSet[str] = field(default_factory=frozenset)
    negative_spec: Tuple[str, ...] = ()
    child_permission_allowlist: Optional[FrozenSet[str]] = None
    child_permission_denylist: FrozenSet[str] = field(default_factory=frozenset)
    child_max_turns_ceiling: Optional[int] = None
    child_max_depth_ceiling: Optional[int] = None

    # -- validation ---------------------------------------------------------

    def errors(self) -> List[str]:
        """Why this ward set is not bounded. Empty means it is."""

        errors: List[str] = []
        if self.max_turns is None:
            errors.append("max_turns is required: an agent without a turn limit is unbounded")
        elif self.max_turns < 1:
            errors.append(f"max_turns must be >= 1, got {self.max_turns}")
        if self.max_depth is None:
            errors.append("max_depth is required: spawn depth must be bounded")
        elif self.max_depth < 0:
            errors.append(f"max_depth must be >= 0, got {self.max_depth}")
        for name in ("max_children", "timeout_s"):
            value = getattr(self, name)
            if value is not None and value < 0:
                errors.append(f"{name} must be >= 0, got {value}")
        if not self.done_gate:
            errors.append("done_gate is required: a task with no terminator never ends")
        elif self.done_gate not in KNOWN_DONE_GATES:
            errors.append(
                f"done_gate {self.done_gate!r} is not verifiable; "
                f"known gates: {sorted(KNOWN_DONE_GATES)}"
            )
        if self.child_permission_allowlist is not None:
            leaked = sorted(self.child_permission_allowlist - self.permissions)
            if leaked:
                errors.append(
                    f"child_permission_allowlist grants permissions this agent lacks: {leaked}"
                )
        return errors

    @property
    def is_bounded(self) -> bool:
        return not self.errors()

    @property
    def can_spawn(self) -> bool:
        """Whether this agent may spawn any child at all."""

        if self.max_depth is None or self.max_depth < 1:
            return False
        return self.max_children is None or self.max_children > 0

    # -- serialization (for dispatch rationale stamps and bundles) ----------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_turns": self.max_turns,
            "max_depth": self.max_depth,
            "max_children": self.max_children,
            "timeout_s": self.timeout_s,
            "done_gate": self.done_gate,
            "require_artifact_on_done": self.require_artifact_on_done,
            "permissions": sorted(self.permissions),
            "negative_spec": list(self.negative_spec),
            "child_permission_allowlist": (
                None
                if self.child_permission_allowlist is None
                else sorted(self.child_permission_allowlist)
            ),
            "child_permission_denylist": sorted(self.child_permission_denylist),
            "child_max_turns_ceiling": self.child_max_turns_ceiling,
            "child_max_depth_ceiling": self.child_max_depth_ceiling,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WardSet":
        allow = data.get("child_permission_allowlist")
        return cls(
            max_turns=data.get("max_turns"),
            max_depth=data.get("max_depth"),
            max_children=data.get("max_children"),
            timeout_s=data.get("timeout_s"),
            done_gate=data.get("done_gate", "artifact"),
            require_artifact_on_done=bool(data.get("require_artifact_on_done", True)),
            permissions=frozenset(data.get("permissions", ())),
            negative_spec=tuple(data.get("negative_spec", ())),
            child_permission_allowlist=None if allow is None else frozenset(allow),
            child_permission_denylist=frozenset(data.get("child_permission_denylist", ())),
            child_max_turns_ceiling=data.get("child_max_turns_ceiling"),
            child_max_depth_ceiling=data.get("child_max_depth_ceiling"),
        )


def from_chain(chain: DelegationChain, base: Optional[WardSet] = None, **kwargs: Any) -> WardSet:
    """Build a ward set whose permissions come from a *verified* chain.

    Deny-by-default: an invalid, expired, or over-scoped chain contributes no
    permissions. Other fields come from ``base`` (or defaults) overridden by
    ``kwargs``. The result may still be unbounded; check :meth:`WardSet.errors`.
    """

    ok, _ = chain.verify()
    permissions = frozenset(chain.effective_permissions()) if ok else frozenset()
    return replace(base or WardSet(), permissions=permissions, **kwargs)


@dataclass(frozen=True)
class ComposeResult:
    """Outcome of composing a child's declared wards under a parent."""

    wards: Optional[WardSet]
    errors: List[str]

    @property
    def accepted(self) -> bool:
        return self.wards is not None and not self.errors


def _min_opt(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def compose(parent: WardSet, child: WardSet) -> ComposeResult:
    """Compose a spawned ``child``'s declared wards under ``parent``.

    Spawning consumes a level: the parent must be able to spawn and the
    child's depth is capped at ``parent.max_depth - 1``. See
    :func:`_compose` for the shared narrowing rules.
    """

    return _compose(parent, child, spawn=True)


def _compose(parent: WardSet, child: WardSet, *, spawn: bool) -> ComposeResult:
    """Shared narrowing. ``spawn=False`` is a peer running *under* the parent
    (a claimant under a track stamp): no spawn check, no depth consumed.

    Order matters and mirrors Cantrip: the parent must be bounded and able
    to spawn; the child's *declaration* is checked against the parent's
    declaration-time wards (rejected, not rewritten); only then are runtime
    wards composed. Numeric wards narrow with ``min``, booleans with ``or``,
    permissions by intersection, and the negative spec accumulates. The
    child's effective ``max_depth`` is additionally capped at
    ``parent.max_depth - 1`` so depth is consumed by each level.

    The composed set never widens the parent in any dimension (INV-6).
    """

    errors: List[str] = [f"parent: {e}" for e in parent.errors()]
    if spawn and not errors and not parent.can_spawn:
        errors.append(
            f"parent may not spawn children (max_depth={parent.max_depth}, "
            f"max_children={parent.max_children})"
        )
    errors.extend(f"child: {e}" for e in child.errors())
    if errors:
        return ComposeResult(None, errors)

    # Declaration-time child wards: the child must declare within them.
    if parent.child_permission_allowlist is not None:
        extra = sorted(child.permissions - parent.child_permission_allowlist)
        if extra:
            errors.append(f"child declares permissions outside allowlist: {extra}")
    denied = sorted(child.permissions & parent.child_permission_denylist)
    if denied:
        errors.append(f"child declares denylisted permissions: {denied}")
    widened = sorted(child.permissions - parent.permissions)
    if widened:
        errors.append(f"child declares permissions the parent lacks: {widened}")
    if (
        parent.child_max_turns_ceiling is not None
        and child.max_turns is not None
        and child.max_turns > parent.child_max_turns_ceiling
    ):
        errors.append(
            f"child max_turns {child.max_turns} exceeds ceiling "
            f"{parent.child_max_turns_ceiling}"
        )
    if (
        parent.child_max_depth_ceiling is not None
        and child.max_depth is not None
        and child.max_depth > parent.child_max_depth_ceiling
    ):
        errors.append(
            f"child max_depth {child.max_depth} exceeds ceiling "
            f"{parent.child_max_depth_ceiling}"
        )
    if errors:
        return ComposeResult(None, errors)

    # Runtime composition. Depth is consumed by the level being spawned.
    parent_depth = parent.max_depth if parent.max_depth is not None else 0
    depth_cap = parent_depth - 1 if spawn else parent_depth
    composed = WardSet(
        max_turns=int(_min_opt(parent.max_turns, child.max_turns) or 0) or None,
        max_depth=int(_min_opt(depth_cap, child.max_depth) or 0),
        max_children=(
            None
            if parent.max_children is None and child.max_children is None
            else int(_min_opt(parent.max_children, child.max_children) or 0)
        ),
        timeout_s=_min_opt(parent.timeout_s, child.timeout_s),
        done_gate=child.done_gate,
        require_artifact_on_done=(
            parent.require_artifact_on_done or child.require_artifact_on_done
        ),
        permissions=parent.permissions & child.permissions,
        negative_spec=parent.negative_spec
        + tuple(s for s in child.negative_spec if s not in parent.negative_spec),
        # Declaration-time wards flow down and can only tighten.
        child_permission_allowlist=_narrow_allowlist(
            parent.child_permission_allowlist, child.child_permission_allowlist
        ),
        child_permission_denylist=parent.child_permission_denylist
        | child.child_permission_denylist,
        child_max_turns_ceiling=_int_or_none(
            _min_opt(parent.child_max_turns_ceiling, child.child_max_turns_ceiling)
        ),
        child_max_depth_ceiling=_int_or_none(
            _min_opt(parent.child_max_depth_ceiling, child.child_max_depth_ceiling)
        ),
    )
    # A composed allowlist may not grant what the composed agent lacks.
    if composed.child_permission_allowlist is not None:
        composed = replace(
            composed,
            child_permission_allowlist=composed.child_permission_allowlist
            & composed.permissions,
        )
    return ComposeResult(composed, composed.errors())


def _narrow_allowlist(
    a: Optional[FrozenSet[str]], b: Optional[FrozenSet[str]]
) -> Optional[FrozenSet[str]]:
    if a is None:
        return b
    if b is None:
        return a
    return a & b


def _int_or_none(value: Optional[float]) -> Optional[int]:
    return None if value is None else int(value)


def never_widens(parent: WardSet, child: WardSet) -> List[str]:
    """Dimensions in which ``child`` exceeds ``parent``. Empty means narrower-or-equal.

    This is the INV-6 predicate. It is deliberately independent of
    :func:`compose` so a test can check compose's output against it rather
    than trusting compose to grade itself.
    """

    out: List[str] = []
    for name in _NUMERIC_FIELDS:
        p, c = getattr(parent, name), getattr(child, name)
        if p is not None and (c is None or c > p):
            out.append(f"{name}: {c} widens {p}")
    for name in _BOOLEAN_FIELDS:
        if getattr(parent, name) and not getattr(child, name):
            out.append(f"{name}: parent requires it, child drops it")
    if not child.permissions <= parent.permissions:
        out.append(f"permissions: {sorted(child.permissions - parent.permissions)}")
    if not set(parent.negative_spec) <= set(child.negative_spec):
        out.append("negative_spec: child dropped a parent entry")
    if not parent.child_permission_denylist <= child.child_permission_denylist:
        out.append("child_permission_denylist: child dropped a parent entry")
    if parent.child_permission_allowlist is not None and (
        child.child_permission_allowlist is None
        or not child.child_permission_allowlist <= parent.child_permission_allowlist
    ):
        out.append("child_permission_allowlist: child widens parent")
    for name in ("child_max_turns_ceiling", "child_max_depth_ceiling"):
        p, c = getattr(parent, name), getattr(child, name)
        if p is not None and (c is None or c > p):
            out.append(f"{name}: {c} widens {p}")
    return out


def compose_chain(root: WardSet, declarations: Sequence[WardSet]) -> ComposeResult:
    """Compose a lineage root -> d1 -> d2 -> ... ; stop at the first rejection."""

    current = root
    for index, declared in enumerate(declarations):
        result = compose(current, declared)
        if not result.accepted or result.wards is None:
            return ComposeResult(None, [f"level {index + 1}: {e}" for e in result.errors])
        current = result.wards
    return ComposeResult(current, [])


# ---------------------------------------------------------------------------
# Dispatch stamps and the claim-time gate (bead illq.2)
# ---------------------------------------------------------------------------
#
# The dispatcher (`/bv-dispatch` full mode) stamps each shortlisted bead with an
# assignee-free rationale comment. Until now the NEGATIVE SPEC and gate class in
# that stamp were prose the Auditor and retro had to grep. A stamp now carries
# one machine-readable line, ``WARDS: {json}``, holding the track's WardSet.
# ``/claim`` reads the latest such line and composes it with whatever the
# claimant declares; a declaration that would widen the track's bounds is
# refused before the claim is taken.

import json as _json  # noqa: E402  (kept local to this section)
import os as _os  # noqa: E402
import subprocess as _subprocess  # noqa: E402
from pathlib import Path as _Path  # noqa: E402
from typing import Iterable  # noqa: E402

STAMP_PREFIX = "WARDS:"
DECLARED_WARDS_FILE = ".agentgit/wards.json"
DECLARED_WARDS_ENV = "AGENT_WARDS"

#: Gate class (bv-dispatch step 7) -> default runtime wards. Search aggression
#: scales inversely with gate cost: a mechanical gate (G0) can afford many
#: turns and children because a wrong answer dies cheaply; a judgment gate
#: (G2) gets one conservative agent and no children.
GATE_CLASS_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "G0": {"max_turns": 12, "max_depth": 1, "max_children": 4},
    "G1": {"max_turns": 8, "max_depth": 1, "max_children": 2},
    "G2": {"max_turns": 4, "max_depth": 0, "max_children": 0},
}


def for_gate_class(
    gate_class: str,
    *,
    negative_spec: Sequence[str] = (),
    permissions: Iterable[str] = (),
    **overrides: Any,
) -> WardSet:
    """A bounded track WardSet from a dispatch gate class plus the negative spec."""

    key = gate_class.upper()
    if key not in GATE_CLASS_DEFAULTS:
        raise ValueError(f"unknown gate class {gate_class!r}; expected one of {sorted(GATE_CLASS_DEFAULTS)}")
    fields: Dict[str, Any] = dict(GATE_CLASS_DEFAULTS[key])
    fields.update(overrides)
    return WardSet(
        permissions=frozenset(permissions),
        negative_spec=tuple(negative_spec),
        **fields,
    )


def format_stamp(wards: WardSet) -> str:
    """The single ``WARDS: {...}`` line that goes into a rationale comment."""

    return f"{STAMP_PREFIX} {_json.dumps(wards.to_dict(), sort_keys=True, separators=(',', ':'))}"


def parse_stamp(text: str) -> Optional[WardSet]:
    """The last well-formed ``WARDS:`` line in ``text``, or None.

    Malformed lines are skipped, not raised: comments are untrusted input.
    """

    found: Optional[WardSet] = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith(STAMP_PREFIX):
            continue
        payload = stripped[len(STAMP_PREFIX):].strip()
        try:
            data = _json.loads(payload)
        except _json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            try:
                found = WardSet.from_dict(data)
            except (TypeError, ValueError):
                continue
    return found


def latest_stamp(comments: Iterable[str]) -> Optional[WardSet]:
    """Latest parsable stamp across comments given oldest-first."""

    found: Optional[WardSet] = None
    for text in comments:
        parsed = parse_stamp(text)
        if parsed is not None:
            found = parsed
    return found


def load_bead_wards(task_id: str, *, timeout: float = 15) -> Optional[WardSet]:
    """Track wards from the bead's comments via ``bd comments <id> --json``.

    Returns None when bd is unavailable, the bead has no stamp, or output is
    unreadable: the claim gate then falls back to "no track bound declared",
    which is the pre-illq.2 behaviour, so a missing bd never blocks a claim.
    """

    try:
        proc = _subprocess.run(
            ["bd", "comments", task_id, "--json"],
            check=False, capture_output=True, text=True, timeout=timeout,
        )
    except (FileNotFoundError, _subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    try:
        rows = _json.loads(proc.stdout or "[]")
    except _json.JSONDecodeError:
        return None
    if not isinstance(rows, list):
        return None
    rows = sorted(
        (r for r in rows if isinstance(r, dict)),
        key=lambda r: str(r.get("created_at", "")),
    )
    return latest_stamp(str(r.get("text", "")) for r in rows)


def load_declared_wards(repo_root: _Path, override: Optional[str] = None) -> Optional[WardSet]:
    """What the claimant declares it will run under.

    Sources, first hit wins: ``override`` (a JSON string or a path), the
    ``AGENT_WARDS`` env var (same forms), then ``.agentgit/wards.json`` under
    ``repo_root``. None means the claimant declares nothing and inherits the
    track's bounds unchanged.
    """

    candidates = [override, _os.environ.get(DECLARED_WARDS_ENV)]
    for raw in candidates:
        if not raw:
            continue
        text = raw
        path = _Path(raw)
        if path.exists():
            text = path.read_text()
        data = _json.loads(text)
        return WardSet.from_dict(data)
    path = _Path(repo_root) / DECLARED_WARDS_FILE
    if path.exists():
        return WardSet.from_dict(_json.loads(path.read_text()))
    return None


def claim_gate(track: Optional[WardSet], declared: Optional[WardSet]) -> ComposeResult:
    """What a claimant runs under, or why the claim is refused.

    Refusal is ``result.errors`` being non-empty. ``result.wards`` may be None
    with no errors when neither side declared anything (unbounded, the
    pre-illq.2 behaviour), so callers must not use ``accepted`` here.

    - No track stamp: nothing to enforce; the declaration (possibly None)
      stands. This keeps unstamped beads claimable exactly as before.
    - Track stamp, no declaration: the claimant inherits the track wards.
    - Both: ``compose(track, declared)`` — refused if it would widen.
    """

    if track is None:
        return ComposeResult(declared, [])
    if declared is None:
        return ComposeResult(track, track.errors())
    # A claimant is a peer running under the track, not a child spawned by
    # it: no spawn check, no depth consumed, every other dimension narrows.
    return _compose(track, declared, spawn=False)
