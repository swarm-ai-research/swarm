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
    """Compose ``child``'s declared wards under ``parent``.

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
    if not errors and not parent.can_spawn:
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
    composed = WardSet(
        max_turns=int(_min_opt(parent.max_turns, child.max_turns) or 0) or None,
        max_depth=int(_min_opt(parent_depth - 1, child.max_depth) or 0),
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
