"""Translate delegated permissions into enforceable sandbox capabilities.

enqe (`swarm.agentgit.identity`) made an agent's authority *verifiable*: a
`DelegationChain` records the signed `human -> org -> agent` grant and which
permissions flow to the agent. But verification alone is still advisory — it
proves what an agent *was allowed* to do without stopping it from doing more.

This module closes that gap for command execution. It maps named permissions
to the concrete command binaries they authorize, and derives the *enforced*
command allowlist from a delegation chain. The contract is deny-by-default:

- An invalid, expired, or over-scoped chain grants **no** commands.
- Only commands authorized by a verified permission are returned.

The worktree sandbox already gates command execution against a per-agent
allowlist (`swarm.bridges.worktree.policy.WorktreePolicy`); feeding it an
allowlist derived here makes the cryptographically delegated capabilities the
ones that are *physically* enforced at dispatch time.

Permission tokens that do not correspond to a command capability (e.g.
``open_pr``, a downstream *action* capability) are simply ignored here — they
authorize nothing at the command layer and are enforced elsewhere.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set, Tuple

from swarm.agentgit.identity import DelegationChain

# Maps a permission token to the command binaries it authorizes inside a
# sandbox. Kept aligned with WorktreePolicy.COMMAND_RISK_MAP / the default
# allowlist; network-reaching commands (ssh/scp, git push|fetch|pull|clone)
# are unconditionally blocked by WorktreePolicy regardless of capability.
CAPABILITY_COMMANDS: Dict[str, Set[str]] = {
    "read": {"ls", "cat", "head", "tail", "wc", "diff", "grep", "find"},
    "write": {"mkdir", "touch", "echo"},
    "test": {"pytest", "python"},
    "lint": {"ruff", "mypy"},
    "package": {"pip"},
    "vcs": {"git"},
}


def known_capabilities() -> List[str]:
    """Permission tokens that authorize commands (sorted)."""

    return sorted(CAPABILITY_COMMANDS)


def granted_commands(permissions: Iterable[str]) -> List[str]:
    """Union of command binaries authorized by ``permissions`` (sorted).

    Unknown tokens (non-command capabilities) contribute nothing.
    """

    commands: Set[str] = set()
    for permission in permissions:
        commands |= CAPABILITY_COMMANDS.get(permission, set())
    return sorted(commands)


def enforced_allowlist_for_chain(
    chain: DelegationChain,
    *,
    expected_subject_did: Optional[str] = None,
    context: Optional[str] = None,
) -> Tuple[List[str], List[str]]:
    """Derive the enforced command allowlist from a delegation chain.

    Returns ``(allowed_commands, errors)``. If the chain fails verification the
    allowlist is empty (deny-by-default) and ``errors`` explains why.
    ``context`` is the task/session the chain is presented in; a chain bound
    to another context grants nothing (zsof).
    """

    ok, errors = chain.verify(expected_subject_did=expected_subject_did, context=context)
    if not ok:
        return [], errors
    return granted_commands(chain.effective_permissions()), []
