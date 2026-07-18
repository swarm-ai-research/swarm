"""Agent-first git provenance primitives.

The MVP in this package keeps git as the storage substrate while adding
task-scoped policy checks and signed provenance bundles for agent-authored
changes.
"""

from swarm.agentgit.bundle import (
    CommandRecord,
    build_bundle,
    verify_bundle,
    write_bundle,
)
from swarm.agentgit.capabilities import (
    CAPABILITY_COMMANDS,
    enforced_allowlist_for_chain,
    granted_commands,
)
from swarm.agentgit.identity import (
    AgentIdentity,
    AgentKeypair,
    DelegationChain,
    DelegationLink,
    sign_link,
)
from swarm.agentgit.policy import (
    AgentGitPolicy,
    ConditionalRule,
    PolicyDecision,
    PolicyFacts,
    gate_bundle,
)
from swarm.agentgit.store import (
    append_to_log,
    attach_note,
    list_noted_commits,
    read_log,
    read_notes,
    store_bundle,
    verify_log,
)

__all__ = [
    "CAPABILITY_COMMANDS",
    "AgentGitPolicy",
    "AgentIdentity",
    "AgentKeypair",
    "CommandRecord",
    "ConditionalRule",
    "DelegationChain",
    "DelegationLink",
    "PolicyDecision",
    "PolicyFacts",
    "append_to_log",
    "attach_note",
    "build_bundle",
    "enforced_allowlist_for_chain",
    "gate_bundle",
    "granted_commands",
    "list_noted_commits",
    "read_log",
    "read_notes",
    "sign_link",
    "store_bundle",
    "verify_bundle",
    "verify_log",
    "write_bundle",
]
