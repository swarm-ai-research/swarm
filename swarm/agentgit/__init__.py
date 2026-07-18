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
from swarm.agentgit.coordination import (
    ClaimResult,
    CoordinationBoard,
    LockResult,
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
from swarm.agentgit.review import (
    DEFAULT_REVIEWERS,
    DependencyReviewer,
    Reviewer,
    ReviewerReport,
    ReviewFinding,
    ReviewSynthesis,
    SecurityReviewer,
    TestCoverageReviewer,
    run_review_panel,
    synthesize,
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
    "ClaimResult",
    "CommandRecord",
    "ConditionalRule",
    "CoordinationBoard",
    "LockResult",
    "DEFAULT_REVIEWERS",
    "DelegationChain",
    "DelegationLink",
    "DependencyReviewer",
    "PolicyDecision",
    "PolicyFacts",
    "ReviewFinding",
    "ReviewSynthesis",
    "Reviewer",
    "ReviewerReport",
    "SecurityReviewer",
    "TestCoverageReviewer",
    "append_to_log",
    "attach_note",
    "build_bundle",
    "enforced_allowlist_for_chain",
    "gate_bundle",
    "granted_commands",
    "list_noted_commits",
    "read_log",
    "read_notes",
    "run_review_panel",
    "sign_link",
    "store_bundle",
    "synthesize",
    "verify_bundle",
    "verify_log",
    "write_bundle",
]
