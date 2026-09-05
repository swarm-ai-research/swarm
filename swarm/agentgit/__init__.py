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
    NonceRegistry,
    sign_link,
)
from swarm.agentgit.memory import (
    MemoryEntry,
    MemoryStore,
)
from swarm.agentgit.policy import (
    AgentGitPolicy,
    ConditionalRule,
    PolicyDecision,
    PolicyFacts,
    gate_bundle,
)
from swarm.agentgit.push_tokens import (
    PushGrant,
    check_push,
    mint_push_grant,
    verify_grant,
)
from swarm.agentgit.reputation import (
    ReputationKey,
    ReputationLedger,
    TrackRecord,
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
from swarm.agentgit.wards import (
    GATE_CLASS_DEFAULTS,
    KNOWN_DONE_GATES,
    STAMP_PREFIX,
    ComposeResult,
    WardSet,
    claim_gate,
    compose,
    compose_chain,
    for_gate_class,
    format_stamp,
    never_widens,
    parse_stamp,
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
    "NonceRegistry",
    "DependencyReviewer",
    "MemoryEntry",
    "MemoryStore",
    "PushGrant",
    "check_push",
    "mint_push_grant",
    "verify_grant",
    "PolicyDecision",
    "PolicyFacts",
    "ReputationKey",
    "ReputationLedger",
    "ReviewFinding",
    "ReviewSynthesis",
    "Reviewer",
    "ReviewerReport",
    "SecurityReviewer",
    "TestCoverageReviewer",
    "TrackRecord",
    "GATE_CLASS_DEFAULTS",
    "KNOWN_DONE_GATES",
    "STAMP_PREFIX",
    "claim_gate",
    "for_gate_class",
    "format_stamp",
    "parse_stamp",
    "ComposeResult",
    "WardSet",
    "compose",
    "compose_chain",
    "never_widens",
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
