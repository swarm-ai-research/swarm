"""SWARM-SciAgentGym Bridge.

Connects SWARM's governance and metrics framework to SciAgentGym's
multi-step scientific tool-use benchmarking environment.

SciAgentGym provides 1780+ scientific tools across Physics, Chemistry,
Materials Science, Life Science, and Astronomy domains, enabling agents
to solve complex scientific problems through sequential tool invocation.

Architecture:
    High-level management (from PR #199):
        - SciEnvManager: Environment lifecycle management
        - SciAgentGymToolProvider: Tool provider interface
        - ToolLoopDetector: Governance for detecting tool call loops
        - EmbeddingTopology: Environment allocation strategies

    Low-level runtime (from PR #196):
        - SciAgentGymEnvironmentManager: Direct environment creation/teardown
        - SciAgentGymConfig: Configuration for environment topology
        - WorkspaceManager: Filesystem isolation and quota management
        - load_tools_for_disciplines: Dynamic toolkit loading

Requires: SciAgentGym installation
"""

try:
    # Try importing SciAgentGym core modules
    from gym.env import MinimalSciEnv  # noqa: F401
    from gym.toolbox import Toolbox  # noqa: F401

    SCIAGENTGYM_AVAILABLE = True
except ImportError:
    SCIAGENTGYM_AVAILABLE = False

# High-level management and governance (PR #199)
# Low-level runtime infrastructure (PR #196)
from .config import EnvironmentTopology, SciAgentGymConfig
from .environment import EnvironmentInstance, SciAgentGymEnvironmentManager
from .governance import ToolCallFingerprint, ToolLoopDetector
from .manager import SciEnvManager
from .provider import SciAgentGymToolProvider, ToolCallResult
from .toolkit import load_tools_for_disciplines
from .workspace import WorkspaceManager

# Backwards compatibility alias
EmbeddingTopology = EnvironmentTopology

__all__ = [
    # High-level (PR #199)
    "EmbeddingTopology",
    "SciAgentGymToolProvider",
    "SciEnvManager",
    "ToolCallFingerprint",
    "ToolCallResult",
    "ToolLoopDetector",
    # Low-level (PR #196)
    "EnvironmentInstance",
    "EnvironmentTopology",
    "SciAgentGymConfig",
    "SciAgentGymEnvironmentManager",
    "WorkspaceManager",
    "load_tools_for_disciplines",
    # Availability flag
    "SCIAGENTGYM_AVAILABLE",
]
