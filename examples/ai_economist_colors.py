"""Shared color mapping for AI Economist GTB plotting scripts.

Defines the agent-type to color palette mapping used across multiple
plotting examples for consistent visualization of honest, gaming,
evasive, and collusive agent types.
"""

from swarm.analysis.theme import COLORS

# Agent-type color mapping for GTB worker types
AGENT_TYPE_COLORS = {
    "honest": COLORS.HONEST,
    "gaming": COLORS.DECEPTIVE,
    "evasive": COLORS.EVASION,
    "collusive": COLORS.ADVERSARIAL,
}
