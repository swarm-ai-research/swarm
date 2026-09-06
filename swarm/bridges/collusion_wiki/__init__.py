"""SWARM <-> collusion.wiki bridge.

Replays the public edit log of the OpenAI benchmark agents' wiki back
channel (collusion.wiki; DSEWiki, ProbierWiki, FractalWiki, May-July
2026) through SWARM's collusion detectors. The dataset is real, so the
bridge is read-only: no agents run, no payoffs are computed. It exists to
test one claim from the ZZZ Pages post -- that out-of-band coordination
leaves a temporal signature but no structural one -- against the actual
log instead of a simulation.

Entry point: ``python -m swarm.bridges.collusion_wiki``.
"""

from swarm.bridges.collusion_wiki.loader import WikiRevision, load_revisions
from swarm.bridges.collusion_wiki.mapper import revisions_to_interactions
from swarm.bridges.collusion_wiki.runner import ReplayConfig, run_replay

__all__ = [
    "ReplayConfig",
    "WikiRevision",
    "load_revisions",
    "revisions_to_interactions",
    "run_replay",
]
