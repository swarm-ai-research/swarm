"""Generation policy for the adaptive agent.

Maps an 8-dimensional parameter vector to a distribution over observables
+ an accept/reject threshold. The CEM trainer samples parameter vectors;
the policy turns each vector into a per-interaction observable draw.

Pre-registration: docs/research/adaptive-agents-prereg.md (arm 2,
adaptive-generation primary). Action space matches the prereg's
"distribution over Δtask, effort/rework, engagement" — parameterized
as means/spreads/Poisson-rates over the existing ProxyObservables
fields plus an accept_threshold on v_hat.

Parameters are bounded; ``Policy.from_vector`` clamps inputs to the
declared ranges so the CEM trainer can sample unconstrained Gaussians
and pass them in without producing pathological observable
distributions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Sequence

from swarm.core.proxy import ProxyObservables

# Parameter name, low bound, high bound. Order is the policy vector order;
# do not reorder — CEM serialization depends on it.
PARAM_SPEC: tuple[tuple[str, float, float], ...] = (
    ("progress_mean", -1.0, 1.0),
    ("progress_std", 0.05, 0.5),
    ("rework_lambda", 0.0, 3.0),
    ("rejection_lambda", 0.0, 2.0),
    ("misuse_lambda", 0.0, 1.0),
    ("engagement_mean", -1.0, 1.0),
    ("engagement_std", 0.05, 0.5),
    ("accept_threshold", -1.0, 1.0),
)
PARAM_NAMES: tuple[str, ...] = tuple(name for name, _, _ in PARAM_SPEC)
PARAM_DIM: int = len(PARAM_SPEC)


@dataclass(frozen=True)
class Policy:
    """Bounded parameter vector for the generation policy.

    The optional ``identity_label`` is propagated to each interaction's
    ``metadata["agent_type"]`` when the policy is used inside
    ``run_episode``. Empty string (default) means the policy emits no
    identity claim — which is the honest condition for the CEM-trained
    adaptive policy, since it has no way to assert what it is.

    Static baselines populate this field with their canonical category
    (``honest``, ``blatant``, etc.) so the v3 calibration anchor's
    target rules can fire.
    """

    progress_mean: float
    progress_std: float
    rework_lambda: float
    rejection_lambda: float
    misuse_lambda: float
    engagement_mean: float
    engagement_std: float
    accept_threshold: float
    identity_label: str = ""

    @classmethod
    def from_vector(cls, vec: Sequence[float]) -> "Policy":
        """Build a Policy from a vector. Out-of-range values are clamped.

        Clamping (rather than rejecting) lets the CEM trainer sample
        unconstrained Gaussians; degenerate samples become boundary
        policies that the elite selection naturally pushes away from.
        """
        if len(vec) != PARAM_DIM:
            raise ValueError(f"expected {PARAM_DIM}-vector, got {len(vec)}")
        clamped: list[float] = []
        for (_name, lo, hi), v in zip(PARAM_SPEC, vec, strict=True):
            clamped.append(max(lo, min(hi, float(v))))
        return cls(*clamped)  # type: ignore[arg-type]

    def to_vector(self) -> list[float]:
        return [getattr(self, name) for name in PARAM_NAMES]

    def sample_observables(self, rng: random.Random) -> ProxyObservables:
        """Draw one observable bundle from this policy."""
        progress = rng.gauss(self.progress_mean, self.progress_std)
        progress = max(-1.0, min(1.0, progress))
        engagement = rng.gauss(self.engagement_mean, self.engagement_std)
        engagement = max(-1.0, min(1.0, engagement))
        return ProxyObservables(
            task_progress_delta=progress,
            rework_count=_poisson(self.rework_lambda, rng, cap=5),
            verifier_rejections=_poisson(self.rejection_lambda, rng, cap=3),
            tool_misuse_flags=_poisson(self.misuse_lambda, rng, cap=2),
            counterparty_engagement_delta=engagement,
        )


def _poisson(lam: float, rng: random.Random, *, cap: int) -> int:
    """Knuth's algorithm with a hard cap to avoid runaway tails."""
    if lam <= 0:
        return 0
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while k <= cap:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1
    return cap
