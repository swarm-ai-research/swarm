"""Certificate gate governance lever (beads pins).

Verifies interactions against recorded latent ground truth rather than the
proxy label p. Where the random audit lever asks "does the proxy think this
was bad?", the certificate gate asks "was this actually bad?" — but only for
the fraction of interactions a certificate-grade verifier covers
(``certificate_coverage``), and only when ground truth was recorded at all
(``SoftInteraction.ground_truth`` is not None).

Motivation and predictions: docs/research/dgg-counterexample-lessons.md §2
and §5. The qualitative claim under test is that conditioning consequences
on certificates rather than plausibility flips the effect of demand
pressure on ecosystem quality.
"""

import random
from typing import Optional

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction


class CertificateGateLever(GovernanceLever):
    """Verify covered interactions against latent ground truth.

    A caught fabrication (ground_truth == -1) incurs a flat cost penalty
    and an equal reputation penalty on the initiator. The reputation
    penalty feeds back into observable quality via reputation-modulated
    generation, and gives adaptive workers the "caught" signal they use to
    decay fabrication propensity.
    """

    def __init__(self, config: GovernanceConfig, seed: Optional[int] = None):
        super().__init__(config)
        self._rng = random.Random(seed)

    @property
    def name(self) -> str:
        return "certificate_gate"

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """Verify the interaction if it falls under certificate coverage."""
        if not self.config.certificate_gate_enabled:
            return LeverEffect(lever_name=self.name)

        if interaction.ground_truth is None:
            return LeverEffect(
                lever_name=self.name,
                details={"verified": False, "reason": "no_certificate"},
            )

        if self._rng.random() > self.config.certificate_coverage:
            return LeverEffect(
                lever_name=self.name,
                details={"verified": False, "reason": "not_covered"},
            )

        caught = interaction.ground_truth == -1
        if not caught:
            return LeverEffect(
                lever_name=self.name,
                details={"verified": True, "caught": False},
            )

        penalty = self.config.certificate_penalty
        return LeverEffect(
            cost_a=penalty,
            reputation_deltas={interaction.initiator: -penalty},
            lever_name=self.name,
            details={
                "verified": True,
                "caught": True,
                "penalty_amount": penalty,
            },
        )

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = random.Random(seed)
