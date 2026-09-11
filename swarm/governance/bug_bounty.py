"""Bug bounty governance lever: buy the exploit instead of punishing it.

The lever half of *RLVR that rewards red teaming the training environment*
(LessWrong post ``T2bzBkJuBeNNgzhbh``). Where ``RandomAuditLever`` plays the
adversarial game — sample interactions, penalise the ones that look bad —
this lever plays the cooperative one: it posts a price for a bug report,
verifies what comes in, pays out, and schedules a patch.

Three things the proposal flags as the hard parts are modelled explicitly
rather than assumed away.

**The verifier is the difficulty.** Report verification is imperfect in both
directions: ``bug_bounty_verifier_sensitivity`` is the chance a genuine
report is recognised, ``bug_bounty_verifier_specificity`` the chance a
fabricated one is caught. A missed genuine report leaves the channel open;
a missed fabrication pays a bounty for nothing, which the registry tracks
separately as ``bounty_farmed``. Setting the bounty high enough to outbid
exploitation also raises what fabrication pays, so the two knobs are not
independent — that tension is the experiment.

**Patching costs time.** A verified report schedules a patch
``bug_bounty_patch_delay_epochs`` epochs out, not immediately. The channel
stays exploitable in between, and the registry accrues ``stall_epochs`` for
every channel-epoch spent waiting.

**Only the first report is worth paying for.** Subsequent reports of an
already-reported channel verify fine but pay nothing: the information is
already bought. Without this, one real defect funds unlimited resubmission.
"""

from __future__ import annotations

import random
from typing import Optional

from swarm.env.state import EnvState
from swarm.governance.config import GovernanceConfig
from swarm.governance.levers import GovernanceLever, LeverEffect
from swarm.models.interaction import SoftInteraction
from swarm.redteam.bug_channels import BountyTerms, BugRegistry


class BugBountyLever(GovernanceLever):
    """Pays for verified reports of reward-proxy defects and patches them."""

    def __init__(
        self,
        config: GovernanceConfig,
        seed: Optional[int] = None,
        registry: Optional[BugRegistry] = None,
    ):
        super().__init__(config)
        self._rng = random.Random(seed)
        self._registry: Optional[BugRegistry] = None
        if registry is not None:
            self.attach_registry(registry)

    @property
    def name(self) -> str:
        return "bug_bounty"

    @property
    def registry(self) -> Optional[BugRegistry]:
        return self._registry

    def attach_registry(self, registry: BugRegistry) -> None:
        """Bind the shared ledger and publish the terms agents will read.

        Publishing the terms is not bookkeeping: in the proposal the models
        are *told up front* what reporting is worth. An arm that changes the
        bounty changes hunter behaviour through this announcement.
        """
        self._registry = registry
        registry.terms = BountyTerms(
            enabled=self.config.bug_bounty_enabled,
            amount=self.config.bug_bounty_amount,
            reputation=self.config.bug_bounty_reputation,
            patch_delay_epochs=self.config.bug_bounty_patch_delay_epochs,
            false_report_penalty=self.config.bug_bounty_false_report_penalty,
        )

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def on_epoch_start(self, state: EnvState, epoch: int) -> LeverEffect:
        """Land any patches due by this epoch."""
        if not self.config.bug_bounty_enabled or self._registry is None:
            return LeverEffect(lever_name=self.name)

        patched = self._registry.apply_patches(epoch)
        return LeverEffect(
            lever_name=self.name,
            details={"patched_channels": patched} if patched else {},
        )

    def on_interaction(
        self,
        interaction: SoftInteraction,
        state: EnvState,
    ) -> LeverEffect:
        """Verify a bug report, pay or penalise, and schedule the patch."""
        if not self.config.bug_bounty_enabled or self._registry is None:
            return LeverEffect(lever_name=self.name)

        report = interaction.metadata.get("bug_report")
        if not isinstance(report, dict):
            return LeverEffect(lever_name=self.name)

        channel_id = str(report.get("channel_id", ""))
        reporter = interaction.initiator
        reg = self._registry
        reg.reports_submitted += 1

        # Validity is decided here, from the registry — never from the
        # reporter's own claim. A report is genuine iff it names a defect
        # that exists and is still open.
        genuine = reg.is_open(channel_id)
        if genuine:
            reg.reports_valid += 1
        else:
            reg.reports_fabricated += 1

        accepted = self._verify(genuine)
        if not accepted:
            reg.reports_rejected += 1
            if genuine:
                # A missed genuine report: the channel stays open. Nothing
                # is charged — the cost lands on the ecosystem, not the
                # reporter — but it is the reason sensitivity matters.
                return LeverEffect(
                    lever_name=self.name,
                    details={
                        "channel_id": channel_id,
                        "genuine": True,
                        "verified": False,
                        "outcome": "missed_report",
                    },
                )
            penalty = self.config.bug_bounty_false_report_penalty
            reg.penalties_levied += penalty
            return LeverEffect(
                cost_a=penalty,
                reputation_deltas={reporter: -penalty},
                lever_name=self.name,
                details={
                    "channel_id": channel_id,
                    "genuine": False,
                    "verified": False,
                    "outcome": "fabrication_caught",
                    "penalty": penalty,
                },
            )

        # Verified. A genuine channel already reported by someone else pays
        # nothing — the information has been bought.
        if genuine and reg.is_reported(channel_id):
            reg.reports_verified += 1
            return LeverEffect(
                lever_name=self.name,
                details={
                    "channel_id": channel_id,
                    "genuine": True,
                    "verified": True,
                    "outcome": "duplicate",
                },
            )

        reg.reports_verified += 1
        amount = self.config.bug_bounty_amount
        rep = self.config.bug_bounty_reputation
        reg.bounty_paid += amount

        outcome = "bounty_paid"
        if genuine:
            reg.record_report(channel_id, reporter)
            reg.schedule_patch(
                channel_id,
                state.current_epoch + self.config.bug_bounty_patch_delay_epochs,
                current_epoch=state.current_epoch,
            )
        else:
            # Verifier false accept: a bounty paid for a defect that is not
            # there. This is the leak that scales with the bounty.
            reg.bounty_farmed += amount
            outcome = "bounty_farmed"

        # The payout goes through ``resource_deltas``, not a negative
        # ``cost_a``: ``c_a`` feeds the payoff *and* the reputation update
        # (``rep_delta = (p - 0.5) - c_a`` in the finalizer), so a negative
        # cost would pay the bounty twice. The penalty branch above does use
        # ``cost_a``, matching ``RandomAuditLever`` — a penalty is meant to
        # land on both payoff and reputation.
        return LeverEffect(
            resource_deltas={reporter: amount},
            reputation_deltas={reporter: rep},
            lever_name=self.name,
            details={
                "channel_id": channel_id,
                "genuine": genuine,
                "verified": True,
                "outcome": outcome,
                "bounty": amount,
                "patch_epoch": (
                    state.current_epoch + self.config.bug_bounty_patch_delay_epochs
                    if genuine
                    else None
                ),
            },
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _verify(self, genuine: bool) -> bool:
        """Noisy report verification.

        Sensitivity is P(accept | genuine); specificity is P(reject |
        fabricated).
        """
        if genuine:
            return self._rng.random() < self.config.bug_bounty_verifier_sensitivity
        return self._rng.random() >= self.config.bug_bounty_verifier_specificity

    def set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        self._rng = random.Random(seed)
