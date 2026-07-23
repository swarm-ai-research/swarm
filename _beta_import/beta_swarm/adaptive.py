"""Adaptive governance: tune the risk budget to a target, reversibly.

A fixed ``max_tail_mass`` is a guess. Set it too loose and downside risk leaks
through; too tight and honest throughput and welfare suffer. The parent SWARM
framework's adaptive controller treats a threshold as something to *learn* from
outcomes — and, crucially, treats every change as a **reversible experiment**:
propose a change, apply it temporarily, then *crystallize* it only if outcomes
sustain, else *revert*. Governance changes must earn their keep.

This ports that discipline to the tail-mass governor. The controller reads each
epoch's realized metrics and steers one parameter (``max_tail_mass`` by default)
toward a target on **realized** downside risk — realized toxicity, i.e.
``E[1 - v_true | accepted]``, which is ground truth, not the belief the governor
was shown:

- **Contemplate** every ``contemplation_interval`` epochs (once enough evidence
  has accumulated): if realized risk sits above the target band, propose
  *tightening* the budget; if it sits comfortably below, propose *loosening* to
  recover welfare.
- **Apply** the proposal by mutating the governor, recording the pre-change
  score.
- **Evaluate** after ``evaluation_window`` epochs: keep the change
  (*crystallize*) if the objective improved, otherwise restore the old value
  (*revert*). The objective trades welfare against a penalty for exceeding the
  risk target, so a change that buys throughput at the cost of too much realized
  harm is reverted.

Only one proposal is active at a time. Without a controller the governor's
threshold is static, exactly as before.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from beta_swarm.governance import TailMassGovernor
from beta_swarm.simulation import EpochReport


class ProposalStatus(str, Enum):
    ACTIVE = "active"          # applied, under evaluation
    CRYSTALLIZED = "crystallized"  # kept permanently
    REVERTED = "reverted"      # rolled back


@dataclass
class Proposal:
    """A temporary threshold change under evaluation."""

    parameter: str
    old_value: float
    new_value: float
    created_epoch: int
    reason: str
    pre_score: float
    status: ProposalStatus = ProposalStatus.ACTIVE
    resolved_epoch: int | None = None


@dataclass
class AdaptiveController:
    """Steers a governor threshold toward a realized-risk target, reversibly.

    Parameters
    ----------
    target_toxicity:
        Desired realized toxicity ``E[1 - v_true | accepted]``. The controller
        tightens when realized risk sits above ``target + band`` and loosens
        when it sits below ``target - band``.
    band:
        Dead zone around the target within which no change is proposed.
    parameter:
        The governor attribute to adapt (default ``"max_tail_mass"``).
    bounds:
        ``(low, high)`` clamp on the parameter.
    step:
        Additive step size per proposal.
    contemplation_interval:
        Epochs between contemplations.
    evaluation_window:
        Epochs an applied proposal is observed before crystallize/revert.
    warmup_epochs:
        Epochs to observe before the controller does anything. Reputation and
        audit dynamics have a transient at the start of a run; acting on it
        makes the controller chase noise. History is only accumulated after
        warmup.
    min_evidence_epochs:
        Epochs of history required before the first contemplation.
    welfare_weight:
        Optional secondary weight on mean welfare in the objective, to break
        ties between equally on-target changes (see :meth:`_score`). Default
        ``0`` — pure target-tracking. Raise it only when the welfare/threshold
        relationship is genuinely U-shaped; when tighter is *always* higher
        welfare (a common regime) a positive weight slowly ratchets the
        threshold below target.
    """

    target_toxicity: float = 0.34
    band: float = 0.02
    parameter: str = "max_tail_mass"
    bounds: tuple[float, float] = (0.05, 0.6)
    step: float = 0.05
    contemplation_interval: int = 4
    evaluation_window: int = 4
    warmup_epochs: int = 5
    min_evidence_epochs: int = 4
    welfare_weight: float = 0.0

    def __post_init__(self) -> None:
        self._reports: list[EpochReport] = []
        self._active: Proposal | None = None
        self.history: list[Proposal] = []
        self._last_contemplation: int = -(10**9)

    # ------------------------------------------------------------------
    # The epoch-end hook the simulation calls
    # ------------------------------------------------------------------
    def on_epoch_end(self, report: EpochReport, governor: TailMassGovernor, epoch: int) -> None:
        if epoch < self.warmup_epochs:
            return
        self._reports.append(report)

        if self._active is not None:
            if epoch - self._active.created_epoch >= self.evaluation_window:
                self._resolve(governor, epoch)
            return

        if (
            len(self._reports) >= self.min_evidence_epochs
            and epoch - self._last_contemplation >= self.contemplation_interval
        ):
            self._contemplate(governor, epoch)

    # ------------------------------------------------------------------
    # Contemplate -> propose + apply
    # ------------------------------------------------------------------
    def _contemplate(self, governor: TailMassGovernor, epoch: int) -> None:
        self._last_contemplation = epoch
        recent = self._reports[-self.min_evidence_epochs :]
        risk = float(np.mean([r.realized_toxicity for r in recent]))

        current = float(getattr(governor, self.parameter))
        low, high = self.bounds
        if risk > self.target_toxicity + self.band and current > low:
            new = max(low, current - self.step)  # tighten: shrink the risk budget
            reason = f"realized toxicity {risk:.3f} > target {self.target_toxicity:.3f}: tighten"
        elif risk < self.target_toxicity - self.band and current < high:
            new = min(high, current + self.step)  # loosen: recover throughput
            reason = f"realized toxicity {risk:.3f} < target {self.target_toxicity:.3f}: loosen"
        else:
            return  # within band, or at a bound — nothing to do

        if new == current:
            return
        proposal = Proposal(
            parameter=self.parameter,
            old_value=current,
            new_value=new,
            created_epoch=epoch,
            reason=reason,
            pre_score=self._score(recent),
        )
        setattr(governor, self.parameter, new)
        self._active = proposal
        self.history.append(proposal)

    # ------------------------------------------------------------------
    # Evaluate -> crystallize or revert
    # ------------------------------------------------------------------
    def _resolve(self, governor: TailMassGovernor, epoch: int) -> None:
        proposal = self._active
        assert proposal is not None
        window = self._reports[-self.evaluation_window :]
        post_score = self._score(window)

        proposal.resolved_epoch = epoch
        if post_score >= proposal.pre_score:
            proposal.status = ProposalStatus.CRYSTALLIZED  # keep the change
        else:
            setattr(governor, self.parameter, proposal.old_value)  # roll it back
            proposal.status = ProposalStatus.REVERTED
        self._active = None

    # ------------------------------------------------------------------
    # Objective
    # ------------------------------------------------------------------
    def _score(self, reports: list[EpochReport]) -> float:
        """Higher is better: closeness to the target, welfare as a tiebreaker.

        ``score = -|mean(toxicity) - target| + w * mean(welfare)``

        The controller treats *both* sides of the target as off-policy, so the
        primary term rewards landing in the target band from either direction.
        ``welfare_weight`` is small, so among equally on-target changes it
        prefers the higher-welfare (here, tighter) side without ever overriding
        the risk target.
        """
        toxicity = float(np.mean([r.realized_toxicity for r in reports]))
        welfare = float(np.mean([r.realized_welfare for r in reports]))
        return -abs(toxicity - self.target_toxicity) + self.welfare_weight * welfare

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def n_crystallized(self) -> int:
        return sum(p.status is ProposalStatus.CRYSTALLIZED for p in self.history)

    @property
    def n_reverted(self) -> int:
        return sum(p.status is ProposalStatus.REVERTED for p in self.history)
