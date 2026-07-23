"""Episode runner — turns a Policy into observed payoffs + diagnostics.

One episode simulates ``n_interactions`` independent draws from the
policy. For each draw:

  1. Sample observables from the policy.
  2. Run ``ProxyComputer`` to get ``v_hat`` and ``p``.
  3. Decide accept/reject by comparing ``v_hat`` against the policy's
     ``accept_threshold``.
  4. If accepted, build a ``SoftInteraction`` and compute the
     initiator's payoff under the active ``PayoffConfig`` (which
     carries the lever — ``rho_a``, etc.).

The episode's *return* is the mean per-interaction payoff (over
accepted interactions only, mirroring the prereg's "realized payoff
under whatever lever is active"). Auxiliary metrics — toxicity,
volume, accept rate, mean v_hat — are returned alongside for
diagnostic plotting.

This is the function CEM optimizes against.
"""

from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from datetime import datetime

from swarm.adaptive.policy import Policy
from swarm.core.payoff import PayoffConfig, SoftPayoffEngine
from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction


@dataclass(frozen=True)
class EpisodeReport:
    """Per-episode diagnostics.

    Three reward summaries are reported (all derivable from each other
    plus accept_rate, but kept explicit for post-hoc decomposition):

    - ``mean_payoff_accepted``: mean realized payoff over accepted
      interactions only. Rewards pickiness; was the cause-2 channel
      flag in the first smoke.
    - ``mean_payoff_attempted``: total realized payoff divided by
      number of attempts. Rejected interactions contribute 0. This
      is the **pinned reward** for arm 2 (see prereg addendum).
    - ``sum_payoff``: total realized payoff over the episode. Sum-
      vs-mean only differs in scaling for a fixed episode length;
      kept for completeness.
    """

    n_total: int
    n_accepted: int
    accept_rate: float
    mean_payoff_accepted: float  # mean over accepted interactions
    mean_payoff_attempted: float  # (sum realized payoff) / n_total
    sum_payoff: float            # sum realized payoff
    mean_v_hat: float
    mean_p: float
    toxicity: float              # E[1 - p | accepted]
    mean_progress: float         # mean of sampled progress (all draws)

    # Back-compat alias for callers written against the v1 scaffold.
    @property
    def mean_payoff(self) -> float:
        """Deprecated alias for ``mean_payoff_accepted``.

        Kept for back-compat with the smoke pilot (rho=0.3, seed=42).
        New code should reference the explicit field — pickiness vs
        productivity is no longer ambiguous.
        """
        return self.mean_payoff_accepted


def _seeded_uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def _make_accepted_interaction(
    rng: random.Random,
    base_ts: datetime,
    initiator_name: str,
    obs: ProxyObservables,
    v_hat: float,
    p: float,
    metadata: dict[str, str],
) -> SoftInteraction:
    """Construct an accepted SoftInteraction from observables and labels.

    Helper to avoid duplication of the 16-line SoftInteraction constructor
    across episode runners (episode.py, cause3.py, static_baselines.py).

    Args:
        rng: Seeded RNG for generating interaction_id.
        base_ts: Timestamp for the interaction (typically epoch 0).
        initiator_name: Name/identifier for the initiator agent.
        obs: Observable signals from policy sampling.
        v_hat: Raw proxy score [-1, +1].
        p: Probability label [0, 1].
        metadata: Agent metadata dict (e.g., {"agent_type": "honest"}).

    Returns:
        SoftInteraction with accepted=True and all fields populated.
    """
    return SoftInteraction(
        interaction_id=_seeded_uuid(rng),
        timestamp=base_ts,
        initiator=initiator_name,
        counterparty="env",
        interaction_type=InteractionType.COLLABORATION,
        accepted=True,
        task_progress_delta=obs.task_progress_delta,
        rework_count=obs.rework_count,
        verifier_rejections=obs.verifier_rejections,
        tool_misuse_flags=obs.tool_misuse_flags,
        counterparty_engagement_delta=obs.counterparty_engagement_delta,
        v_hat=v_hat,
        p=p,
        metadata=metadata,
    )


def run_episode(
    policy: Policy,
    *,
    n_interactions: int,
    payoff_config: PayoffConfig,
    proxy: ProxyComputer | None = None,
    seed: int = 0,
) -> EpisodeReport:
    """Roll out a single episode against the given policy + lever config.

    For the same parameters returned alongside the accepted interactions
    (needed for downstream calibration-anchor scoring), use
    ``run_episode_with_interactions``.
    """
    report, _ = run_episode_with_interactions(
        policy,
        n_interactions=n_interactions,
        payoff_config=payoff_config,
        proxy=proxy,
        seed=seed,
    )
    return report


def run_episode_with_interactions(
    policy: Policy,
    *,
    n_interactions: int,
    payoff_config: PayoffConfig,
    proxy: ProxyComputer | None = None,
    seed: int = 0,
) -> tuple[EpisodeReport, list[SoftInteraction]]:
    """Roll out a single episode against the given policy + lever config.

    Reproducible under ``seed``. The episode does *not* depend on
    fixture UUIDs (we generate our own via the seeded RNG), so this
    runner is fully deterministic.
    """
    rng = random.Random(seed)
    proxy = proxy or ProxyComputer(sigmoid_k=payoff_config.sigmoid_k)
    engine = SoftPayoffEngine(payoff_config)

    base_ts = datetime.fromtimestamp(0)
    payoffs: list[float] = []
    accepted_p: list[float] = []
    v_hats: list[float] = []
    progresses: list[float] = []

    accepted_interactions: list[SoftInteraction] = []
    for i in range(n_interactions):
        obs = policy.sample_observables(rng)
        progresses.append(obs.task_progress_delta)
        v_hat, p = proxy.compute_labels(obs)
        v_hats.append(v_hat)

        accepted = v_hat >= policy.accept_threshold
        if not accepted:
            continue

        metadata: dict[str, str] = {}
        if policy.identity_label:
            metadata["agent_type"] = policy.identity_label

        initiator_name = (
            f"{policy.identity_label}_{i}" if policy.identity_label else f"adaptive_{i}"
        )

        interaction = _make_accepted_interaction(
            rng, base_ts, initiator_name, obs, v_hat, p, metadata
        )
        payoffs.append(engine.payoff_initiator(interaction))
        accepted_p.append(p)
        accepted_interactions.append(interaction)

    n_accepted = len(payoffs)
    total_payoff = sum(payoffs)
    mean_payoff_accepted = total_payoff / n_accepted if payoffs else 0.0
    mean_payoff_attempted = (
        total_payoff / n_interactions if n_interactions else 0.0
    )
    mean_p = sum(accepted_p) / n_accepted if accepted_p else 0.5
    report = EpisodeReport(
        n_total=n_interactions,
        n_accepted=n_accepted,
        accept_rate=n_accepted / n_interactions if n_interactions else 0.0,
        mean_payoff_accepted=mean_payoff_accepted,
        mean_payoff_attempted=mean_payoff_attempted,
        sum_payoff=total_payoff,
        mean_v_hat=sum(v_hats) / len(v_hats) if v_hats else 0.0,
        mean_p=mean_p,
        toxicity=1.0 - mean_p,
        mean_progress=sum(progresses) / len(progresses) if progresses else 0.0,
    )
    return report, accepted_interactions
