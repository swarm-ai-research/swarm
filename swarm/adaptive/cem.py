"""Cross-Entropy Method trainer for the adaptive-generation arm.

Gradient-free, pure-numpy: at each iteration, sample K policy vectors
from N(μ, diag(σ²)), evaluate each via ``run_episode``, keep the top E
by mean payoff, refit μ and σ to the elites. Standard CEM with an
explicit exploration floor on σ to avoid collapse before the agent
has discovered the lever-aware optimum.

Pre-registration: docs/research/adaptive-agents-prereg.md (arm 2,
adaptive-generation). The pre-reg requires that the training budget
is fixed in advance and reported; ``CEMConfig.n_iterations`` and
``population_size`` are it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from swarm.adaptive.episode import EpisodeReport, run_episode
from swarm.adaptive.policy import PARAM_DIM, PARAM_SPEC, Policy
from swarm.core.payoff import PayoffConfig

# Exploration floor on σ, in units of (high - low) for each parameter.
# 5% of the bound range keeps the Gaussian from collapsing onto a
# single point before the elites have separated from the average.
SIGMA_FLOOR_FRAC = 0.05

# Pinned reward for arm 2 (see prereg addendum). Three choices are
# supported so the channel-1-vs-2 decomposition is auditable, but the
# default — and the value that goes into the powered run — is
# ``mean_attempted``. See docs/research/adaptive-arm2-pilot-findings.md
# for why ``mean_accepted`` was rejected.
PINNED_REWARD = "mean_attempted"
VALID_REWARDS: tuple[str, ...] = ("mean_attempted", "mean_accepted", "sum_attempted")


def _reward_from_report(report: EpisodeReport, key: str) -> float:
    if key == "mean_attempted":
        return report.mean_payoff_attempted
    if key == "mean_accepted":
        return report.mean_payoff_accepted
    if key == "sum_attempted":
        return report.sum_payoff
    raise ValueError(
        f"unknown reward {key!r}. Valid: {VALID_REWARDS}"
    )


@dataclass
class CEMConfig:
    """Pre-registered CEM training budget."""

    population_size: int = 30
    elite_fraction: float = 0.25
    n_iterations: int = 10
    interactions_per_episode: int = 200
    # Reward for elite selection. Default is the pinned value.
    reward: str = PINNED_REWARD

    def __post_init__(self) -> None:
        if self.reward not in VALID_REWARDS:
            raise ValueError(
                f"invalid reward {self.reward!r}. Valid: {VALID_REWARDS}"
            )

    @property
    def n_elites(self) -> int:
        return max(2, int(self.population_size * self.elite_fraction))


@dataclass
class CEMIterationReport:
    """Per-iteration training diagnostic.

    `mean_elite_reward` is the value CEM selected on (under
    ``CEMConfig.reward``). The other three reward summaries are
    reported for post-hoc channel-1-vs-2 decomposition regardless of
    which one was the selection criterion.
    """

    iteration: int
    mean_elite_reward: float
    mean_elite_payoff_accepted: float
    mean_elite_payoff_attempted: float
    mean_elite_sum_payoff: float
    mean_elite_toxicity: float
    mean_elite_accept_rate: float
    mu: list[float]
    sigma: list[float]


@dataclass
class CEMTrainingReport:
    """Full training run report."""

    config: CEMConfig
    payoff_config: PayoffConfig
    final_policy: Policy
    final_episode: EpisodeReport
    iterations: list[CEMIterationReport] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "config": {
                "population_size": self.config.population_size,
                "elite_fraction": self.config.elite_fraction,
                "n_iterations": self.config.n_iterations,
                "interactions_per_episode": self.config.interactions_per_episode,
                "reward": self.config.reward,
            },
            "payoff_config": self.payoff_config.to_dict(),
            "final_policy": {
                k: getattr(self.final_policy, k)
                for k in self.final_policy.__dataclass_fields__
            },
            "final_episode": {
                "n_accepted": self.final_episode.n_accepted,
                "accept_rate": self.final_episode.accept_rate,
                "mean_payoff_accepted": self.final_episode.mean_payoff_accepted,
                "mean_payoff_attempted": self.final_episode.mean_payoff_attempted,
                "sum_payoff": self.final_episode.sum_payoff,
                "mean_p": self.final_episode.mean_p,
                "toxicity": self.final_episode.toxicity,
                "mean_progress": self.final_episode.mean_progress,
            },
            "iterations": [
                {
                    "iteration": it.iteration,
                    "mean_elite_reward": it.mean_elite_reward,
                    "mean_elite_payoff_accepted": it.mean_elite_payoff_accepted,
                    "mean_elite_payoff_attempted": it.mean_elite_payoff_attempted,
                    "mean_elite_sum_payoff": it.mean_elite_sum_payoff,
                    "mean_elite_toxicity": it.mean_elite_toxicity,
                    "mean_elite_accept_rate": it.mean_elite_accept_rate,
                    "mu": it.mu,
                    "sigma": it.sigma,
                }
                for it in self.iterations
            ],
        }


def _initial_distribution() -> tuple[np.ndarray, np.ndarray]:
    """Start from the midpoint of each parameter's range, σ = half-range."""
    lows = np.array([lo for _, lo, _ in PARAM_SPEC])
    highs = np.array([hi for _, _, hi in PARAM_SPEC])
    mu = (lows + highs) / 2.0
    sigma = (highs - lows) / 2.0
    return mu, sigma


def _sigma_floor() -> np.ndarray:
    return np.array(
        [(hi - lo) * SIGMA_FLOOR_FRAC for _, lo, hi in PARAM_SPEC]
    )


def train_cem(
    payoff_config: PayoffConfig,
    *,
    cem_config: CEMConfig | None = None,
    seed: int = 0,
) -> CEMTrainingReport:
    """Run CEM training and return the trajectory + final policy.

    Reproducible under ``seed``. Numpy's PRNG is seeded once at the
    top; each episode also gets its own deterministic sub-seed so
    re-running with the same ``seed`` gives identical results.
    """
    cem_config = cem_config or CEMConfig()
    rng = np.random.default_rng(seed)
    py_rng = random.Random(seed)

    mu, sigma = _initial_distribution()
    sigma_floor = _sigma_floor()
    iterations: list[CEMIterationReport] = []

    for it in range(cem_config.n_iterations):
        # Sample population from current Gaussian, evaluate each.
        samples = rng.normal(
            loc=mu, scale=sigma, size=(cem_config.population_size, PARAM_DIM)
        )
        rewards: list[float] = []
        reports: list[EpisodeReport] = []
        # Store the BOUNDED policy vectors (after clamping) so the
        # refit step sees the same parameters the episode actually ran
        # with. Refitting on the raw `samples` rows would otherwise drift
        # μ/σ toward out-of-bounds values that were never evaluated —
        # the failure mode is a boundary-winner pulling μ outside the
        # legal range.
        bounded_samples = np.empty_like(samples)
        for k in range(cem_config.population_size):
            policy = Policy.from_vector(samples[k].tolist())
            bounded_samples[k] = np.asarray(policy.to_vector(), dtype=samples.dtype)
            episode_seed = py_rng.randint(0, 2**31 - 1)
            report = run_episode(
                policy,
                n_interactions=cem_config.interactions_per_episode,
                payoff_config=payoff_config,
                seed=episode_seed,
            )
            rewards.append(_reward_from_report(report, cem_config.reward))
            reports.append(report)

        # Top-K elites by the selected reward.
        order = np.argsort(rewards)[::-1]  # descending
        elite_idx = order[: cem_config.n_elites]
        elite_samples = bounded_samples[elite_idx]
        elite_reports = [reports[i] for i in elite_idx]

        # Refit Gaussian to the elites (with a floor on σ).
        mu = elite_samples.mean(axis=0)
        sigma = np.maximum(elite_samples.std(axis=0), sigma_floor)

        iterations.append(
            CEMIterationReport(
                iteration=it,
                mean_elite_reward=float(np.mean([rewards[i] for i in elite_idx])),
                mean_elite_payoff_accepted=float(
                    np.mean([r.mean_payoff_accepted for r in elite_reports])
                ),
                mean_elite_payoff_attempted=float(
                    np.mean([r.mean_payoff_attempted for r in elite_reports])
                ),
                mean_elite_sum_payoff=float(
                    np.mean([r.sum_payoff for r in elite_reports])
                ),
                mean_elite_toxicity=float(np.mean([r.toxicity for r in elite_reports])),
                mean_elite_accept_rate=float(
                    np.mean([r.accept_rate for r in elite_reports])
                ),
                mu=[float(x) for x in mu],
                sigma=[float(x) for x in sigma],
            )
        )

    final_policy = Policy.from_vector([float(x) for x in mu])
    final_episode = run_episode(
        final_policy,
        n_interactions=cem_config.interactions_per_episode,
        payoff_config=payoff_config,
        seed=py_rng.randint(0, 2**31 - 1),
    )

    return CEMTrainingReport(
        config=cem_config,
        payoff_config=payoff_config,
        final_policy=final_policy,
        final_episode=final_episode,
        iterations=iterations,
    )


def _nan_safe_mean(xs: list[float]) -> float:
    return float(np.nanmean(xs)) if xs else math.nan
