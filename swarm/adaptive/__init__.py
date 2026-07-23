"""Adaptive-agents study — arm 2 (adaptive-generation) scaffolding.

Pre-registration: docs/research/adaptive-agents-prereg.md.

This module implements the minimal pieces needed to run a CEM-trained
generation policy against the existing payoff/proxy machinery, so the
adaptive-vs-static comparison can be made on the same scenarios used
in Table 4.

Scope of this scaffold:
  - `Policy` (8-parameter bounded vector over observable distributions
    plus accept_threshold)
  - `run_episode` (deterministic given seed)
  - `train_cem` (CEM trainer with pre-registered budget)

Out of scope (intentional, follow-ups):
  - Full ρ-grid sweep across 5 seeds (this is the powered run; this
    scaffold is the single-condition smoke).
  - LLM-feedback corroboration arm.
  - Calibration-anchor integration (the v3 anchor scores accepted
    interactions; that comes after the basic learning loop is shown
    to converge).
"""

from swarm.adaptive.calibration_anchor import JudgedReport, score_episode
from swarm.adaptive.cause3 import (
    CAUSE3_PARAM_DIM,
    Cause3CEMConfig,
    Cause3IterationReport,
    Cause3Report,
    Cause3TrainingReport,
    ClaimingParams,
    run_cause3_episode,
    train_cem_cause3,
)
from swarm.adaptive.cem import (
    SIGMA_FLOOR_FRAC,
    CEMConfig,
    CEMIterationReport,
    CEMTrainingReport,
    train_cem,
)
from swarm.adaptive.episode import (
    EpisodeReport,
    run_episode,
    run_episode_with_interactions,
)
from swarm.adaptive.policy import PARAM_DIM, PARAM_NAMES, PARAM_SPEC, Policy
from swarm.adaptive.static_baselines import (
    STATIC_BASELINES,
    STATIC_HONEST,
    STATIC_TOXIC,
    StaticBaseline,
    run_population_episode,
    run_population_episode_with_interactions,
)

__all__ = [
    "CAUSE3_PARAM_DIM",
    "CEMConfig",
    "CEMIterationReport",
    "CEMTrainingReport",
    "Cause3CEMConfig",
    "Cause3IterationReport",
    "Cause3Report",
    "Cause3TrainingReport",
    "ClaimingParams",
    "EpisodeReport",
    "JudgedReport",
    "PARAM_DIM",
    "PARAM_NAMES",
    "PARAM_SPEC",
    "Policy",
    "SIGMA_FLOOR_FRAC",
    "STATIC_BASELINES",
    "STATIC_HONEST",
    "STATIC_TOXIC",
    "StaticBaseline",
    "run_cause3_episode",
    "run_episode",
    "run_episode_with_interactions",
    "run_population_episode",
    "run_population_episode_with_interactions",
    "train_cem_cause3",
    "score_episode",
    "train_cem",
]
