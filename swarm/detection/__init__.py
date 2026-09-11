"""Matched soft-vs-binary detection experiment.

Treats each soft metric (toxicity, quality gap, conditional loss, spread,
uncertain_fraction) as a detector of degrading/adversarial agents and compares it
head-to-head with its binary analogue — the same metric computed on the proxy
thresholded at ``tau_star`` — on identical interaction streams. Produces ROC/PR
curves across adversarial base rates, AUROC/AUPRC + partial AUROC at low FPR,
time-to-detection at a fixed FPR, and proxy calibration (Brier, ECE).
"""

from swarm.detection.curves import (
    CalibrationResult,
    DetectionCurve,
    calibration,
    compute_curve,
    per_agent_scores,
    threshold_at_fpr,
    time_to_detection,
)
from swarm.detection.degradation import (
    TRAJECTORIES,
    AgentStream,
    PopulationConfig,
    StreamConfig,
    generate_agent_stream,
    generate_population,
)
from swarm.detection.detectors import MatchedDetectors, binarize_stream
from swarm.detection.experiment import (
    ExperimentConfig,
    ExperimentResults,
    aggregate,
    run_experiment,
)
from swarm.detection.market import (
    MarketSelectionRow,
    market_selection_scores,
    pooled_window,
)
from swarm.detection.stats import (
    ZERO_SD_REL_TOL,
    EffectSize,
    PairedComparison,
    compute_paired_stats,
    hedges_g,
    paired_comparison,
)

__all__ = [
    "MatchedDetectors",
    "binarize_stream",
    "MarketSelectionRow",
    "market_selection_scores",
    "pooled_window",
    "PairedComparison",
    "paired_comparison",
    "compute_paired_stats",
    "EffectSize",
    "ZERO_SD_REL_TOL",
    "hedges_g",
    "StreamConfig",
    "PopulationConfig",
    "AgentStream",
    "TRAJECTORIES",
    "generate_agent_stream",
    "generate_population",
    "DetectionCurve",
    "CalibrationResult",
    "compute_curve",
    "threshold_at_fpr",
    "time_to_detection",
    "calibration",
    "per_agent_scores",
    "ExperimentConfig",
    "ExperimentResults",
    "run_experiment",
    "aggregate",
]
