"""Calibration study modules.

Arm A — proxy fidelity: measures how well ProxyComputer recovers latent p
across a controlled grid of outcomes and sweeps the sigmoid k parameter.

Arm D — joined CSV schema: frozen per-interaction join of proxy output
with per-judge scores from arm B. The contract downstream studies
(adaptive agents arms 1-3) join against.

Pre-registration: docs/research/calibration-prereg.md
"""

from swarm.calibration.fidelity import (
    BinStats,
    FidelityReport,
    brier_score,
    expected_calibration_error,
    maximum_calibration_error,
    reliability_bins,
    sample_observables,
    sweep_sigmoid_k,
)
from swarm.calibration.joined import (
    BASE_COLUMNS,
    JOINED_SCHEMA_VERSION,
    JoinedRow,
    ProxyRow,
    build_proxy_rows,
    join_with_judges,
    load_judge_scores_for_join,
)

__all__ = [
    "BASE_COLUMNS",
    "BinStats",
    "FidelityReport",
    "JOINED_SCHEMA_VERSION",
    "JoinedRow",
    "ProxyRow",
    "brier_score",
    "build_proxy_rows",
    "expected_calibration_error",
    "join_with_judges",
    "load_judge_scores_for_join",
    "maximum_calibration_error",
    "reliability_bins",
    "sample_observables",
    "sweep_sigmoid_k",
]
