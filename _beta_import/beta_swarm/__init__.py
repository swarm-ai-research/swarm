"""beta-swarm: distributional safety with a full belief over a continuous outcome.

Where the soft-label formalism carried a scalar ``p = P(v = +1)``, beta-swarm
carries a ``Beta(alpha, beta)`` belief over the continuous outcome ``v in [0,1]``,
exposing both a **mean** (recovering ``p``) and a **concentration** (epistemic
certainty). Payoffs become ``E_F[surplus(v)]``, governance triggers on tail mass
``P(v < tau)``, and the quality gap generalizes to a Wasserstein/KL distance
between accepted and rejected belief distributions.
"""

from beta_swarm.adaptive import AdaptiveController, Proposal, ProposalStatus
from beta_swarm.agents import Archetype, SimAgent, make_population
from beta_swarm.belief import BetaBelief
from beta_swarm.calibration import (
    BinStats,
    CalibrationReport,
    calibrate,
    calibrate_interactions,
    fidelity_run,
    pit_values,
    sweep_evidence_scale,
    tail_reliability_bins,
)
from beta_swarm.collusion import CollusionDetector, PairReport
from beta_swarm.governance import (
    Decision,
    GovernanceVerdict,
    TailMassGovernor,
    ValueAtRiskGovernor,
)
from beta_swarm.interaction import BetaInteraction, InteractionType
from beta_swarm.levers import (
    GovernanceLever,
    GovernanceStack,
    LeverContext,
    LeverEffect,
    RiskTaxLever,
    StakingLever,
    TailCircuitBreaker,
)
from beta_swarm.metrics import (
    DistributionalMetrics,
    beta_kl,
    kl_mixtures,
    wasserstein1_mixtures,
)
from beta_swarm.payoff import (
    DistributionalPayoffEngine,
    PayoffBreakdown,
    PayoffConfig,
)
from beta_swarm.proxy import BetaProxyComputer, ProxyObservables
from beta_swarm.scenarios import ScenarioConfig, load_scenario, run_scenario
from beta_swarm.redteam import (
    AttackCategory,
    AttackLibrary,
    AttackResult,
    AttackScenario,
    ReputationFarmer,
    VolumeForger,
    run_attack,
)
from beta_swarm.simulation import (
    EpochReport,
    Simulation,
    SimulationConfig,
    SimulationResult,
    decay_belief,
    pool_beliefs,
)

__version__ = "0.1.0"

__all__ = [
    "BetaBelief",
    "BetaInteraction",
    "InteractionType",
    "DistributionalPayoffEngine",
    "PayoffConfig",
    "PayoffBreakdown",
    "TailMassGovernor",
    "ValueAtRiskGovernor",
    "GovernanceVerdict",
    "Decision",
    "DistributionalMetrics",
    "beta_kl",
    "wasserstein1_mixtures",
    "kl_mixtures",
    "BetaProxyComputer",
    "ProxyObservables",
    "Archetype",
    "SimAgent",
    "make_population",
    "Simulation",
    "SimulationConfig",
    "SimulationResult",
    "EpochReport",
    "pool_beliefs",
    "decay_belief",
    "CollusionDetector",
    "PairReport",
    "CalibrationReport",
    "BinStats",
    "calibrate",
    "calibrate_interactions",
    "fidelity_run",
    "sweep_evidence_scale",
    "pit_values",
    "tail_reliability_bins",
    "AttackCategory",
    "AttackLibrary",
    "AttackScenario",
    "AttackResult",
    "VolumeForger",
    "ReputationFarmer",
    "run_attack",
    "GovernanceStack",
    "GovernanceLever",
    "LeverEffect",
    "LeverContext",
    "StakingLever",
    "TailCircuitBreaker",
    "RiskTaxLever",
    "AdaptiveController",
    "Proposal",
    "ProposalStatus",
    "ScenarioConfig",
    "load_scenario",
    "run_scenario",
    "__version__",
]
