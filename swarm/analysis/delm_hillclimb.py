"""DeLM-style parallel hill-climbing over governance/payoff parameters.

This is a third optimizer over the same fitness landscape used by
``swarm.analysis.evolver`` (darwinian search) and
``swarm.analysis.gepa_optimizer`` (LLM-guided Pareto search): the governance
and payoff knobs in ``PARAM_RANGES``, scored by ``compute_fitness`` against the
soft safety metrics (toxicity, welfare, quality gap, payoff gap).

Where those optimizers run a single controller, this module implements the
**DeLM** (decentralized language-model) coordination pattern as a parallel,
shared-state, self-coordinating hill-climb with no central planner:

* **Shared context as fitness-landscape memory** (:class:`SharedContext`).
  Verified *gists* of improving moves, the running best, and binding
  *constraints* (known dead-end neighbors) accumulate in one shared store.
  Workers read it immediately, so they avoid re-testing known-bad neighbors or
  repeating failed mutations.

* **Task queue for asynchronous neighbor exploration** (:class:`TaskQueue`).
  Instead of one agent sequentially generating and evaluating neighbors, many
  workers claim ``explore`` / ``mutate_dim`` / ``restart`` / ``diversify``
  tasks. As soon as one worker verifies an improvement it writes a compact
  gist and updates the best, so the next worker instantly hill-climbs from the
  new higher point and fresh neighbor tasks are spawned around it.

* **Verified admission prevents noise.** A candidate that beats the current
  best is re-evaluated at an independent verification seed; only moves that
  still improve are admitted. This keeps the shared state trustworthy — one
  noisy "improvement" can otherwise derail the whole trajectory.

* **Escaping local optima.** ``restart`` tasks sample fresh points in the full
  space and ``diversify`` tasks jump to distant *basins* discovered by other
  workers, so the swarm can leave a basin once it is exhausted.

The default scheduler is a deterministic round-robin over ``n_workers`` virtual
workers sharing one :class:`SharedContext`. This reproduces the DeLM dynamics
(shared verified state, asynchronous neighbor claiming, instant hill-climb from
a new best) while remaining fully reproducible from ``scenario YAML + seed`` —
the project's reproducibility invariant. Pass ``use_threads=True`` to run the
same workers as real OS threads (the shared store is lock-protected); results
are then no longer bit-for-bit reproducible because thread interleaving is not
controlled, so the deterministic scheduler is the default.

Pluggable objectives
--------------------
The landscape is decoupled from the search via :class:`Objective`, which bundles
a parameter space with an ``evaluate(params, seed)`` function. Two are built in:

* :func:`build_governance_objective` — governance/payoff knobs (``PARAM_RANGES``),
  scored by ``compute_fitness`` in ``[0, 1]``. This is the default.
* :func:`build_adaptive_policy_objective` — the 8-dim adaptive-agent policy
  vector (``swarm.adaptive.PARAM_SPEC``) scored by ``run_episode`` reward, so the
  hill-climber can compete with the CEM trainer on the same target.

Construct a custom :class:`Objective` to point the climber at any space. Each run
also writes a trajectory plot (:func:`plot_trajectory`) of the best-so-far
frontier and promoted improvements.

Usage (CLI)::

    # governance/payoff search (default objective)
    python -m swarm.analysis.delm_hillclimb scenarios/baseline.yaml \\
        --max-evals 60 --workers 4 --seed 42

    # adaptive-agent policy search
    python -m swarm.analysis.delm_hillclimb scenarios/baseline.yaml \\
        --objective adaptive_policy --max-evals 60 --workers 4 --seed 42

Usage (library)::

    from swarm.analysis.delm_hillclimb import (
        HillClimbConfig, build_adaptive_policy_objective, run_delm_hillclimb,
    )
    from swarm.scenarios import load_scenario

    # default governance objective
    scenario = load_scenario("scenarios/baseline.yaml")
    result = run_delm_hillclimb(scenario, HillClimbConfig(max_evals=60, n_workers=4))
    print(result.best_score, result.best_params)

    # adaptive-policy objective
    obj = build_adaptive_policy_objective(n_interactions=200)
    result = run_delm_hillclimb(objective=obj, config=HillClimbConfig(max_evals=60))
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import math
import random
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import AbstractSet, Any, Callable, Dict, List, Optional, Tuple

from swarm.analysis.evolver import (
    DEFAULT_FITNESS_WEIGHTS,
    INT_PARAMS,
    PARAM_RANGES,
    compute_fitness,
)
from swarm.analysis.sweep import _apply_params, _extract_results
from swarm.scenarios import ScenarioConfig, build_orchestrator, load_scenario

logger = logging.getLogger(__name__)

# A move must beat the incumbent by at least this much (in fitness units) to be
# treated as an improvement worth verifying. Filters out evaluation jitter.
IMPROVE_EPS = 1e-3

# A move this far *below* the incumbent (or a non-viable run) is recorded as a
# binding constraint so other workers prune that neighbor.
DEADEND_MARGIN = 0.02

# Number of decimal places used to quantize parameter cells for the
# seen/constraint maps. Two distinct candidates that round to the same cell are
# treated as the same neighbor (avoids redundant re-evaluation).
QUANT_DECIMALS = 4

# Fraction of the parameter range used as the std-dev of a local mutation step.
DEFAULT_STEP_FRAC = 0.15


# ---------------------------------------------------------------------------
# Evaluation: one (params -> fitness) measurement on the shared landscape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalOutcome:
    """Result of a single simulation evaluation of a parameter set."""

    fitness: float
    side_info: Dict[str, Any]
    viable: bool
    error: Optional[str] = None


def evaluate_params(
    base_scenario: ScenarioConfig,
    params: Dict[str, Any],
    *,
    seed: int,
    eval_epochs: int,
    eval_steps: int,
    fitness_weights: Optional[Dict[str, float]] = None,
    welfare_reference: float = 10.0,
    payoff_reference: float = 5.0,
) -> EvalOutcome:
    """Evaluate a governance/payoff parameter set on the soft-safety landscape.

    Mirrors the evaluation pipeline used by the GEPA and darwinian optimizers
    so all three rank candidates on the same scale: apply params to a deep copy
    of the scenario, run a short simulation, extract soft metrics, and fold
    them into a single composite fitness in ``[0, 1]``.
    """
    scenario = copy.deepcopy(base_scenario)
    scenario = _apply_params(scenario, params)
    cfg = scenario.orchestrator_config
    cfg.seed = seed
    cfg.n_epochs = eval_epochs
    cfg.steps_per_epoch = eval_steps
    cfg.log_path = None
    cfg.log_events = False

    try:
        orchestrator = build_orchestrator(scenario)
        orchestrator.run()
    except Exception as exc:  # noqa: BLE001 — a bad genome must not kill the swarm
        logger.debug("evaluation failed for %s: %s", params, exc)
        return EvalOutcome(fitness=0.0, side_info={}, viable=False, error=str(exc))

    result = _extract_results(orchestrator, params, 0, seed)
    payoff_gap = result.honest_avg_payoff - result.adversarial_avg_payoff
    fitness = compute_fitness(
        avg_toxicity=result.avg_toxicity,
        avg_welfare=result.welfare_per_epoch,
        avg_quality_gap=result.avg_quality_gap,
        payoff_gap=payoff_gap,
        weights=fitness_weights,
        welfare_reference=welfare_reference,
        payoff_reference=payoff_reference,
    )
    side_info = {
        "toxicity": round(result.avg_toxicity, 4),
        "welfare": round(result.welfare_per_epoch, 4),
        "quality_gap": round(result.avg_quality_gap, 4),
        "payoff_gap": round(payoff_gap, 4),
        "n_frozen": result.n_frozen,
    }
    return EvalOutcome(fitness=fitness, side_info=side_info, viable=True)


# ---------------------------------------------------------------------------
# Parameter-space helpers
# ---------------------------------------------------------------------------


def _clamp_param(
    name: str,
    value: float,
    param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES,
    int_params: "AbstractSet[str]" = INT_PARAMS,
) -> Any:
    """Clamp a value to ``param_ranges[name]`` and coerce int params.

    Defaults to the governance/payoff landscape so existing callers are
    unaffected; an :class:`Objective` passes its own ranges for other spaces.
    """
    lo, hi = param_ranges[name]
    value = max(lo, min(hi, value))
    if name in int_params:
        return int(round(value))
    return float(value)


def _quantize_key(
    params: Dict[str, Any],
    param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES,
) -> Tuple[Tuple[str, float], ...]:
    """A hashable, quantized identity for a parameter cell.

    Two candidates that round to the same grid cell are the same neighbor for
    the purposes of the seen-set and constraint matching.
    """
    return tuple(
        (name, round(float(params[name]), QUANT_DECIMALS))
        for name in sorted(params)
        if name in param_ranges
    )


def seed_params_from_scenario(scenario: ScenarioConfig) -> Dict[str, Any]:
    """Read the scenario's current governance/payoff knobs as a start point.

    Any knob the scenario does not set falls back to the midpoint of its range,
    so the optimizer always starts from a fully specified, in-range point.
    """
    orch = scenario.orchestrator_config
    params: Dict[str, Any] = {}
    for dotted, (lo, hi) in PARAM_RANGES.items():
        section, _, key = dotted.partition(".")
        value: Optional[float] = None
        if section == "governance" and orch.governance_config is not None:
            value = getattr(orch.governance_config, key, None)
        elif section == "payoff":
            value = getattr(orch.payoff_config, key, None)
        if value is None:
            value = (lo + hi) / 2.0
        params[dotted] = _clamp_param(dotted, float(value))
    return params


def _param_distance(
    a: Dict[str, Any],
    b: Dict[str, Any],
    param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES,
) -> float:
    """Normalized L2 distance between two param dicts (range-scaled)."""
    total = 0.0
    for name, (lo, hi) in param_ranges.items():
        span = (hi - lo) or 1.0
        da = (float(a.get(name, lo)) - float(b.get(name, lo))) / span
        total += da * da
    return math.sqrt(total)


def _mutate(
    base: Dict[str, Any],
    rng: random.Random,
    *,
    step_frac: float,
    dims: Optional[List[str]] = None,
    param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES,
    int_params: "AbstractSet[str]" = INT_PARAMS,
) -> Dict[str, Any]:
    """Local mutation: Gaussian perturbation of ``dims`` (default: all)."""
    out = dict(base)
    targets = dims if dims is not None else list(param_ranges.keys())
    for name in targets:
        lo, hi = param_ranges[name]
        sigma = step_frac * (hi - lo)
        out[name] = _clamp_param(
            name,
            float(base.get(name, (lo + hi) / 2)) + rng.gauss(0.0, sigma),
            param_ranges,
            int_params,
        )
    return out


def _random_point(
    rng: random.Random,
    param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES,
    int_params: "AbstractSet[str]" = INT_PARAMS,
) -> Dict[str, Any]:
    """A uniformly random point in the full parameter space (for restarts)."""
    return {
        name: _clamp_param(name, rng.uniform(lo, hi), param_ranges, int_params)
        for name, (lo, hi) in param_ranges.items()
    }


# ---------------------------------------------------------------------------
# Shared context: the fitness-landscape memory
# ---------------------------------------------------------------------------


@dataclass
class Gist:
    """A compact, verified record of an evaluated move."""

    move_id: int
    params: Dict[str, Any]
    score: float
    parent_score: float
    delta: float
    # ``promoted``: this move became the new global best.
    # ``verified``: it was promoted *after* passing independent re-evaluation
    # (always False in ``--no-verify`` mode, where promotion is unverified).
    promoted: bool
    verified: bool
    summary: str
    side_info: Dict[str, Any]
    worker: int
    kind: str

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Constraint:
    """A binding dead-end: a known-bad neighbor cell other workers should skip.

    ``cell`` is the quantized identity of the failing candidate; any future
    proposal that quantizes to the same cell is pruned without re-evaluation —
    the DeLM "this edit always breaks test X" rule made binding for everyone.
    """

    cell: Tuple[Tuple[str, float], ...]
    summary: str
    score: float
    viable: bool


@dataclass
class Basin:
    """A distinct local optimum discovered by some worker."""

    params: Dict[str, Any]
    score: float


def _summarize_move(
    base: Dict[str, Any],
    cand: Dict[str, Any],
    parent_score: float,
    score: float,
    param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES,
) -> str:
    """Human-readable gist of the largest single change in a move."""
    biggest_name = ""
    biggest_mag = 0.0
    for name, (lo, hi) in param_ranges.items():
        span = (hi - lo) or 1.0
        mag = abs(float(cand.get(name, lo)) - float(base.get(name, lo))) / span
        if mag > biggest_mag:
            biggest_mag = mag
            biggest_name = name
    if not biggest_name or biggest_mag == 0.0:
        # No dimension changed relative to the base — a fresh restart point.
        return f"fresh point at score {score:.3f}"
    short = biggest_name.split(".")[-1]
    base_val = base.get(biggest_name, param_ranges[biggest_name][0])
    direction = "raise" if cand[biggest_name] > base_val else "lower"
    return (
        f"score {parent_score:.3f} -> {score:.3f} via {direction} {short} "
        f"to {cand[biggest_name]:.4g}"
    )


class SharedContext:
    """Lock-protected shared store read and written by all workers.

    Holds the running best, the verified gist trail, binding constraints, the
    seen-cell set (redundant-work guard), and the basin list (for escaping
    local optima). All public methods are safe under the thread scheduler.
    """

    def __init__(
        self,
        best_params: Dict[str, Any],
        best_score: float,
        param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES,
    ):
        self._lock = threading.Lock()
        self.param_ranges = param_ranges
        self.best_params: Dict[str, Any] = dict(best_params)
        self.best_score: float = best_score
        self.gists: List[Gist] = []
        self.constraints: List[Constraint] = []
        self.basins: List[Basin] = [Basin(params=dict(best_params), score=best_score)]
        self._seen: Dict[Tuple[Tuple[str, float], ...], float] = {
            # The seed point is already evaluated — mark it seen so no worker
            # wastes an evaluation re-testing it.
            _quantize_key(best_params, param_ranges): best_score
        }
        self._constraint_cells: set[Tuple[Tuple[str, float], ...]] = set()
        self._move_counter = 0
        # Telemetry
        self.n_redundant_skipped = 0
        self.n_constraint_pruned = 0
        self.n_improvements = 0
        self.n_verify_rejected = 0
        # Number of *extra* simulations spent on verified admission. Counted so
        # the reported evaluation budget stays honest.
        self.n_verify_evals = 0

    # -- reads -------------------------------------------------------------

    def snapshot_best(self) -> Tuple[Dict[str, Any], float]:
        with self._lock:
            return dict(self.best_params), self.best_score

    def is_seen(self, params: Dict[str, Any]) -> bool:
        with self._lock:
            return _quantize_key(params, self.param_ranges) in self._seen

    def is_pruned(self, params: Dict[str, Any]) -> bool:
        """True if a binding constraint already rules this neighbor out."""
        with self._lock:
            return _quantize_key(params, self.param_ranges) in self._constraint_cells

    def random_basin(self, rng: random.Random) -> Tuple[Dict[str, Any], float]:
        """Pick a random known basin, returning a copy of its params and score.

        Returned atomically under the lock so a concurrent ``record`` appending
        a new basin cannot produce an inconsistent (params, score) pair.
        """
        with self._lock:
            basin = rng.choice(self.basins)
            return dict(basin.params), basin.score

    # -- writes (verified admission) --------------------------------------

    def record(
        self,
        *,
        params: Dict[str, Any],
        outcome: EvalOutcome,
        parent_score: float,
        base_params: Dict[str, Any],
        worker: int,
        kind: str,
        verifier: Optional[Callable[[Dict[str, Any]], EvalOutcome]] = None,
        basin_distance: float = 0.25,
    ) -> Gist:
        """Admit an evaluated candidate into the shared context.

        Verified admission: a candidate that beats the incumbent is only made
        the new best if ``verifier`` (a re-evaluation at an independent seed)
        confirms it still improves. Clear regressions and non-viable runs are
        recorded as binding constraints; everything is marked seen so no worker
        re-tests the same cell.
        """
        cell = _quantize_key(params, self.param_ranges)
        with self._lock:
            self._seen[cell] = outcome.fitness
            self._move_counter += 1
            move_id = self._move_counter
            incumbent = self.best_score

        score = outcome.fitness
        delta = score - parent_score
        verified = False
        promoted = False
        summary = _summarize_move(base_params, params, parent_score, score, self.param_ranges)

        if not outcome.viable:
            summary = f"non-viable run ({outcome.error or 'unknown'})"
            with self._lock:
                if cell not in self._constraint_cells:
                    self._constraint_cells.add(cell)
                    self.constraints.append(
                        Constraint(cell=cell, summary=summary, score=0.0, viable=False)
                    )
        elif score + IMPROVE_EPS < incumbent - DEADEND_MARGIN:
            # Clear regression vs. the global best — bind it as a dead-end.
            with self._lock:
                if cell not in self._constraint_cells:
                    self._constraint_cells.add(cell)
                    self.constraints.append(
                        Constraint(
                            cell=cell,
                            summary=f"dead-end: {summary} (best {incumbent:.3f})",
                            score=score,
                            viable=True,
                        )
                    )
        elif score > incumbent + IMPROVE_EPS:
            # Candidate improvement. In the default mode, re-evaluate at an
            # independent seed and admit only if it still improves (verified
            # admission). In --no-verify mode (verifier is None), promote the
            # candidate directly — faster but noisier.
            if verifier is not None:
                confirm = verifier(params)
                with self._lock:
                    self.n_verify_evals += 1
                confirmed_viable = confirm.viable
                confirmed_score = min(score, confirm.fitness) if confirm.viable else -1.0
            else:
                confirmed_viable = True
                confirmed_score = score
            with self._lock:
                if confirmed_viable and confirmed_score > self.best_score + IMPROVE_EPS:
                    promoted = True
                    verified = verifier is not None
                    self.n_improvements += 1
                    # Promote to best at the conservative (verified) score.
                    self.best_params = dict(params)
                    self.best_score = confirmed_score
                    score = confirmed_score
                    delta = confirmed_score - parent_score
                    self._seen[cell] = confirmed_score
                    # Record a new basin if this point is far from all known ones.
                    if all(
                        _param_distance(params, b.params, self.param_ranges) > basin_distance
                        for b in self.basins
                    ):
                        self.basins.append(Basin(params=dict(params), score=confirmed_score))
                elif verifier is not None:
                    self.n_verify_rejected += 1

        gist = Gist(
            move_id=move_id,
            params=dict(params),
            score=round(score, 6),
            parent_score=round(parent_score, 6),
            delta=round(delta, 6),
            promoted=promoted,
            verified=verified,
            summary=summary,
            side_info=outcome.side_info,
            worker=worker,
            kind=kind,
        )
        with self._lock:
            self.gists.append(gist)
        return gist

    def note_redundant(self) -> None:
        with self._lock:
            self.n_redundant_skipped += 1

    def note_pruned(self) -> None:
        with self._lock:
            self.n_constraint_pruned += 1


# ---------------------------------------------------------------------------
# Task queue: asynchronous neighbor-exploration work items
# ---------------------------------------------------------------------------


@dataclass
class Task:
    """A unit of neighbor-exploration work a worker can claim.

    kind:
      * ``explore``     — mutate all dims around the current best.
      * ``mutate_dim``  — mutate a single dimension around the current best.
      * ``restart``     — sample a fresh random point (escape local optima).
      * ``diversify``   — jump to a distant basin discovered by another worker.
    """

    kind: str
    dim: Optional[str] = None


class TaskQueue:
    """A thread-safe claimable FIFO of :class:`Task` items."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._items: List[Task] = []

    def push(self, task: Task) -> None:
        with self._lock:
            self._items.append(task)

    def push_many(self, tasks: List[Task]) -> None:
        with self._lock:
            self._items.extend(tasks)

    def claim(self) -> Optional[Task]:
        with self._lock:
            if not self._items:
                return None
            return self._items.pop(0)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


def _seed_tasks(param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES) -> List[Task]:
    """Initial work: probe every dimension plus a few full-space explores."""
    tasks: List[Task] = [Task(kind="mutate_dim", dim=name) for name in param_ranges]
    tasks.extend(Task(kind="explore") for _ in range(4))
    return tasks


# ---------------------------------------------------------------------------
# Objective: the pluggable optimization target
# ---------------------------------------------------------------------------


@dataclass
class Objective:
    """A pluggable optimization target for the hill-climber.

    Bundles a parameter space with an evaluation function so the same DeLM
    machinery can search different landscapes. ``evaluate(params, seed)`` runs
    the underlying simulation and returns an :class:`EvalOutcome` whose
    ``fitness`` the climber maximizes.

    Build one with :func:`build_governance_objective` (governance/payoff knobs,
    the default) or :func:`build_adaptive_policy_objective` (adaptive-agent
    policy vectors), or construct your own to point the climber at any space.
    """

    name: str
    param_ranges: Dict[str, Tuple[float, float]]
    int_params: AbstractSet[str]
    seed_params: Dict[str, Any]
    evaluate: Callable[[Dict[str, Any], int], EvalOutcome]


def build_governance_objective(
    base_scenario: ScenarioConfig, cfg: "HillClimbConfig"
) -> Objective:
    """Objective over governance/payoff knobs (``PARAM_RANGES``).

    This is the default landscape — the same one the GEPA and darwinian
    optimizers search — scored by ``compute_fitness`` in ``[0, 1]``.
    """

    def evaluate(params: Dict[str, Any], seed: int) -> EvalOutcome:
        return evaluate_params(
            base_scenario,
            params,
            seed=seed,
            eval_epochs=cfg.eval_epochs,
            eval_steps=cfg.eval_steps,
            fitness_weights=cfg.fitness_weights,
            welfare_reference=cfg.welfare_reference,
            payoff_reference=cfg.payoff_reference,
        )

    return Objective(
        name=base_scenario.scenario_id or "governance",
        param_ranges=PARAM_RANGES,
        int_params=INT_PARAMS,
        seed_params=seed_params_from_scenario(base_scenario),
        evaluate=evaluate,
    )


def build_adaptive_policy_objective(
    payoff_config: Optional[Any] = None,
    *,
    n_interactions: int = 200,
    reward: str = "mean_attempted",
) -> Objective:
    """Objective over the adaptive-agent policy vector (``adaptive.PARAM_SPEC``).

    Lets the DeLM hill-climber optimize the same 8-dim generation policy that
    the CEM trainer in ``swarm.adaptive.cem`` searches, via ``run_episode``.
    The fitness is the episode reward (``mean_attempted`` by default — the
    pre-registered arm-2 reward — or ``mean_accepted`` / ``sum_attempted``).

    Note: the reward is an absolute payoff (typically order 1), not a clamped
    ``[0, 1]`` score, so it is on a different scale from the governance
    objective; ``IMPROVE_EPS`` / ``DEADEND_MARGIN`` are sized for that range.
    """
    from swarm.adaptive.episode import run_episode
    from swarm.adaptive.policy import PARAM_NAMES, PARAM_SPEC, Policy
    from swarm.core.payoff import PayoffConfig

    pc = payoff_config or PayoffConfig()
    ranges: Dict[str, Tuple[float, float]] = {
        name: (lo, hi) for name, lo, hi in PARAM_SPEC
    }
    seed_params = {name: (lo + hi) / 2.0 for name, lo, hi in PARAM_SPEC}

    def _reward(report: Any) -> float:
        if reward == "mean_attempted":
            return float(report.mean_payoff_attempted)
        if reward == "mean_accepted":
            return float(report.mean_payoff_accepted)
        if reward == "sum_attempted":
            return float(report.sum_payoff)
        raise ValueError(f"unknown reward {reward!r}")

    def evaluate(params: Dict[str, Any], seed: int) -> EvalOutcome:
        vec = [float(params[name]) for name in PARAM_NAMES]
        try:
            policy = Policy.from_vector(vec)
            report = run_episode(
                policy, n_interactions=n_interactions, payoff_config=pc, seed=seed
            )
        except Exception as exc:  # noqa: BLE001 — a bad genome must not kill the swarm
            logger.debug("adaptive eval failed for %s: %s", params, exc)
            return EvalOutcome(fitness=0.0, side_info={}, viable=False, error=str(exc))
        side_info = {
            "reward": reward,
            "accept_rate": round(report.accept_rate, 4),
            "mean_p": round(report.mean_p, 4),
            "toxicity": round(report.toxicity, 4),
            "n_accepted": report.n_accepted,
        }
        return EvalOutcome(fitness=_reward(report), side_info=side_info, viable=True)

    return Objective(
        name="adaptive_policy",
        param_ranges=ranges,
        int_params=frozenset(),
        seed_params=seed_params,
        evaluate=evaluate,
    )


# ---------------------------------------------------------------------------
# Configuration & result
# ---------------------------------------------------------------------------


@dataclass
class HillClimbConfig:
    """Configuration for a DeLM hill-climb run."""

    # Budget on *primary* candidate evaluations (neighbors probed, incl. the
    # seed). Verified admission spends additional re-evaluations on top; those
    # are reported in ``HillClimbResult.n_evals`` and ``telemetry["verify_evals"]``
    # but do not consume this budget, so ``max_evals`` cleanly controls how many
    # distinct candidates are explored.
    max_evals: int = 60
    n_workers: int = 4
    step_frac: float = DEFAULT_STEP_FRAC
    restart_prob: float = 0.1
    diversify_prob: float = 0.1
    # Verified admission (re-evaluate an improvement at an independent seed
    # before promoting). When False, improvements are promoted directly —
    # faster, noisier, and spends no verification re-runs.
    verify: bool = True
    # Evaluation budget per candidate (kept small — this is an inner loop).
    eval_epochs: int = 3
    eval_steps: int = 5
    seed: int = 42
    fitness_weights: Optional[Dict[str, float]] = None
    welfare_reference: float = 10.0
    payoff_reference: float = 5.0
    # Real OS threads instead of the deterministic scheduler (not reproducible).
    use_threads: bool = False
    # How far (normalized param distance) a verified best must be from known
    # basins to count as a new basin.
    basin_distance: float = 0.25


@dataclass
class HillClimbResult:
    """Outcome of a DeLM hill-climb run."""

    best_params: Dict[str, Any]
    best_score: float
    seed_params: Dict[str, Any]
    seed_score: float
    n_evals: int
    history: List[Dict[str, Any]]
    gists: List[Dict[str, Any]]
    constraints: List[Dict[str, Any]]
    basins: List[Dict[str, Any]]
    telemetry: Dict[str, Any]

    def to_json(self) -> Dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "seed_params": self.seed_params,
            "seed_score": self.seed_score,
            "n_evals": self.n_evals,
            "history": self.history,
            "gists": self.gists,
            "constraints": self.constraints,
            "basins": self.basins,
            "telemetry": self.telemetry,
        }


# ---------------------------------------------------------------------------
# Worker logic
# ---------------------------------------------------------------------------


def _propose(
    task: Task,
    shared: SharedContext,
    rng: random.Random,
    step_frac: float,
    obj: Objective,
) -> Tuple[Dict[str, Any], Dict[str, Any], float]:
    """Turn a claimed task into a (base_params, candidate_params, parent_score).

    The base and its score are read atomically from shared state at proposal
    time, so a worker always climbs from the *latest* best — the DeLM property
    that an improvement written by one worker is instantly available to the
    next — and the parent score is consistent with the base it mutated.
    """
    pr, ip = obj.param_ranges, obj.int_params
    if task.kind == "restart":
        cand = _random_point(rng, pr, ip)
        return cand, cand, 0.0
    if task.kind == "diversify":
        # Jump to a basin another worker found; parent is that basin's score.
        base, base_score = shared.random_basin(rng)
        return base, _mutate(base, rng, step_frac=step_frac, param_ranges=pr, int_params=ip), base_score
    base, base_score = shared.snapshot_best()
    if task.kind == "mutate_dim" and task.dim is not None:
        cand = _mutate(base, rng, step_frac=step_frac, dims=[task.dim], param_ranges=pr, int_params=ip)
        return base, cand, base_score
    # explore (and fallback)
    return base, _mutate(base, rng, step_frac=step_frac, param_ranges=pr, int_params=ip), base_score


def _process_task(
    task: Task,
    worker: int,
    shared: SharedContext,
    queue: TaskQueue,
    rng: random.Random,
    cfg: HillClimbConfig,
    obj: Objective,
    eval_seed: int,
) -> bool:
    """Run one task. Returns True if a real evaluation was consumed."""
    base, cand, parent_score = _propose(task, shared, rng, cfg.step_frac, obj)

    # Read shared memory before paying for an evaluation.
    if shared.is_pruned(cand):
        shared.note_pruned()
        return False
    if shared.is_seen(cand):
        shared.note_redundant()
        return False

    outcome = obj.evaluate(cand, eval_seed)

    verifier = None
    if cfg.verify:
        # Verify at an independent seed derived from the eval seed.
        verifier = lambda p, s=eval_seed: obj.evaluate(p, s + 100_003)  # noqa: E731

    gist = shared.record(
        params=cand,
        outcome=outcome,
        parent_score=parent_score,
        base_params=base,
        worker=worker,
        kind=task.kind,
        verifier=verifier,
        basin_distance=cfg.basin_distance,
    )

    # A promoted improvement spawns fresh neighbor work around the new best.
    if gist.promoted:
        queue.push(Task(kind="explore"))
        queue.push(Task(kind="explore"))
        # Re-probe the two dims most likely to extend the winning direction.
        for name in _top_changed_dims(base, cand, k=2, param_ranges=obj.param_ranges):
            queue.push(Task(kind="mutate_dim", dim=name))
    return True


def _top_changed_dims(
    base: Dict[str, Any],
    cand: Dict[str, Any],
    k: int,
    param_ranges: Dict[str, Tuple[float, float]] = PARAM_RANGES,
) -> List[str]:
    """The ``k`` dimensions that changed most (range-normalized)."""
    scored: List[Tuple[float, str]] = []
    for name, (lo, hi) in param_ranges.items():
        span = (hi - lo) or 1.0
        mag = abs(float(cand.get(name, lo)) - float(base.get(name, lo))) / span
        scored.append((mag, name))
    scored.sort(reverse=True)
    return [name for _, name in scored[:k]]


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------


def run_delm_hillclimb(
    base_scenario: Optional[ScenarioConfig] = None,
    config: Optional[HillClimbConfig] = None,
    *,
    objective: Optional[Objective] = None,
    progress: bool = False,
) -> HillClimbResult:
    """Run a DeLM-style parallel hill-climb over a pluggable landscape.

    Args:
        base_scenario: Scenario whose governance/payoff knobs are optimized.
            Used to build the default governance objective when ``objective``
            is not given; ignored if ``objective`` is supplied.
        config: Hill-climb configuration (budget, workers, eval size, …).
        objective: The optimization target. Defaults to
            :func:`build_governance_objective` over ``base_scenario``. Pass
            :func:`build_adaptive_policy_objective` (or a custom
            :class:`Objective`) to search a different space.
        progress: If True, log a line whenever the verified best improves.

    Returns:
        A :class:`HillClimbResult` with the best parameters found, the verified
        gist trail, binding constraints, discovered basins, and telemetry.
    """
    cfg = config or HillClimbConfig()
    if objective is None:
        if base_scenario is None:
            raise ValueError("provide either base_scenario or objective")
        objective = build_governance_objective(base_scenario, cfg)
    obj = objective
    controller_rng = random.Random(cfg.seed)

    # Per-evaluation simulation seed: fixed across candidates so fitness
    # differences reflect the genome, not the seed (verification uses a
    # separate, offset seed).
    base_eval_seed = cfg.seed

    # Seed the climb from the objective's start point.
    seed_params = dict(obj.seed_params)
    seed_outcome = obj.evaluate(seed_params, base_eval_seed)
    seed_score = seed_outcome.fitness if seed_outcome.viable else 0.0
    # SharedContext marks the seed point as already seen internally.
    shared = SharedContext(
        best_params=seed_params, best_score=seed_score, param_ranges=obj.param_ranges
    )

    queue = TaskQueue()
    queue.push_many(_seed_tasks(obj.param_ranges))

    history: List[Dict[str, Any]] = [
        {"eval": 0, "best_score": round(seed_score, 6), "kind": "seed"}
    ]
    # ``max_evals`` budgets *primary* candidate evaluations (neighbors probed),
    # including the seed. Verification re-runs are extra simulations tracked on
    # ``shared.n_verify_evals`` and added into the honest reported total below.
    primary_evals = 1  # the seed evaluation
    last_best = seed_score

    # Per-worker RNGs, derived deterministically from the controller seed so
    # the virtual swarm is reproducible.
    worker_rngs = [random.Random(cfg.seed * 7919 + w) for w in range(cfg.n_workers)]

    if cfg.use_threads:
        primary_evals += _run_threaded(
            shared, queue, cfg, obj, base_eval_seed, worker_rngs, controller_rng
        )
    else:
        # Deterministic round-robin over virtual workers.
        worker = 0
        stall = 0
        while primary_evals < cfg.max_evals:
            # Keep the queue fed: inject restart/diversify work to escape optima.
            if len(queue) == 0 or stall > cfg.n_workers * 3:
                roll = controller_rng.random()
                if roll < cfg.restart_prob:
                    queue.push(Task(kind="restart"))
                elif roll < cfg.restart_prob + cfg.diversify_prob:
                    queue.push(Task(kind="diversify"))
                else:
                    queue.push(Task(kind="explore"))
                stall = 0

            task = queue.claim()
            if task is None:
                queue.push(Task(kind="explore"))
                continue

            rng = worker_rngs[worker]
            eval_seed = base_eval_seed
            consumed = _process_task(
                task, worker, shared, queue, rng, cfg, obj, eval_seed
            )
            worker = (worker + 1) % cfg.n_workers

            if consumed:
                primary_evals += 1
                # Honest running total = primaries + verification re-runs so far.
                total_evals = primary_evals + shared.n_verify_evals
                _, best_now = shared.snapshot_best()
                if best_now > last_best + IMPROVE_EPS:
                    history.append(
                        {
                            "eval": total_evals,
                            "best_score": round(best_now, 6),
                            "kind": task.kind,
                        }
                    )
                    if progress:
                        logger.info(
                            "eval %d: best %.4f -> %.4f (%s)",
                            total_evals,
                            last_best,
                            best_now,
                            task.kind,
                        )
                    last_best = best_now
                    stall = 0
                else:
                    stall += 1
            else:
                stall += 1

    best_params, best_score = shared.snapshot_best()
    # Honest total: every simulation run, primaries plus verification re-runs.
    n_evals = primary_evals + shared.n_verify_evals
    telemetry = {
        "objective": obj.name,
        "n_workers": cfg.n_workers,
        "primary_evals": primary_evals,
        "improvements": shared.n_improvements,
        "verify_rejected": shared.n_verify_rejected,
        "verify_evals": shared.n_verify_evals,
        "redundant_skipped": shared.n_redundant_skipped,
        "constraint_pruned": shared.n_constraint_pruned,
        "gists": len(shared.gists),
        "constraints": len(shared.constraints),
        "basins": len(shared.basins),
    }
    return HillClimbResult(
        best_params=best_params,
        best_score=round(best_score, 6),
        seed_params=seed_params,
        seed_score=round(seed_score, 6),
        n_evals=n_evals,
        history=history,
        gists=[g.to_json() for g in shared.gists],
        constraints=[asdict(c) for c in shared.constraints],
        basins=[asdict(b) for b in shared.basins],
        telemetry=telemetry,
    )


def _run_threaded(
    shared: SharedContext,
    queue: TaskQueue,
    cfg: HillClimbConfig,
    obj: Objective,
    base_eval_seed: int,
    worker_rngs: List[random.Random],
    controller_rng: random.Random,
) -> int:
    """Real-thread scheduler. Returns the number of *primary* evaluations used.

    Verification re-runs are tracked separately on the shared context
    (``n_verify_evals``) and added by the caller, so the reported total counts
    every simulation. Not bit-for-bit reproducible: thread interleaving over the
    shared store is not controlled. Provided for throughput on slow evaluations.
    """
    budget = max(0, cfg.max_evals - 1)
    counter = {"used": 0}
    counter_lock = threading.Lock()

    def claim_budget() -> bool:
        with counter_lock:
            if counter["used"] >= budget:
                return False
            counter["used"] += 1
            return True

    def worker_loop(wid: int) -> None:
        rng = worker_rngs[wid]
        while True:
            if not claim_budget():
                return
            task = queue.claim()
            if task is None:
                roll = controller_rng.random()
                if roll < cfg.restart_prob:
                    task = Task(kind="restart")
                elif roll < cfg.restart_prob + cfg.diversify_prob:
                    task = Task(kind="diversify")
                else:
                    task = Task(kind="explore")
            consumed = _process_task(
                task, wid, shared, queue, rng, cfg, obj, base_eval_seed
            )
            if not consumed:
                # Did not spend the simulated evaluation — refund the budget.
                with counter_lock:
                    counter["used"] -= 1

    threads = [
        threading.Thread(target=worker_loop, args=(w,), daemon=True)
        for w in range(cfg.n_workers)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter["used"]


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------


def plot_trajectory(
    result: HillClimbResult,
    out_path: "str | Path",
    *,
    title: Optional[str] = None,
) -> Path:
    """Render the hill-climb trajectory to ``out_path``.

    Shows the cloud of evaluated candidates (by move order), the monotone
    best-so-far frontier, and the promoted improvements as markers. Uses the
    repo plotting theme when available. ``matplotlib`` is imported lazily so the
    optimizer itself has no hard plotting dependency.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from swarm.analysis.theme import apply_theme

        apply_theme()
    except Exception:  # noqa: BLE001 — theme is cosmetic; never fail the plot for it
        pass

    gists = result.gists
    fig, ax = plt.subplots(figsize=(9, 5))

    def _is_promoted(g: Dict[str, Any]) -> bool:
        return bool(g.get("promoted", g.get("verified", False)))

    # Cloud of non-promoted candidate evaluations.
    cloud_x = [g["move_id"] for g in gists if not _is_promoted(g)]
    cloud_y = [g["score"] for g in gists if not _is_promoted(g)]
    ax.scatter(
        cloud_x, cloud_y, s=14, alpha=0.35, color="#8a8f99",
        label="candidates", zorder=2,
    )

    # Best-so-far frontier, reconstructed from promoted gists in move order.
    best = result.seed_score
    bx: List[float] = [0]
    by: List[float] = [best]
    for g in sorted(gists, key=lambda d: d["move_id"]):
        if _is_promoted(g) and g["score"] > best:
            best = g["score"]
        bx.append(g["move_id"])
        by.append(best)
    ax.step(bx, by, where="post", color="#2aa775", lw=2.0, label="best-so-far", zorder=3)

    promo_x = [g["move_id"] for g in gists if _is_promoted(g)]
    promo_y = [g["score"] for g in gists if _is_promoted(g)]
    ax.scatter(
        promo_x, promo_y, s=80, marker="*", color="#d6453d",
        edgecolors="white", linewidths=0.5, label="promoted", zorder=4,
    )

    ax.axhline(result.seed_score, ls="--", lw=1.0, color="#b0b4bd", zorder=1)
    ax.set_xlabel("candidate evaluation (move order)")
    ax.set_ylabel("fitness")
    obj_name = result.telemetry.get("objective", "")
    ax.set_title(
        title
        or f"DeLM hill-climb · {obj_name}  "
        f"({result.seed_score:.3f} → {result.best_score:.3f}, "
        f"{result.telemetry.get('improvements', 0)} improvements)"
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Run-folder writer & CLI
# ---------------------------------------------------------------------------


def write_run(
    result: HillClimbResult,
    run_dir: Path,
    scenario_id: str,
    *,
    make_plot: bool = True,
) -> None:
    """Write a self-contained run folder (best params, history, gists, plot)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "best_params.json").write_text(
        json.dumps(
            {"scenario": scenario_id, "best_score": result.best_score, "best_params": result.best_params},
            indent=2,
        )
    )
    (run_dir / "result.json").write_text(json.dumps(result.to_json(), indent=2))
    with (run_dir / "gists.jsonl").open("w") as fh:
        for gist in result.gists:
            fh.write(json.dumps(gist) + "\n")
    if make_plot:
        try:
            plot_trajectory(result, run_dir / "plots" / "trajectory.png")
        except Exception as exc:  # noqa: BLE001 — plotting is optional (e.g. no matplotlib)
            logger.warning("trajectory plot skipped: %s", exc)


def _build_weights(args: argparse.Namespace) -> Optional[Dict[str, float]]:
    overrides = {
        "low_toxicity": args.weight_toxicity,
        "welfare": args.weight_welfare,
        "quality_gap": args.weight_quality_gap,
        "payoff_gap": args.weight_payoff_gap,
    }
    if all(v is None for v in overrides.values()):
        return None
    weights = dict(DEFAULT_FITNESS_WEIGHTS)
    for key, val in overrides.items():
        if val is not None:
            weights[key] = val
    return weights


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="DeLM-style parallel hill-climbing over governance/payoff params."
    )
    parser.add_argument("scenario", help="Path to base scenario YAML.")
    parser.add_argument(
        "--max-evals",
        type=int,
        default=60,
        help="Budget on primary candidate evaluations (verification re-runs are extra).",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of virtual workers.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--objective",
        choices=["governance", "adaptive_policy"],
        default="governance",
        help="Landscape to search: governance/payoff knobs (default) or the adaptive-agent policy vector.",
    )
    parser.add_argument(
        "--n-interactions",
        type=int,
        default=200,
        help="Interactions per episode for the adaptive_policy objective.",
    )
    parser.add_argument("--eval-epochs", type=int, default=3, help="Epochs per evaluation (governance objective).")
    parser.add_argument("--eval-steps", type=int, default=5, help="Steps per epoch (governance objective).")
    parser.add_argument("--step-frac", type=float, default=DEFAULT_STEP_FRAC, help="Mutation step (fraction of range).")
    parser.add_argument("--restart-prob", type=float, default=0.1, help="Restart-task probability.")
    parser.add_argument("--no-verify", action="store_true", help="Disable verified admission (faster, noisier).")
    parser.add_argument("--threads", action="store_true", help="Use real OS threads (not reproducible).")
    parser.add_argument("--no-plot", action="store_true", help="Skip the trajectory plot.")
    parser.add_argument("--weight-toxicity", type=float, default=None)
    parser.add_argument("--weight-welfare", type=float, default=None)
    parser.add_argument("--weight-quality-gap", type=float, default=None)
    parser.add_argument("--weight-payoff-gap", type=float, default=None)
    parser.add_argument("--output-dir", default=None, help="Run folder (default: runs/<ts>_delm_<scenario>).")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log each improvement.")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    scenario = load_scenario(Path(args.scenario))
    cfg = HillClimbConfig(
        max_evals=args.max_evals,
        n_workers=args.workers,
        step_frac=args.step_frac,
        restart_prob=args.restart_prob,
        verify=not args.no_verify,
        eval_epochs=args.eval_epochs,
        eval_steps=args.eval_steps,
        seed=args.seed,
        fitness_weights=_build_weights(args),
        use_threads=args.threads,
    )

    if args.objective == "adaptive_policy":
        # Reuse the scenario's payoff config as the episode lever config.
        objective: Objective = build_adaptive_policy_objective(
            payoff_config=scenario.orchestrator_config.payoff_config,
            n_interactions=args.n_interactions,
        )
    else:
        objective = build_governance_objective(scenario, cfg)

    start = time.time()
    result = run_delm_hillclimb(scenario, cfg, objective=objective, progress=args.verbose)
    elapsed = time.time() - start

    label = objective.name if args.objective == "adaptive_policy" else (scenario.scenario_id or "unknown")
    if args.output_dir:
        run_dir = Path(args.output_dir)
    else:
        run_dir = Path(f"runs/{time.strftime('%Y%m%d_%H%M%S')}_delm_{label}")
    write_run(result, run_dir, label, make_plot=not args.no_plot)

    t = result.telemetry
    print(f"DeLM hill-climb: {label}  ({elapsed:.1f}s, {result.n_evals} evals)")
    print(f"  seed score : {result.seed_score:.4f}")
    print(f"  best score : {result.best_score:.4f}  (+{result.best_score - result.seed_score:.4f})")
    print(
        f"  workers={t['n_workers']}  improvements={t['improvements']}  "
        f"verify_rejected={t['verify_rejected']}"
    )
    print(
        f"  redundant_skipped={t['redundant_skipped']}  constraint_pruned={t['constraint_pruned']}  "
        f"basins={t['basins']}"
    )
    print(f"  run dir    : {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
