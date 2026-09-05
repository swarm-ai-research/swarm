"""Gossip-board search model (bead fcj5).

The world is a separable objective over ``n_dims`` discrete dimensions with
``n_values`` settings each, plus one *hidden* dimension whose gain is large
but whose prior mass (probability an agent thinks to mutate it) is small.
That is the position-encoding blind spot from the Hyperspace retrospective,
exposed as ``hidden_dim_prior`` for the prior-diversity sweep (bead khs2).

Each round every active agent picks one move:

    baseline  reset to the baseline config (rare; models a fresh start)
    peer      read the most recent board entries and adopt one, at the
              fidelity the board allows
    mutate    change one dimension of its own best config

Evaluation is noisy. An agent keeps a config when the *observed* score beats
its best observed score, and publishes it to the board only then unless
``publish_failures`` is set. Every published entry carries a ground-truth
``parent`` (the entry it was derived from, or ``None``), which is the
lineage field the real board lacked.
"""

from __future__ import annotations

import copy
import itertools
import json
import random
import statistics
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import yaml

FIDELITY_MODES = ("code", "description", "score_only")


@dataclass
class BoardConfig:
    scenario_id: str = "gossip_board_fidelity"
    seed: int = 42
    # population
    n_agents: int = 24
    n_rounds: int = 120
    late_joiner_fraction: float = 0.25  # agents that join at late_join_round
    late_join_round: int = 60
    # search space
    n_dims: int = 8
    n_values: int = 5
    hidden_dim_gain: float = 3.0  # gain of the hidden dimension relative to others
    hidden_dim_prior: float = 0.02  # prior mass on mutating the hidden dimension
    diverse_fraction: float = 0.0  # share of agents with a UNIFORM prior over dims (bead khs2)
    eval_noise: float = 0.15
    # move policy
    p_baseline: float = 0.02
    p_peer: float = 0.4
    board_window: int = 20  # entries a reader considers (most recent)
    # THE lever
    fidelity: str = "code"
    description_error: float = 0.0  # chance a description is misread (value off by one)
    # publication rule
    publish_failures: bool = False
    frontier_threshold: float = 0.9  # fraction of optimum counted as "at frontier"

    @classmethod
    def from_yaml(cls, path: Path) -> "BoardConfig":
        doc = yaml.safe_load(Path(path).read_text()) or {}
        cfg = cls(
            scenario_id=str(doc.get("scenario_id", cls.scenario_id)),
            seed=int(doc.get("seed", cls.seed)),
        )
        for k, v in (doc.get("board") or {}).items():
            if not hasattr(cfg, k):
                raise KeyError(f"unknown board field: {k}")
            setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.fidelity not in FIDELITY_MODES:
            raise ValueError(f"fidelity must be one of {FIDELITY_MODES}, got {self.fidelity!r}")
        if not 0.0 <= self.hidden_dim_prior <= 1.0:
            raise ValueError("hidden_dim_prior must be in [0, 1]")
        if not 0.0 <= self.diverse_fraction <= 1.0:
            raise ValueError("diverse_fraction must be in [0, 1]")
        if not 0.0 <= self.p_peer + self.p_baseline <= 1.0:
            raise ValueError("p_peer + p_baseline must be in [0, 1]")


@dataclass
class Entry:
    """One board row. ``parent`` is the ground-truth lineage field."""

    id: int
    agent: int
    round: int
    score: float  # observed (self-graded) score
    config: Tuple[int, ...]
    diff: Optional[Tuple[int, int]]  # (dim, value) that produced it, if a mutation
    parent: Optional[int]  # entry id this was derived from
    success: bool


@dataclass
class Agent:
    id: int
    join_round: int
    config: Tuple[int, ...]
    best_observed: float
    derived_from: Optional[int] = None  # board entry the current config came from
    adoptions: int = 0  # times this agent took a peer result
    rediscoveries: int = 0  # own mutations that matched an existing board diff
    first_frontier_round: Optional[int] = None
    diverse: bool = False  # uniform prior over dimensions instead of the shared one


@dataclass
class RoundMetrics:
    round: int
    active_agents: int
    mean_true_score: float
    frontier_fraction: float
    modal_cluster: int  # agents sharing the most common config
    identical_pairs: float  # fraction of agent pairs with identical configs
    hidden_dim_adoption: float  # fraction of agents at the hidden dim's best value
    board_size: int


@dataclass
class RunResult:
    config: BoardConfig
    rounds: List[RoundMetrics]
    summary: Dict[str, Any]
    board: List[Entry] = field(default_factory=list)

    def to_json(self) -> Dict[str, Any]:
        return {
            "config": asdict(self.config),
            "rounds": [asdict(r) for r in self.rounds],
            "summary": self.summary,
            "board": [asdict(e) for e in self.board],
        }


class Landscape:
    """Separable objective with one high-gain, low-prior hidden dimension."""

    def __init__(self, cfg: BoardConfig, rng: random.Random) -> None:
        self.n_dims = cfg.n_dims
        self.n_values = cfg.n_values
        self.hidden_dim = cfg.n_dims - 1
        self.table: List[List[float]] = []
        for d in range(cfg.n_dims):
            gain = cfg.hidden_dim_gain if d == self.hidden_dim else 1.0
            vals = [rng.random() * gain for _ in range(cfg.n_values)]
            self.table.append(vals)
        self.optimum = sum(max(v) for v in self.table)
        self.baseline = tuple(0 for _ in range(cfg.n_dims))
        self.best_hidden_value = max(
            range(cfg.n_values), key=lambda v: self.table[self.hidden_dim][v]
        )

    def score(self, config: Sequence[int]) -> float:
        return sum(self.table[d][v] for d, v in enumerate(config))


def _dim_prior(cfg: BoardConfig) -> List[float]:
    other = (1.0 - cfg.hidden_dim_prior) / max(cfg.n_dims - 1, 1)
    return [other] * (cfg.n_dims - 1) + [cfg.hidden_dim_prior]


def run_board(cfg: BoardConfig, seed: Optional[int] = None) -> RunResult:
    """Run one board for ``cfg.n_rounds`` and return per-round metrics + summary."""
    cfg.validate()
    rng = random.Random(cfg.seed if seed is None else seed)
    land = Landscape(cfg, rng)
    prior = _dim_prior(cfg)
    uniform = [1.0 / cfg.n_dims] * cfg.n_dims
    dims = list(range(cfg.n_dims))
    n_diverse = int(round(cfg.n_agents * cfg.diverse_fraction))

    n_late = int(round(cfg.n_agents * cfg.late_joiner_fraction))
    agents: List[Agent] = []
    for i in range(cfg.n_agents):
        join = cfg.late_join_round if i >= cfg.n_agents - n_late else 0
        agents.append(Agent(i, join, land.baseline, float("-inf"),
                            diverse=(i % cfg.n_agents) < n_diverse))

    board: List[Entry] = []
    attempts_all: List[float] = []  # true score of every evaluated attempt
    attempts_by_round: Dict[int, List[float]] = {}
    published_by_round: Dict[int, List[float]] = {}
    mutated_dims: set[int] = set()
    hidden_discovery_round: Optional[int] = None  # first mutation of the hidden dim
    diff_seen: Dict[Tuple[int, int], int] = {}  # first entry id that published a diff
    rounds: List[RoundMetrics] = []
    next_id = itertools.count()

    def observe(config: Tuple[int, ...]) -> float:
        return land.score(config) + rng.gauss(0.0, cfg.eval_noise)

    def publish(agent: Agent, config: Tuple[int, ...], observed: float,
                diff: Optional[Tuple[int, int]], parent: Optional[int],
                success: bool) -> Entry:
        e = Entry(next(next_id), agent.id, r, observed, config, diff, parent, success)
        board.append(e)
        if success and diff is not None and diff not in diff_seen:
            diff_seen[diff] = e.id
        return e

    for r in range(cfg.n_rounds):
        active = [a for a in agents if a.join_round <= r]
        for a in active:
            # A fresh agent evaluates its baseline once so it has a best_observed.
            if a.best_observed == float("-inf"):
                a.best_observed = observe(a.config)
                attempts_all.append(land.score(a.config))

            u = rng.random()
            recent = [e for e in board if e.success and e.agent != a.id][-cfg.board_window:]
            if u < cfg.p_baseline:
                cand, diff, parent, via_peer = land.baseline, None, None, False
            elif u < cfg.p_baseline + cfg.p_peer and recent:
                e = max(recent, key=lambda x: x.score)  # readers chase the leaderboard
                via_peer = True
                if cfg.fidelity == "code":
                    cand, diff, parent = e.config, e.diff, e.id
                elif cfg.fidelity == "description" and e.diff is not None:
                    d, v = e.diff
                    if rng.random() < cfg.description_error:
                        v = (v + rng.choice((-1, 1))) % cfg.n_values
                    c = list(a.config)
                    c[d] = v
                    cand, diff, parent = tuple(c), (d, v), e.id
                else:  # score_only, or a description with nothing to describe
                    d = rng.choices(dims, uniform if a.diverse else prior)[0]
                    c = list(a.config)
                    c[d] = rng.randrange(cfg.n_values)
                    cand, diff, parent, via_peer = tuple(c), (d, c[d]), None, False
            else:
                d = rng.choices(dims, uniform if a.diverse else prior)[0]
                c = list(a.config)
                c[d] = rng.randrange(cfg.n_values)
                cand, diff, parent, via_peer = tuple(c), (d, c[d]), None, False

            if diff is not None and not via_peer:
                mutated_dims.add(diff[0])
                if diff[0] == land.hidden_dim and hidden_discovery_round is None:
                    hidden_discovery_round = r
            observed = observe(cand)
            attempts_all.append(land.score(cand))
            attempts_by_round.setdefault(r, []).append(land.score(cand))
            success = observed > a.best_observed
            if success or cfg.publish_failures:
                published_by_round.setdefault(r, []).append(land.score(cand))
            if success:
                a.config, a.best_observed = cand, observed
                if via_peer:
                    a.adoptions += 1
                    a.derived_from = parent
                else:
                    if diff is not None and diff in diff_seen:
                        a.rediscoveries += 1
                    a.derived_from = None
                if (a.first_frontier_round is None
                        and land.score(cand) >= cfg.frontier_threshold * land.optimum):
                    a.first_frontier_round = r
                publish(a, cand, observed, diff, parent, True)
            elif cfg.publish_failures:
                publish(a, cand, observed, diff, parent, False)

        # per-round metrics
        configs = [a.config for a in active]
        counts = Counter(configs)
        modal = max(counts.values()) if counts else 0
        n = len(configs)
        pairs = n * (n - 1) / 2
        same = sum(c * (c - 1) / 2 for c in counts.values())
        true_scores = [land.score(c) for c in configs]
        rounds.append(RoundMetrics(
            round=r,
            active_agents=n,
            mean_true_score=statistics.fmean(true_scores) / land.optimum if n else 0.0,
            frontier_fraction=(sum(s >= cfg.frontier_threshold * land.optimum
                                   for s in true_scores) / n) if n else 0.0,
            modal_cluster=modal,
            identical_pairs=(same / pairs) if pairs else 0.0,
            hidden_dim_adoption=(sum(c[land.hidden_dim] == land.best_hidden_value
                                     for c in configs) / n) if n else 0.0,
            board_size=len(board),
        ))

    published_true = [land.score(e.config) for e in board]
    # Survivorship gap, contemporaneous: what a reader of the board sees in a
    # round minus what actually happened that round. Pooling across rounds
    # confounds the gap with search progress (successes cluster early).
    # With publish_failures the board shows every attempt and the gap is 0.
    round_gaps = [
        statistics.fmean(published_by_round[k]) - statistics.fmean(attempts_by_round[k])
        for k in published_by_round if k in attempts_by_round
    ]
    early = [a for a in agents if a.join_round == 0]
    late = [a for a in agents if a.join_round > 0]

    def _ttf(group: List[Agent]) -> Optional[float]:
        vals = [a.first_frontier_round - a.join_round for a in group
                if a.first_frontier_round is not None]
        return statistics.fmean(vals) if vals else None

    final = rounds[-1]
    summary: Dict[str, Any] = {
        "fidelity": cfg.fidelity,
        "optimum": land.optimum,
        "final_mean_true_score": final.mean_true_score,
        "final_frontier_fraction": final.frontier_fraction,
        "final_modal_cluster": final.modal_cluster,
        "final_modal_cluster_fraction": final.modal_cluster / max(final.active_agents, 1),
        "final_identical_pairs": final.identical_pairs,
        "hidden_dim_adoption": final.hidden_dim_adoption,
        "hidden_dim_explored": land.hidden_dim in mutated_dims,
        "hidden_discovery_round": hidden_discovery_round,  # None = never in n_rounds
        "hidden_discovery_round_or_max": (cfg.n_rounds if hidden_discovery_round is None
                                          else hidden_discovery_round),
        "diverse_agents": n_diverse,
        "unexplored_dims": cfg.n_dims - len(mutated_dims),
        "board_entries": len(board),
        "board_success_entries": sum(e.success for e in board),
        "attempts": len(attempts_all),
        "published_fraction": (len(board) / len(attempts_all)) if attempts_all else 0.0,
        # survivorship gap: what the board says minus what actually happened
        "survivorship_gap": (statistics.fmean(round_gaps) / land.optimum) if round_gaps else 0.0,
        "survivorship_gap_pooled": ((statistics.fmean(published_true)
                                     - statistics.fmean(attempts_all))
                                    / land.optimum) if published_true and attempts_all else 0.0,
        "adoptions": sum(a.adoptions for a in agents),
        "rediscoveries": sum(a.rediscoveries for a in agents),
        "time_to_frontier_early": _ttf(early),
        "time_to_frontier_late": _ttf(late),
        "late_frontier_fraction": (sum(a.first_frontier_round is not None for a in late)
                                   / len(late)) if late else None,
    }
    return RunResult(cfg, rounds, summary, board)


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, ValueError):
        return raw


def sweep_board(cfg: BoardConfig, axes: Dict[str, List[Any]], seeds: Sequence[int]
                ) -> List[Dict[str, Any]]:
    """Cross product of ``axes`` x ``seeds``; one summary row per cell."""
    rows: List[Dict[str, Any]] = []
    keys = list(axes)
    for combo in itertools.product(*(axes[k] for k in keys)):
        for s in seeds:
            c = copy.deepcopy(cfg)
            for k, v in zip(keys, combo, strict=False):
                if not hasattr(c, k):
                    raise KeyError(f"unknown board field: {k}")
                setattr(c, k, v)
            res = run_board(c, seed=s)
            row = dict(zip(keys, combo, strict=False))
            row["seed"] = s
            row.update(res.summary)
            rows.append(row)
    return rows


def aggregate(rows: List[Dict[str, Any]], keys: List[str]) -> List[Dict[str, Any]]:
    """Mean over seeds of every numeric summary field, grouped by ``keys``."""
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(row[k] for k in keys), []).append(row)
    out: List[Dict[str, Any]] = []
    for g, members in groups.items():
        agg: Dict[str, Any] = dict(zip(keys, g, strict=False))
        agg["n_seeds"] = len(members)
        for f_name in members[0]:
            if f_name in keys or f_name == "seed":
                continue
            raw = [m[f_name] for m in members]
            nums = [v for v in raw if isinstance(v, (int, float)) and not isinstance(v, bool)]
            if nums:
                agg[f_name] = statistics.fmean(nums)  # None values (never) are skipped
            elif all(isinstance(v, bool) for v in raw):
                agg[f_name] = sum(raw) / len(raw)
            else:
                agg[f_name] = None
        out.append(agg)
    return out
