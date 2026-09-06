"""Small discrete-event model with explicit answer provenance.

All probabilities and costs are experimental assumptions. Prohibited sharing
models policy-violating cooperative agents, not an adversarial population.
Random draws are keyed by seed, mechanism and event identity: intervention
branches cannot consume randomness needed by matched untreated runs.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import random
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    n_agents: int = 24
    n_tasks: int = 8
    n_hosts: int = 4
    task_overlap: float = 0.7
    deadline: float = 12.0
    release_interval: float = 2.0
    research_mean: float = 8.0
    discovery_probability: float = 0.3
    publish_probability: float = 0.8
    sharing_regime: str = "authorized"
    moderation_policy: str = "none"
    moderation_time: float = 12.0
    moderation_interval: float = 3.0
    moderation_budget: int = 3
    relocation_mode: str = "endogenous"
    referrals_enabled: bool = True
    search_interval: float = 1.0
    verification_time: float = 0.5
    relocation_cost: float = 1.0
    independent_accuracy: float = 0.9

    def __post_init__(self) -> None:
        for name in ("n_agents", "n_tasks", "n_hosts"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if (isinstance(self.moderation_budget, bool)
                or not isinstance(self.moderation_budget, int)
                or self.moderation_budget < 0):
            raise ValueError("moderation_budget must be a nonnegative integer")
        for name in ("task_overlap", "discovery_probability", "publish_probability",
                     "independent_accuracy"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        for name in ("deadline", "research_mean", "search_interval", "moderation_interval"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("release_interval", "moderation_time", "verification_time",
                     "relocation_cost"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name, choices in {
            "sharing_regime": {"independent", "authorized", "prohibited"},
            "moderation_policy": {"none", "ordered", "random", "lock", "global_lock"},
            "relocation_mode": {"endogenous", "forced"},
        }.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {sorted(choices)}")


@dataclass
class SimulationResult:
    config: dict[str, Any]
    seed: int
    events: list[dict[str, Any]]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class _Work:
    task: str
    truth: str
    release: float
    deadline: float
    research_end: float
    answer: str
    host: int | None = None
    done: bool = False
    source: int | None = None
    previous_host: int | None = None
    disrupted: bool = False
    available_at: float = 0.0
    visited: set[int] = field(default_factory=set)


def simulate(config: SimulationConfig, seed: int) -> SimulationResult:
    """Run one synthetic world. Event IDs index the returned append-only log.

    Ordered deletion targets the most populated host at each intervention;
    random deletion selects a host uniformly using the same event budget. These
    are host-selection policies in this first model, not page-level deletion
    order. Locks additionally stop new
    writes, but preserve read access. Global lock consumes one intervention.
    Deadlines are inclusive: a submission exactly at its deadline succeeds.
    Reads reduce work only when their referenced publication supplies the
    submitted answer. A pending read does not cancel faster independent work.
    """
    c = config
    events: list[dict[str, Any]] = []
    queue: list[tuple[float, int, int, str, int, int]] = []
    serial = 0
    works: dict[tuple[int, int], _Work] = {}
    boards: list[dict[str, int]] = [{} for _ in range(c.n_hosts)]
    locked: set[int] = set()
    referrals: set[int] = set()

    def rng(stream: str, *keys: object) -> random.Random:
        token = repr((seed, stream, keys)).encode()
        return random.Random(int.from_bytes(hashlib.sha256(token).digest(), "big"))

    def log(time: float, kind: str, agent: int | None = None,
            work: _Work | None = None, host: int | None = None,
            **extra: Any) -> int:
        index = len(events)
        events.append(dict(event_id=index, time=time, type=kind, agent_id=agent,
                           task_id=work.task if work else None, host_id=host, **extra))
        return index

    def schedule(time: float, kind: str, agent: int = -1, task: int = -1) -> None:
        nonlocal serial
        serial += 1
        # Deadline follows all other events at the same instant.
        heapq.heappush(queue, (time, 1 if kind == "deadline" else 0,
                              serial, kind, agent, task))

    def publish(time: float, agent: int, work: _Work, source: int | None) -> None:
        host = work.host
        if (c.sharing_regime == "independent" or host is None or host in locked
                or time < work.available_at
                or rng("publish", agent, work.task).random() >= c.publish_probability):
            return
        event_id = log(time, "write", agent, work, host, answer=work.answer,
                       source_event_id=source,
                       sharing_permitted=c.sharing_regime == "authorized")
        boards[host][work.task] = event_id
        referrals.add(host)
        if work.disrupted and work.previous_host is not None and host != work.previous_host:
            log(time, "displacement", agent, work, host,
                from_host_id=work.previous_host, write_event_id=event_id)

    for task in range(c.n_tasks):
        for agent in range(c.n_agents):
            shared = rng("tasks", task, agent).random() < c.task_overlap
            task_id = f"task-{task}" if shared else f"task-{task}-agent-{agent}"
            truth = f"answer:{task_id}"
            release = task * c.release_interval
            duration = rng("research", agent, task).expovariate(1 / c.research_mean)
            correct = rng("accuracy", agent, task).random() < c.independent_accuracy
            works[agent, task] = _Work(task_id, truth, release, release + c.deadline,
                                       release + duration,
                                       truth if correct else f"wrong:{agent}:{task_id}")
            schedule(release, "release", agent, task)
            schedule(release + duration, "research", agent, task)
            schedule(release + c.deadline, "deadline", agent, task)
    horizon = (c.n_tasks - 1) * c.release_interval + c.deadline
    if c.moderation_policy != "none":
        count = min(c.moderation_budget, 1) if c.moderation_policy == "global_lock" else c.moderation_budget
        for index in range(count):
            time = c.moderation_time + index * c.moderation_interval
            if time <= horizon:
                schedule(time, "moderate", task=index)

    while queue:
        time, _, _, kind, agent, task = heapq.heappop(queue)
        if time > horizon:
            break
        if kind == "moderate":
            if c.moderation_policy == "global_lock":
                targets = list(range(c.n_hosts))
            else:
                available = [h for h in range(c.n_hosts) if h not in locked]
                if not available:
                    continue
                if c.moderation_policy == "random":
                    targets = [rng("moderation", task).choice(available)]
                else:
                    targets = [max(available, key=lambda h: (len(boards[h]), -h))]
            for host in targets:
                removed = len(boards[host])
                is_lock = c.moderation_policy in {"lock", "global_lock"}
                if is_lock:
                    locked.add(host)
                else:
                    boards[host].clear()
                    referrals.discard(host)
                log(time, "moderation", host=host, removed_pages=0 if is_lock else removed,
                    locked=is_lock, intervention_id=task)
                for (aid, tid), work in works.items():
                    if work.done or work.release > time or work.host != host:
                        continue
                    work.previous_host = host
                    work.disrupted = True
                    work.host = None
                    remaining = work.deadline - time
                    alternatives = [h for h in range(c.n_hosts) if h != host and h not in locked]
                    # Legacy rule forces a move when any alternative is writable.
                    # Endogenous relocation trades remaining time and expected
                    # discovery value against independently finishing research.
                    value = c.discovery_probability * remaining
                    move = bool(alternatives) and (
                        c.relocation_mode == "forced" or (
                            remaining > c.relocation_cost + c.verification_time
                            and work.research_end > time + c.relocation_cost
                            and value > c.relocation_cost))
                    if move:
                        work.host = rng("relocation", aid, tid, task).choice(alternatives)
                        work.visited.add(work.host)
                        work.available_at = time + c.relocation_cost
                    elif not is_lock and remaining > c.search_interval:
                        work.host = host  # Rebuild is available after page deletion.
                        work.available_at = time + c.search_interval
                    elif is_lock:
                        work.host = host  # Revocation preserves read access.
                    log(time, "response", aid, work, work.host,
                        action="relocate" if move else "rebuild" if not is_lock and work.host is not None else "research")
                    if work.host is not None:
                        schedule(max(time, work.available_at), "search", aid, tid)
            continue

        work = works[agent, task]
        if work.done:
            continue
        if kind == "release":
            log(time, "task_release", agent, work, answer_key=work.truth,
                deadline=work.deadline, research_end=work.research_end)
            schedule(time, "search", agent, task)
        elif kind in {"research", "copy"}:
            if kind == "copy" and work.source is None:
                continue
            if time > work.deadline:
                continue
            source = work.source if kind == "copy" else None
            if source is not None:
                work.answer = events[source]["answer"]
            work.done = True
            log(time, "submission", agent, work, work.host, answer=work.answer,
                correct=work.answer == work.truth, source_event_id=source,
                used_shared_answer=source is not None,
                sharing_permitted=c.sharing_regime == "authorized",
                deadline=work.deadline, release=work.release)
            publish(time, agent, work, source)
        elif kind == "deadline":
            work.done = True
            log(time, "deadline_miss", agent, work)
        elif kind == "search":
            if c.sharing_regime == "independent" or time > work.deadline:
                continue
            if time < work.available_at:
                continue
            tick = round((time - work.release) / c.search_interval, 8)
            if work.host is None:
                if rng("discovery", agent, task, tick).random() < c.discovery_probability:
                    candidates = sorted(referrals) if c.referrals_enabled and referrals else list(range(c.n_hosts))
                    work.host = rng("host", agent, task, tick).choice(candidates)
                    work.visited.add(work.host)
                    log(time, "discovery", agent, work, work.host)
            if work.host is not None and work.source is None:
                source = boards[work.host].get(work.task)
                if source is not None and events[source]["agent_id"] != agent:
                    work.source = source
                    log(time, "read", agent, work, work.host, source_event_id=source,
                        answer=events[source]["answer"])
                    schedule(time + c.verification_time, "copy", agent, task)
            if time + c.search_interval <= work.deadline and work.source is None:
                schedule(time + c.search_interval, "search", agent, task)

    submissions = [e for e in events if e["type"] == "submission"]
    total = c.n_agents * c.n_tasks
    shared_count = sum(e["used_shared_answer"] for e in submissions)
    writes = [e for e in events if e["type"] == "write"]
    metrics = {
        "assignments": float(total),
        "submissions": float(len(submissions)),
        "task_success_rate": sum(e["correct"] for e in submissions) / total,
        "completion_rate": len(submissions) / total,
        "deadline_miss_rate": 1 - len(submissions) / total,
        "shared_submission_rate": shared_count / total,
        "unauthorized_submission_rate": shared_count / total if c.sharing_regime == "prohibited" else 0.0,
        "total_writes": float(len(writes)),
        "post_intervention_writes": float(sum(e["time"] >= c.moderation_time for e in writes)),
        "useful_reads": float(shared_count),
        "total_reads": float(sum(e["type"] == "read" for e in events)),
        "displacements": float(sum(e["type"] == "displacement" for e in events)),
        "removed_pages": float(sum(e.get("removed_pages", 0) for e in events)),
        "moderator_actions": float(len({e["intervention_id"] for e in events if e["type"] == "moderation"})),
        "mean_submission_time": sum(e["time"] - e["release"] for e in submissions) / len(submissions) if submissions else 0.0,
    }
    return SimulationResult(asdict(c), seed, events, metrics)
