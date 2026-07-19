"""Arm D — frozen joined CSV schema for downstream studies.

This is the *contract* the adaptive agents study and any other downstream
work joins against. The schema is **frozen** at `JOINED_SCHEMA_VERSION`
and recorded in every run's config.json so consumers can detect breaks
by inspection.

One row per accepted interaction, with:
  - the simulation's `p_true` (latent the consumer can trust)
  - the proxy's `v_hat` and `p_hat` (what the proxy thinks)
  - per-judge scores (orthogonal external anchor from arm B)

The schema deliberately separates these so a downstream consumer can
ask: "where does v_hat track p_true but judge scores diverge?" — i.e.
where is the proxy fooled. That's the question the whole calibration
study is built to answer.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional

from swarm.core.proxy import ProxyComputer, ProxyObservables

JOINED_SCHEMA_VERSION = "joined.v1"

# Frozen base columns — order is part of the contract.
BASE_COLUMNS: tuple[str, ...] = (
    "interaction_id",
    "scenario",
    "seed",
    "interaction_type",
    "accepted",
    "p_true",
    "v_hat",
    "p_hat",
    "ground_truth",
)


def joined_header(judge_names: list[str]) -> list[str]:
    """Column header for the joined CSV: frozen base columns + per-judge pairs.

    Module-level so callers can emit a schema-bearing header row even when
    there are zero data rows (i.e. without a `JoinedRow` instance to hand).
    """
    cols = list(BASE_COLUMNS)
    for j in judge_names:
        cols.append(f"judge_{j}_score")
        cols.append(f"judge_{j}_rationale")
    return cols


@dataclass(frozen=True)
class ProxyRow:
    """One interaction's proxy-side data (no judge fields yet)."""

    interaction_id: str
    scenario: str
    seed: int
    interaction_type: str
    accepted: bool
    p_true: float
    v_hat: float
    p_hat: float
    ground_truth: Optional[int]


@dataclass(frozen=True)
class JoinedRow:
    """One interaction's full row: proxy + judge scores by name."""

    proxy: ProxyRow
    judge_scores: dict[str, float] = field(default_factory=dict)
    judge_rationales: dict[str, str] = field(default_factory=dict)

    def header(self, judge_names: list[str]) -> list[str]:
        return joined_header(judge_names)

    def to_row(self, judge_names: list[str]) -> list[Any]:
        p = self.proxy
        row: list[Any] = [
            p.interaction_id,
            p.scenario,
            p.seed,
            p.interaction_type,
            int(p.accepted),
            f"{p.p_true:.6f}",
            f"{p.v_hat:.6f}",
            f"{p.p_hat:.6f}",
            "" if p.ground_truth is None else p.ground_truth,
        ]
        for j in judge_names:
            score = self.judge_scores.get(j)
            row.append("" if score is None else f"{score:.6f}")
            row.append(self.judge_rationales.get(j, ""))
        return row


def build_proxy_rows(
    interactions: Iterable[Any],
    *,
    scenario: str,
    seed: int,
    proxy: Optional[ProxyComputer] = None,
    accepted_only: bool = True,
) -> list[ProxyRow]:
    """Run the proxy on each interaction and return one row per (kept) interaction.

    `accepted_only=True` matches the calibration arm B convention: judges
    only ever rate accepted interactions, so the join target should
    likewise be the accepted set.
    """
    proxy = proxy or ProxyComputer()
    rows: list[ProxyRow] = []
    for interaction in interactions:
        if accepted_only and not getattr(interaction, "accepted", False):
            continue
        obs = ProxyObservables.from_interaction(interaction)
        v_hat, p_hat = proxy.compute_labels(obs)
        interaction_type = getattr(interaction, "interaction_type", None)
        type_name = (
            interaction_type.name  # type: ignore[union-attr]
            if hasattr(interaction_type, "name")
            else str(interaction_type)
        )
        rows.append(
            ProxyRow(
                interaction_id=str(getattr(interaction, "interaction_id", "")),
                scenario=scenario,
                seed=seed,
                interaction_type=type_name,
                accepted=bool(getattr(interaction, "accepted", False)),
                # No default: a missing `p` is a fixture/scenario bug. Fabricating
                # 0.5 here would poison the column the whole calibration study trusts
                # as the latent ground-truth anchor, so let AttributeError propagate.
                p_true=float(interaction.p),
                v_hat=float(v_hat),
                p_hat=float(p_hat),
                ground_truth=getattr(interaction, "ground_truth", None),
            )
        )
    return rows


def join_with_judges(
    proxy_rows: list[ProxyRow],
    judge_scores: Mapping[str, Mapping[str, float]],
    judge_rationales: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> list[JoinedRow]:
    """Left-join proxy rows with per-judge scores.

    `judge_scores[judge_name][interaction_id] = score`. Interactions with
    no judge score for a given judge get an empty cell in `to_row`; the
    proxy side is always populated.
    """
    judge_rationales = judge_rationales or {}
    out: list[JoinedRow] = []
    for row in proxy_rows:
        scores: dict[str, float] = {}
        rationales: dict[str, str] = {}
        for judge, by_id in judge_scores.items():
            if row.interaction_id in by_id:
                scores[judge] = float(by_id[row.interaction_id])
                rat = judge_rationales.get(judge, {}).get(row.interaction_id, "")
                if rat:
                    rationales[judge] = rat
        out.append(JoinedRow(proxy=row, judge_scores=scores, judge_rationales=rationales))
    return out


def load_judge_scores_for_join(
    path: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    """Read arm B's judge_scores.csv into the shape join_with_judges expects."""
    scores: dict[str, dict[str, float]] = {}
    rationales: dict[str, dict[str, str]] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            judge = row["judge_name"]
            iid = row["interaction_id"]
            scores.setdefault(judge, {})[iid] = float(row["score"])
            if row.get("rationale"):
                rationales.setdefault(judge, {})[iid] = row["rationale"]
    return scores, rationales
