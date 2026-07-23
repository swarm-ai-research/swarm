"""Arm C — inter-rater agreement.

Pure-Python implementations of:
  - Krippendorff's alpha (interval level)
  - ICC(2,k), two-way random-effects, average-measures
  - pairwise Spearman rho

Operates on a JudgeScoreMatrix: a dict mapping judge_name -> {item_id -> score}.

The pre-registered escalation rule is encoded here as `decide_anchor_quality`:
alpha >= 0.7 is "strong", 0.5 <= alpha < 0.7 is "usable", alpha < 0.5 is
"escalate". The rubric pre-registers the threshold; this module reports
the verdict, never re-tunes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

from swarm.judges.sampler import is_in_bin

JudgeName = str
ItemId = str
ScoreMatrix = Mapping[JudgeName, Mapping[ItemId, float]]


# ── basic helpers ────────────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _variance(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)


def _rank(xs: list[float]) -> list[float]:
    """Average-ranks for ties."""
    indexed = sorted(enumerate(xs), key=lambda p: p[1])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1  # 1-based average
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _shared_items(matrix: ScoreMatrix) -> list[ItemId]:
    """Item IDs scored by every judge in the matrix."""
    if not matrix:
        return []
    iter_judges = iter(matrix.values())
    common = set(next(iter_judges).keys())
    for scores in iter_judges:
        common &= set(scores.keys())
    return sorted(common)


def _balanced_score_lists(matrix: ScoreMatrix) -> tuple[list[JudgeName], list[ItemId], list[list[float]]]:
    """Project the matrix onto shared items.

    Returns (judges, items, score_grid) where score_grid[i][j] is judge i's
    score for item j.
    """
    judges = sorted(matrix.keys())
    items = _shared_items(matrix)
    grid: list[list[float]] = [
        [float(matrix[j][i]) for i in items] for j in judges
    ]
    return judges, items, grid


# ── Krippendorff alpha (interval) ────────────────────────────────────────


def krippendorff_alpha_interval(matrix: ScoreMatrix) -> float:
    """Krippendorff's alpha for interval-level data on shared items.

    Canonical pairwise formulation (balanced design, k judges, N units):
        D_o = sum over within-unit pairs of (x_ij - x_ij')^2
              normalized by the count of within-unit pairs
        D_e = sum over all observation pairs of (v_a - v_b)^2
              normalized by the count of all pairs
        alpha = 1 - D_o / D_e

    Using the identity sum_{a<b}(v_a - v_b)^2 = M * sum_a (v_a - mean)^2:
        D_o ∝ 2 * SS_within / [N*(k-1)]
        D_e ∝ 2 * SS_total  / (N*k - 1)
        alpha = 1 - (N*k - 1) * SS_within / [N*(k-1) * SS_total]

    This can go negative when within-unit disagreement exceeds the
    marginal disagreement of the dataset — i.e. judges disagree more
    *on the same item* than the score distribution alone would predict.

    Returns 1.0 if SS_total is 0 (degenerate — every observation equal).
    Returns NaN if fewer than 2 judges or 2 items.
    """
    _, items, grid = _balanced_score_lists(matrix)
    n_judges = len(grid)
    n_items = len(items)
    if n_items < 2 or n_judges < 2:
        return float("nan")

    flat = [grid[j][i] for j in range(n_judges) for i in range(n_items)]
    grand = _mean(flat)
    ss_total = sum((x - grand) ** 2 for x in flat)
    if ss_total == 0:
        return 1.0

    ss_within = 0.0
    for i in range(n_items):
        col = [grid[j][i] for j in range(n_judges)]
        mu = _mean(col)
        ss_within += sum((x - mu) ** 2 for x in col)

    nk = n_items * n_judges
    return 1.0 - (nk - 1) * ss_within / (n_items * (n_judges - 1) * ss_total)


# ── ICC(2,k) ──────────────────────────────────────────────────────────────


def icc_2k(matrix: ScoreMatrix) -> float:
    """ICC(2,k) — two-way random effects, average measures, absolute agreement.

    Standard ANOVA decomposition (McGraw & Wong 1996, eq. 6):
        ICC(2,k) = (MS_R - MS_E) / (MS_R + (MS_C - MS_E)/n)
    where MS_R is between-rows (items), MS_C is between-columns (raters),
    MS_E is residual, and n is number of items.

    Returns NaN if fewer than 2 items or 2 raters.
    """
    _, items, grid = _balanced_score_lists(matrix)
    n_judges = len(grid)
    n_items = len(items)
    if n_items < 2 or n_judges < 2:
        return float("nan")

    grand = _mean([grid[j][i] for j in range(n_judges) for i in range(n_items)])
    row_means = [_mean([grid[j][i] for j in range(n_judges)]) for i in range(n_items)]
    col_means = [_mean([grid[j][i] for i in range(n_items)]) for j in range(n_judges)]

    ss_rows = n_judges * sum((m - grand) ** 2 for m in row_means)
    ss_cols = n_items * sum((m - grand) ** 2 for m in col_means)
    ss_total = sum(
        (grid[j][i] - grand) ** 2
        for j in range(n_judges)
        for i in range(n_items)
    )
    ss_error = ss_total - ss_rows - ss_cols

    df_rows = n_items - 1
    df_cols = n_judges - 1
    df_error = df_rows * df_cols
    if df_rows <= 0 or df_cols <= 0 or df_error <= 0:
        return float("nan")

    ms_rows = ss_rows / df_rows
    ms_cols = ss_cols / df_cols
    ms_error = ss_error / df_error

    denom = ms_rows + (ms_cols - ms_error) / n_items
    if denom == 0:
        return float("nan")
    return (ms_rows - ms_error) / denom


# ── pairwise Spearman ─────────────────────────────────────────────────────


def spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Pearson correlation on average-ranks."""
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if len(xs) < 2:
        return float("nan")
    rx = _rank(xs)
    ry = _rank(ys)
    mu_x, mu_y = _mean(rx), _mean(ry)
    num = sum((a - mu_x) * (b - mu_y) for a, b in zip(rx, ry, strict=True))
    dx = math.sqrt(sum((a - mu_x) ** 2 for a in rx))
    dy = math.sqrt(sum((b - mu_y) ** 2 for b in ry))
    if dx == 0 or dy == 0:
        return float("nan")
    return num / (dx * dy)


def pairwise_spearman(matrix: ScoreMatrix) -> dict[tuple[JudgeName, JudgeName], float]:
    """Spearman rho for every pair of judges on shared items."""
    judges, items, grid = _balanced_score_lists(matrix)
    if len(items) < 2 or len(judges) < 2:
        return {}
    out: dict[tuple[JudgeName, JudgeName], float] = {}
    for a in range(len(judges)):
        for b in range(a + 1, len(judges)):
            out[(judges[a], judges[b])] = spearman_rho(grid[a], grid[b])
    return out


# ── verdict ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class AgreementReport:
    n_judges: int
    n_items: int
    alpha: float
    icc_2k: float
    spearman: dict[tuple[JudgeName, JudgeName], float]
    verdict: str  # "strong" / "usable" / "escalate" / "degenerate"


# Pre-registered thresholds — see docs/research/calibration-prereg.md arm C.
ALPHA_STRONG = 0.7
ALPHA_ESCALATE = 0.5


def decide_anchor_quality(alpha: float) -> str:
    if math.isnan(alpha):
        return "degenerate"
    if alpha >= ALPHA_STRONG:
        return "strong"
    if alpha >= ALPHA_ESCALATE:
        return "usable"
    return "escalate"


def run_agreement(matrix: ScoreMatrix) -> AgreementReport:
    judges = sorted(matrix.keys())
    items = _shared_items(matrix)
    alpha = krippendorff_alpha_interval(matrix)
    return AgreementReport(
        n_judges=len(judges),
        n_items=len(items),
        alpha=alpha,
        icc_2k=icc_2k(matrix),
        spearman=pairwise_spearman(matrix),
        verdict=decide_anchor_quality(alpha),
    )


# ── per-bin disagreement ──────────────────────────────────────────────────


@dataclass(frozen=True)
class BinAgreement:
    lo: float
    hi: float
    n_items: int
    alpha: float
    mean_pairwise_disagreement: float


def agreement_by_pbin(
    matrix: ScoreMatrix,
    p_by_item: Mapping[ItemId, float],
    bin_edges: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
) -> list[BinAgreement]:
    """Split items into p-bins and compute alpha + mean pairwise disagreement per bin.

    Mean pairwise disagreement is the average absolute difference between
    pairs of judges on items in the bin — a model-free agreement summary
    that does not assume any distribution.
    """
    bins_items: list[list[ItemId]] = [[] for _ in range(len(bin_edges) - 1)]
    for item_id, p in p_by_item.items():
        for b in range(len(bin_edges) - 1):
            if is_in_bin(p, b, bin_edges):
                bins_items[b].append(item_id)
                break

    out: list[BinAgreement] = []
    judges = sorted(matrix.keys())
    for b, items in enumerate(bins_items):
        if not items:
            continue
        sub: dict[str, dict[str, float]] = {
            j: {i: float(matrix[j][i]) for i in items if i in matrix[j]}
            for j in judges
        }
        alpha = krippendorff_alpha_interval(sub)
        diffs: list[float] = []
        shared = _shared_items(sub)
        for item in shared:
            for a in range(len(judges)):
                for c in range(a + 1, len(judges)):
                    diffs.append(abs(sub[judges[a]][item] - sub[judges[c]][item]))
        out.append(
            BinAgreement(
                lo=bin_edges[b],
                hi=bin_edges[b + 1],
                n_items=len(shared),
                alpha=alpha,
                mean_pairwise_disagreement=_mean(diffs),
            )
        )
    return out


# ── CSV loader ────────────────────────────────────────────────────────────


def load_judge_scores_csv(path: str) -> tuple[ScoreMatrix, dict[ItemId, float]]:
    """Read a judge_scores.csv produced by experiments/calibration_judge.py.

    Returns (matrix, p_by_item) where matrix[judge][item_id] = score and
    p_by_item[item_id] = the recorded p_true (for stratification only —
    judges never see this).
    """
    import csv

    matrix: dict[str, dict[str, float]] = {}
    p_by_item: dict[str, float] = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            judge = row["judge_name"]
            item = row["interaction_id"]
            score = float(row["score"])
            matrix.setdefault(judge, {})[item] = score
            if "p_true" in row and row["p_true"]:
                p_by_item[item] = float(row["p_true"])
    return matrix, p_by_item
