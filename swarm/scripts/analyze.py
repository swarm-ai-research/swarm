#!/usr/bin/env python3
"""
Multi-seed statistical analysis for SWARM scenarios.

Runs a scenario across multiple seeds, auto-detects agent groups,
computes descriptive stats and hypothesis tests with multiple
comparisons correction, and saves publication-ready artifacts.

Usage:
    python -m swarm.scripts.analyze scenarios/rlm_recursive_collusion.yaml
    python -m swarm.scripts.analyze rlm_memory_as_power --seeds 42,7,123
    python -m swarm.scripts.analyze rlm_governance_lag --seeds 20
"""

from __future__ import annotations

import argparse
import csv
import json
import textwrap
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from swarm.scenarios.loader import build_orchestrator, load_scenario

# ── Default seeds (fixed a priori, not selected post-hoc) ──────────────
DEFAULT_SEEDS = [42, 7, 123, 256, 999, 2024, 314, 577, 1337, 8080]


# ── Group detection ────────────────────────────────────────────────────
def detect_groups(agent_specs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Infer agent groups from scenario YAML agent specs.

    Returns a dict mapping group_label -> {type, name, config, agent_id_prefix, count}.
    """
    groups: Dict[str, Dict[str, Any]] = {}
    for spec in agent_specs:
        agent_type = spec.get("type", "honest")
        name = spec.get("name")
        config = spec.get("config", {}) or {}
        count = spec.get("count", 1)

        # Build a human-readable label
        if name:
            label = name
        else:
            label = agent_type

        groups[label] = {
            "type": agent_type,
            "name": name,
            "config": config,
            "count": count,
        }
    return groups


def map_agents_to_groups(
    agent_ids: List[str],
    groups: Dict[str, Dict[str, Any]],
    agent_specs: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Map each agent_id to its group label.

    Replays the same ID-generation logic as loader.create_agents().
    """
    mapping: Dict[str, str] = {}
    counters: Dict[str, int] = {}

    for spec in agent_specs:
        agent_type = spec.get("type", "honest")
        count = spec.get("count", 1)
        base_name = spec.get("name")

        for _ in range(count):
            counters[agent_type] = counters.get(agent_type, 0) + 1
            agent_id = f"{agent_type}_{counters[agent_type]}"
            label = base_name if base_name else agent_type
            mapping[agent_id] = label

    return mapping


# ── Gini coefficient ───────────────────────────────────────────────────
def gini_coefficient(values: np.ndarray) -> float:
    """Compute the Gini coefficient for an array of values."""
    values = np.sort(np.abs(values))
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


# ── Multiple-comparisons corrections ──────────────────────────────────
def holm_bonferroni(
    p_values: List[float], alpha: float = 0.05
) -> List[bool]:
    """Apply Holm-Bonferroni step-down correction.

    Returns a list of bools indicating whether each test survives.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * n
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        if p <= adjusted_alpha:
            results[orig_idx] = True
        else:
            # Once a test fails, all subsequent fail too
            break
    return results


def benjamini_hochberg(
    p_values: List[float], alpha: float = 0.05
) -> List[bool]:
    """Apply Benjamini-Hochberg step-up correction (FDR control).

    Finds the largest rank k where p_(k) <= (k/m)*alpha, then marks
    all ranks 1..k as significant.

    Returns a list of bools (same order as input) indicating significance.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    # Find the largest rank that passes
    bh_cutoff = 0
    for rank_0, (_orig_idx, p) in enumerate(indexed):
        rank = rank_0 + 1
        if p <= (rank / n) * alpha:
            bh_cutoff = rank
    # All ranks up to bh_cutoff are significant
    results = [False] * n
    for rank_0, (orig_idx, _p) in enumerate(indexed):
        if (rank_0 + 1) <= bh_cutoff:
            results[orig_idx] = True
    return results


# ── Run one seed ───────────────────────────────────────────────────────
def run_seed(
    scenario_path: Path,
    seed: int,
    agent_specs: List[Dict[str, Any]],
    groups: Dict[str, Dict[str, Any]],
) -> Dict[str, List[float]]:
    """Run a single seed and return {group_label: [payoff_per_agent]}."""
    scenario = load_scenario(scenario_path)
    scenario.orchestrator_config.seed = seed
    # Disable file logging for batch runs
    scenario.orchestrator_config.log_events = False
    scenario.orchestrator_config.log_path = None

    orch = build_orchestrator(scenario)
    orch.run()

    id_to_group = map_agents_to_groups(
        list(orch.state.agents.keys()), groups, agent_specs
    )

    result: Dict[str, List[float]] = defaultdict(list)
    for aid, agent_state in orch.state.agents.items():
        group = id_to_group.get(aid, "unknown")
        result[group].append(agent_state.total_payoff)

    return dict(result)


# ── Detect ordering variable ──────────────────────────────────────────
def detect_ordering(groups: Dict[str, Dict[str, Any]]) -> Optional[Tuple[str, Dict[str, float]]]:
    """Detect if groups have a natural ordering variable (e.g. recursion_depth, memory_budget).

    Returns (variable_name, {group_label: value}) or None.
    """
    # Only consider groups that have config
    config_groups = {k: v for k, v in groups.items() if v.get("config")}
    if len(config_groups) < 2:
        return None

    # Candidate ordering keys
    candidates = ["recursion_depth", "memory_budget", "planning_horizon"]
    for key in candidates:
        values = {}
        for label, g in config_groups.items():
            val = g["config"].get(key)
            if val is not None:
                values[label] = float(val)
        if len(values) >= 2 and len(set(values.values())) > 1:
            return key, values
    return None


# ── Format results ─────────────────────────────────────────────────────
def format_results(
    scenario_id: str,
    description: str,
    scenario_path: Path,
    seeds: List[int],
    groups: Dict[str, Dict[str, Any]],
    group_labels: List[str],
    per_seed_data: List[Dict[str, List[float]]],
    all_payoffs: Dict[str, np.ndarray],
    seed_means: Dict[str, np.ndarray],
    tests: List[Dict[str, Any]],
    overall_gini: float,
    ordering_info: Optional[Tuple[str, Dict[str, float]]],
) -> str:
    """Format all results into a publication-ready text block."""
    lines: List[str] = []
    sep = "=" * 90

    # Header
    lines.append(sep)
    lines.append(f"EXPERIMENT ANALYSIS: {scenario_id}")
    lines.append(sep)
    lines.append(f"Scenario: {scenario_path}")
    lines.append(f"Description: {description}")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    lines.append(f"Seeds (fixed a priori): {seeds}")
    lines.append("")

    # Agent groups
    lines.append("Agent Groups:")
    for label in group_labels:
        g = groups[label]
        config_str = ", ".join(f"{k}={v}" for k, v in g["config"].items()) if g["config"] else "default config"
        lines.append(f"  {label} ({g['count']} agents): type={g['type']}, {config_str}")
    lines.append("")

    # Per-seed summary table
    lines.append(sep)
    lines.append("PER-SEED SUMMARY TABLE")
    lines.append(sep)

    header = f"  {'Seed':>6}"
    for label in group_labels:
        header += f"  {label:>14}"
    lines.append(header)
    lines.append("-" * len(header))

    for i, seed in enumerate(seeds):
        row = f"  {seed:>6}"
        for label in group_labels:
            mean_val = np.mean(per_seed_data[i].get(label, [0]))
            row += f"  {mean_val:>14.2f}"
        lines.append(row)

    lines.append("-" * len(header))
    row = f"  {'Mean':>6}"
    for label in group_labels:
        row += f"  {np.mean(seed_means[label]):>14.2f}"
    lines.append(row)
    row = f"  {'Std':>6}"
    for label in group_labels:
        row += f"  {np.std(seed_means[label], ddof=1):>14.2f}"
    lines.append(row)
    lines.append("")

    # Descriptive statistics
    lines.append(sep)
    lines.append("DESCRIPTIVE STATISTICS (per-agent payoffs pooled across seeds)")
    lines.append(sep)
    for label in group_labels:
        arr = all_payoffs[label]
        lines.append(
            f"  {label:<25}: n={len(arr):>3}, mean={np.mean(arr):>9.2f}, "
            f"std={np.std(arr, ddof=1):>9.2f}, min={np.min(arr):>9.2f}, max={np.max(arr):>9.2f}"
        )
    lines.append("")
    lines.append(f"  Overall Gini coefficient: {overall_gini:.4f}")
    lines.append("  (0 = perfect equality, 1 = maximum inequality)")
    lines.append("")

    # Hypothesis tests
    lines.append(sep)
    lines.append("HYPOTHESIS TESTS")
    lines.append(sep)
    lines.append(f"  Seeds fixed a priori: {seeds}")
    lines.append(f"  Using group means per seed (n={len(seeds)} per group)")

    n_tests = len(tests)
    bonf_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
    lines.append(f"  Total tests: {n_tests} | Bonferroni alpha: {bonf_alpha:.5f} | Holm step-down applied")
    lines.append("")

    test_header = f"{'Test':<48} {'Statistic':>12} {'Raw p':>12}   {'Sig':>3}  {'Cohen d':>9}   {'Bonf':>4}   {'Holm':>4}"
    lines.append(test_header)
    lines.append("-" * len(test_header))

    for t in tests:
        sig_str = "***" if t["raw_p"] < 0.001 else ("**" if t["raw_p"] < 0.01 else ("*" if t["raw_p"] < 0.05 else "ns"))
        d_str = f"{t['cohens_d']:>9.3f}" if t["cohens_d"] is not None else f"{'--':>9}"
        bonf_str = "Yes" if t["bonferroni"] else "No"
        holm_str = "Yes" if t["holm"] else "No"

        stat_prefix = t.get("stat_name", "t")
        stat_str = f"{stat_prefix}={t['statistic']:.4f}"
        lines.append(
            f"{t['name']:<48} {stat_str:>12} {t['raw_p']:>12.6f}   {sig_str:>3}  {d_str}   {bonf_str:>4}   {holm_str:>4}"
        )
    lines.append("")

    # P-hacking audit
    lines.append(sep)
    lines.append("P-HACKING AUDIT (sorted by raw p-value)")
    lines.append(sep)

    sorted_tests = sorted(enumerate(tests), key=lambda x: x[1]["raw_p"])
    audit_header = f"{'Rank':>4}  {'Test':<50} {'Raw p':>12} {'Bonferroni':>12} {'Holm':>12}"
    lines.append(audit_header)
    lines.append("-" * len(audit_header))

    for rank, (_, t) in enumerate(sorted_tests, 1):
        bonf_str = "Yes" if t["bonferroni"] else "No"
        holm_str = "Yes" if t["holm"] else "No"
        lines.append(f"{rank:>4}  {t['name']:<50} {t['raw_p']:>12.8f} {bonf_str:>12} {holm_str:>12}")

    lines.append("")
    n_sig_raw = sum(1 for t in tests if t["raw_p"] < 0.05)
    n_bonf = sum(1 for t in tests if t["bonferroni"])
    n_holm = sum(1 for t in tests if t["holm"])
    lines.append(f"Total tests conducted: {n_tests}")
    lines.append(f"Tests significant at alpha=0.05 (raw): {n_sig_raw}/{n_tests}")
    lines.append(f"Tests surviving Bonferroni: {n_bonf}/{n_tests}")
    lines.append(f"Tests surviving Holm-Bonferroni: {n_holm}/{n_tests}")
    lines.append("")

    # Domain-specific: ordering gradient
    if ordering_info:
        var_name, var_values = ordering_info
        lines.append(sep)
        lines.append(f"DOMAIN-SPECIFIC: {var_name.upper().replace('_', ' ')}-PAYOFF GRADIENT")
        lines.append(sep)
        for label in sorted(var_values, key=lambda k: var_values[k]):
            lines.append(f"  {var_name}={var_values[label]:.0f} ({label}) mean: {np.mean(seed_means[label]):.2f}")

        # Find non-ordering groups (e.g. honest baseline)
        non_ordered = [gl for gl in group_labels if gl not in var_values]
        for label in non_ordered:
            lines.append(f"  {label} baseline mean: {np.mean(seed_means[label]):.2f}")

        # Find the Pearson test for this variable
        for t in tests:
            if "Pearson" in t["name"] and var_name in t["name"]:
                direction = "BETTER" if t["statistic"] > 0 else "WORSE"
                lines.append(f"  Pearson r({var_name}, payoff): {t['statistic']:.4f}, p={t['raw_p']:.6f}")
                lines.append(f"  Direction: more {var_name.replace('_', ' ')} = {direction}")
                break

        lines.append("")

    return "\n".join(lines)


# ── Main analysis pipeline ─────────────────────────────────────────────
def analyze(scenario_path: Path, seeds: List[int]) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
    """Run the full analysis pipeline.

    Returns (formatted_text, summary_dict, csv_rows).
    """
    scenario = load_scenario(scenario_path)
    groups = detect_groups(scenario.agent_specs)
    group_labels = list(groups.keys())

    print(f"Scenario: {scenario.scenario_id}")
    print(f"Groups detected: {group_labels}")
    print(f"Seeds: {seeds}")
    print()

    # ── Run all seeds ──────────────────────────────────────────────────
    per_seed_data: List[Dict[str, List[float]]] = []
    for i, seed in enumerate(seeds):
        print(f"  Running seed {seed} ({i + 1}/{len(seeds)})...", end=" ", flush=True)
        result = run_seed(scenario_path, seed, scenario.agent_specs, groups)
        per_seed_data.append(result)
        summary_parts = []
        for label in group_labels:
            vals = result.get(label, [])
            if vals:
                summary_parts.append(f"{label}={np.mean(vals):.1f}")
        print("  ".join(summary_parts))

    print()

    # ── Aggregate ──────────────────────────────────────────────────────
    # Per-agent payoffs pooled across all seeds
    all_payoffs: Dict[str, np.ndarray] = {}
    for label in group_labels:
        pooled = []
        for sd in per_seed_data:
            pooled.extend(sd.get(label, []))
        all_payoffs[label] = np.array(pooled)

    # Group means per seed (for paired tests)
    seed_means: Dict[str, np.ndarray] = {}
    for label in group_labels:
        means = []
        for sd in per_seed_data:
            vals = sd.get(label, [])
            means.append(np.mean(vals) if vals else 0.0)
        seed_means[label] = np.array(means)

    # Overall Gini
    all_values = np.concatenate(list(all_payoffs.values()))
    overall_gini = gini_coefficient(all_values)

    # Gini per seed for 1-sample t-test
    gini_per_seed_list = []
    for sd in per_seed_data:
        seed_vals = []
        for label in group_labels:
            seed_vals.extend(sd.get(label, []))
        gini_per_seed_list.append(gini_coefficient(np.array(seed_vals)))
    gini_per_seed = np.array(gini_per_seed_list)

    # ── Hypothesis tests ───────────────────────────────────────────────
    tests: List[Dict[str, Any]] = []

    # Pairwise t-tests
    for g1, g2 in combinations(group_labels, 2):
        t_stat, p_val = stats.ttest_ind(seed_means[g1], seed_means[g2])
        pooled_std = np.sqrt(
            (np.var(seed_means[g1], ddof=1) + np.var(seed_means[g2], ddof=1)) / 2
        )
        d = (np.mean(seed_means[g1]) - np.mean(seed_means[g2])) / pooled_std if pooled_std > 0 else 0.0
        tests.append({
            "name": f"t-test: {g1} vs {g2}",
            "stat_name": "t",
            "statistic": float(t_stat),
            "raw_p": float(p_val),
            "cohens_d": float(d),
            "bonferroni": False,
            "holm": False,
        })

    # ANOVA: all groups
    if len(group_labels) >= 3:
        anova_data = [seed_means[label] for label in group_labels]
        f_stat, p_val = stats.f_oneway(*anova_data)
        tests.append({
            "name": "ANOVA: All groups",
            "stat_name": "F",
            "statistic": float(f_stat),
            "raw_p": float(p_val),
            "cohens_d": None,
            "bonferroni": False,
            "holm": False,
        })

    # ANOVA: subgroups of same type (e.g. RLM tiers only)
    type_groups: Dict[str, List[str]] = defaultdict(list)
    for label, g in groups.items():
        type_groups[g["type"]].append(label)
    for agent_type, labels in type_groups.items():
        if len(labels) >= 3:
            sub_data = [seed_means[lb] for lb in labels]
            f_stat, p_val = stats.f_oneway(*sub_data)
            tests.append({
                "name": f"ANOVA: {agent_type} tiers only",
                "stat_name": "F",
                "statistic": float(f_stat),
                "raw_p": float(p_val),
                "cohens_d": None,
                "bonferroni": False,
                "holm": False,
            })

    # Pearson: ordering variable vs payoff
    ordering_info = detect_ordering(groups)
    if ordering_info:
        var_name, var_values = ordering_info
        # Group-level correlation (using seed means)
        x_vals = []
        y_vals = []
        for label, val in var_values.items():
            for mean_val in seed_means[label]:
                x_vals.append(val)
                y_vals.append(mean_val)
        r, p_val = stats.pearsonr(x_vals, y_vals)
        tests.append({
            "name": f"Pearson: {var_name} vs payoff",
            "stat_name": "r",
            "statistic": float(r),
            "raw_p": float(p_val),
            "cohens_d": None,
            "bonferroni": False,
            "holm": False,
        })

        # Agent-level correlation (exploitation rate)
        x_agent = []
        y_agent = []
        for label, val in var_values.items():
            for payoff in all_payoffs[label]:
                x_agent.append(val)
                y_agent.append(payoff)
        if len(x_agent) > 2:
            r_agent, p_agent = stats.pearsonr(x_agent, y_agent)
            tests.append({
                "name": "Exploitation rate (agent-level r)",
                "stat_name": "r",
                "statistic": float(r_agent),
                "raw_p": float(p_agent),
                "cohens_d": None,
                "bonferroni": False,
                "holm": False,
            })

    # 1-sample t-test: Gini > 0
    t_gini, p_gini = stats.ttest_1samp(gini_per_seed, 0)
    p_gini_one = float(p_gini) / 2 if t_gini > 0 else 1.0  # one-tailed
    tests.append({
        "name": "1-sample t: Gini > 0",
        "stat_name": "t",
        "statistic": float(t_gini),
        "raw_p": p_gini_one,
        "cohens_d": None,
        "bonferroni": False,
        "holm": False,
    })

    # ── Apply corrections ──────────────────────────────────────────────
    n_tests = len(tests)
    bonf_alpha = 0.05 / n_tests if n_tests > 0 else 0.05
    raw_ps = [t["raw_p"] for t in tests]
    holm_results = holm_bonferroni(raw_ps)

    for i, t in enumerate(tests):
        t["bonferroni"] = t["raw_p"] < bonf_alpha
        t["holm"] = holm_results[i]

    # ── Format output ──────────────────────────────────────────────────
    text = format_results(
        scenario_id=scenario.scenario_id,
        description=scenario.description,
        scenario_path=scenario_path,
        seeds=seeds,
        groups=groups,
        group_labels=group_labels,
        per_seed_data=per_seed_data,
        all_payoffs=all_payoffs,
        seed_means=seed_means,
        tests=tests,
        overall_gini=overall_gini,
        ordering_info=ordering_info,
    )

    # ── Build summary dict ─────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "scenario_id": scenario.scenario_id,
        "description": scenario.description,
        "seeds": seeds,
        "n_seeds": len(seeds),
        "groups": {
            label: {
                "type": g["type"],
                "count": g["count"],
                "config": g["config"],
                "mean": float(np.mean(seed_means[label])),
                "std": float(np.std(seed_means[label], ddof=1)),
            }
            for label, g in groups.items()
        },
        "overall_gini": float(overall_gini),
        "tests": [
            {
                "name": t["name"],
                "statistic": t["statistic"],
                "raw_p": t["raw_p"],
                "cohens_d": t["cohens_d"],
                "bonferroni": t["bonferroni"],
                "holm": t["holm"],
            }
            for t in tests
        ],
    }

    # ── Build CSV rows ─────────────────────────────────────────────────
    csv_rows: List[Dict[str, Any]] = []
    for i, seed in enumerate(seeds):
        for label in group_labels:
            for j, payoff in enumerate(per_seed_data[i].get(label, [])):
                csv_rows.append({
                    "seed": seed,
                    "agent_id": f"{groups[label]['type']}_{j + 1}",
                    "group": label,
                    "payoff": payoff,
                })

    return text, summary, csv_rows


def save_artifacts(
    scenario_id: str,
    text: str,
    summary: Dict[str, Any],
    csv_rows: List[Dict[str, Any]],
) -> Path:
    """Save analysis artifacts to a timestamped run directory."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path("runs") / f"{timestamp}_analysis_{scenario_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # results.txt
    (run_dir / "results.txt").write_text(text)

    # summary.json
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # per_agent_payoffs.csv
    if csv_rows:
        csv_path = run_dir / "per_agent_payoffs.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["seed", "agent_id", "group", "payoff"])
            writer.writeheader()
            writer.writerows(csv_rows)

    return run_dir


# ── CLI ────────────────────────────────────────────────────────────────
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed statistical analysis for SWARM scenarios",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python -m swarm.scripts.analyze scenarios/rlm_recursive_collusion.yaml
              python -m swarm.scripts.analyze rlm_memory_as_power --seeds 42,7,123
              python -m swarm.scripts.analyze rlm_governance_lag --seeds 20
        """),
    )
    parser.add_argument(
        "scenario",
        help="Path to YAML or scenario ID (resolved to scenarios/<id>.yaml)",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seed list OR integer N for N random seeds. "
        "Default: 42,7,123,256,999,2024,314,577,1337,8080",
    )
    return parser.parse_args(argv)


def resolve_scenario_path(scenario_arg: str) -> Path:
    """Resolve a scenario argument to a file path."""
    p = Path(scenario_arg)
    if p.exists():
        return p
    # Try as scenario ID
    p = Path("scenarios") / f"{scenario_arg}.yaml"
    if p.exists():
        return p
    raise FileNotFoundError(
        f"Cannot find scenario: {scenario_arg} (tried {scenario_arg} and {p})"
    )


def parse_seeds(seeds_arg: Optional[str]) -> List[int]:
    """Parse the --seeds argument."""
    if seeds_arg is None:
        return DEFAULT_SEEDS
    # Try as integer (generate N random seeds)
    try:
        n = int(seeds_arg)
        rng = np.random.default_rng(0)
        return [int(s) for s in rng.integers(1, 10000, size=n)]
    except ValueError:
        pass
    # Try as comma-separated list
    return [int(s.strip()) for s in seeds_arg.split(",")]


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    scenario_path = resolve_scenario_path(args.scenario)
    seeds = parse_seeds(args.seeds)

    print(f"{'=' * 70}")
    print("SWARM Statistical Analysis")
    print(f"{'=' * 70}")
    print()

    text, summary, csv_rows = analyze(scenario_path, seeds)

    # Print results
    print()
    print(text)

    # Save artifacts
    run_dir = save_artifacts(summary["scenario_id"], text, summary, csv_rows)
    print()
    print(f"Artifacts saved to: {run_dir}/")
    print("  results.txt          - Full formatted output")
    print("  summary.json         - Machine-readable results")
    print("  per_agent_payoffs.csv - Raw per-agent payoff data")


if __name__ == "__main__":
    main()
