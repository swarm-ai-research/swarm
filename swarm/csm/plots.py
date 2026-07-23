"""CSM benchmark plotting utilities.

Produces the six deliverable plots from the benchmark specification:
1. Welfare vs Agent Adoption (S-curve by market)
2. Congestion vs Adoption (tipping points)
3. Obfuscation Arms Race (transparency over time)
4. BYO vs Bowling-shoe comparison
5. Preference Dimensionality stress test
6. Identity regime frontier (fraud vs inclusion vs cost)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.csm.runner import CSMEpisodeResult

# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _group_by_key(
    results: List[CSMEpisodeResult],
    key_fn,
) -> Dict[str, List[CSMEpisodeResult]]:
    """Group results by a key function."""
    groups: Dict[str, List[CSMEpisodeResult]] = {}
    for r in results:
        k = str(key_fn(r))
        groups.setdefault(k, []).append(r)
    return groups


# ---------------------------------------------------------------------------
# Plot 1: Welfare vs Agent Adoption
# ---------------------------------------------------------------------------

def welfare_vs_adoption_data(
    results: List[CSMEpisodeResult],
) -> Dict[str, Any]:
    """Extract data for welfare vs adoption S-curve.

    Returns:
        Dict with adoption_rates and welfare values per market module.
    """
    groups = _group_by_key(
        results, lambda r: r.treatment.market_module.value
    )

    data: Dict[str, Any] = {"modules": {}}
    for module, module_results in groups.items():
        rates = []
        welfare = []
        tx_costs = []
        for r in sorted(module_results, key=lambda x: x.treatment.adoption_rate):
            rates.append(r.treatment.adoption_rate)
            welfare.append(r.final_metrics.welfare.total_surplus)
            tx_costs.append(r.final_metrics.transaction_costs.total_search_cost)
        data["modules"][module] = {
            "adoption_rates": rates,
            "welfare": welfare,
            "tx_costs": tx_costs,
        }
    return data


def plot_welfare_vs_adoption(
    results: List[CSMEpisodeResult],
    output_path: Optional[Path] = None,
) -> Optional[Any]:
    """Plot welfare vs agent adoption (deliverable 1).

    Args:
        results: CSM episode results.
        output_path: Optional path to save the figure.

    Returns:
        matplotlib Figure if available, else None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    data = welfare_vs_adoption_data(results)

    fig, ax = plt.subplots(figsize=(8, 5))
    for module, mdata in data["modules"].items():
        ax.plot(
            mdata["adoption_rates"],
            mdata["welfare"],
            marker="o",
            label=module,
        )

    ax.set_xlabel("Agent Adoption Rate")
    ax.set_ylabel("Total Surplus")
    ax.set_title("Welfare vs Agent Adoption (CSM)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Plot 2: Congestion vs Adoption
# ---------------------------------------------------------------------------

def congestion_vs_adoption_data(
    results: List[CSMEpisodeResult],
) -> Dict[str, Any]:
    """Extract congestion vs adoption data."""
    matching = [
        r for r in results
        if r.treatment.market_module.value == "matching"
    ]
    rates = []
    congestion = []
    for r in sorted(matching, key=lambda x: x.treatment.adoption_rate):
        rates.append(r.treatment.adoption_rate)
        congestion.append(r.final_metrics.equilibrium.congestion_index)

    return {"adoption_rates": rates, "congestion": congestion}


def plot_congestion_vs_adoption(
    results: List[CSMEpisodeResult],
    output_path: Optional[Path] = None,
) -> Optional[Any]:
    """Plot congestion vs adoption (deliverable 2)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    data = congestion_vs_adoption_data(results)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(data["adoption_rates"], data["congestion"], marker="s", color="red")
    ax.set_xlabel("Agent Adoption Rate")
    ax.set_ylabel("Congestion Index")
    ax.set_title("Congestion vs Adoption (Tipping Points)")
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Plot 3: Obfuscation Arms Race
# ---------------------------------------------------------------------------

def obfuscation_over_time_data(
    results: List[CSMEpisodeResult],
) -> Dict[str, Any]:
    """Extract obfuscation transparency over epochs.

    Groups by adversarial environment to show arms race dynamics.
    """
    groups = _group_by_key(
        results, lambda r: r.treatment.adversarial_env.value
    )

    data: Dict[str, Any] = {"environments": {}}
    for env, env_results in groups.items():
        # Extract time series for each result with epoch metrics
        for r in env_results:
            if r.epoch_metrics:
                epochs = list(range(len(r.epoch_metrics)))
                transparency = [
                    m.equilibrium.effective_transparency
                    for m in r.epoch_metrics
                ]
                data["environments"].setdefault(env, []).append({
                    "name": r.treatment.name,
                    "epochs": epochs,
                    "transparency": transparency,
                })
    return data


def plot_obfuscation_arms_race(
    results: List[CSMEpisodeResult],
    output_path: Optional[Path] = None,
) -> Optional[Any]:
    """Plot obfuscation arms race (deliverable 3)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    data = obfuscation_over_time_data(results)

    fig, ax = plt.subplots(figsize=(8, 5))
    for env, series_list in data["environments"].items():
        for series in series_list[:3]:  # Limit to 3 per env for readability
            ax.plot(
                series["epochs"],
                series["transparency"],
                label=f"{env}: {series['name'][:20]}",
                alpha=0.8,
            )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective Transparency")
    ax.set_title("Obfuscation Arms Race")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Plot 4: BYO vs Bowling-shoe comparison
# ---------------------------------------------------------------------------

def byo_vs_bowling_shoe_data(
    results: List[CSMEpisodeResult],
) -> Dict[str, Any]:
    """Extract BYO vs bowling-shoe comparison data."""
    byo = [r for r in results if r.treatment.ownership.value == "byo"]
    bs = [r for r in results if r.treatment.ownership.value == "bowling_shoe"]

    def avg_metric(rs, getter):
        vals = [getter(r) for r in rs if r.final_metrics]
        return sum(vals) / max(len(vals), 1)

    return {
        "metrics": ["Surplus", "Search Cost", "Faithfulness"],
        "byo": [
            avg_metric(byo, lambda r: r.final_metrics.welfare.total_surplus),
            avg_metric(byo, lambda r: r.final_metrics.transaction_costs.mean_search_cost),
            avg_metric(byo, lambda r: r.final_metrics.agency.faithfulness),
        ],
        "bowling_shoe": [
            avg_metric(bs, lambda r: r.final_metrics.welfare.total_surplus),
            avg_metric(bs, lambda r: r.final_metrics.transaction_costs.mean_search_cost),
            avg_metric(bs, lambda r: r.final_metrics.agency.faithfulness),
        ],
    }


def plot_byo_vs_bowling_shoe(
    results: List[CSMEpisodeResult],
    output_path: Optional[Path] = None,
) -> Optional[Any]:
    """Plot BYO vs bowling-shoe comparison (deliverable 4)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    data = byo_vs_bowling_shoe_data(results)

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    metrics = data["metrics"]
    byo_vals = data["byo"]
    bs_vals = data["bowling_shoe"]

    for i, (metric, byo_v, bs_v) in enumerate(zip(metrics, byo_vals, bs_vals, strict=True)):
        axes[i].bar(["BYO", "Bowling-shoe"], [byo_v, bs_v],
                     color=["steelblue", "coral"])
        axes[i].set_title(metric)
        axes[i].grid(True, alpha=0.3, axis="y")

    fig.suptitle("BYO vs Bowling-shoe Agent Comparison")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Plot 5: Preference dimensionality stress
# ---------------------------------------------------------------------------

def pref_dim_stress_data(
    results: List[CSMEpisodeResult],
) -> Dict[str, Any]:
    """Extract preference dimensionality stress test data."""
    groups = _group_by_key(
        results, lambda r: r.treatment.preference_dim.value
    )

    data: Dict[str, Any] = {}
    for dim, dim_results in groups.items():
        welfare = [r.final_metrics.welfare.mean_utility for r in dim_results]
        agency = [r.final_metrics.agency.faithfulness for r in dim_results]
        data[dim] = {
            "mean_welfare": sum(welfare) / max(len(welfare), 1),
            "mean_faithfulness": sum(agency) / max(len(agency), 1),
        }
    return data


def plot_pref_dim_stress(
    results: List[CSMEpisodeResult],
    output_path: Optional[Path] = None,
) -> Optional[Any]:
    """Plot preference dimensionality stress (deliverable 5)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    data = pref_dim_stress_data(results)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    dims = list(data.keys())
    welfare_vals = [data[d]["mean_welfare"] for d in dims]
    faith_vals = [data[d]["mean_faithfulness"] for d in dims]

    ax1.bar(dims, welfare_vals, color=["steelblue", "coral"])
    ax1.set_ylabel("Mean Welfare")
    ax1.set_title("Welfare by Preference Dim")
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.bar(dims, faith_vals, color=["steelblue", "coral"])
    ax2.set_ylabel("Faithfulness")
    ax2.set_title("Faithfulness by Preference Dim")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Preference Dimensionality Stress Test")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Plot 6: Identity regime frontier
# ---------------------------------------------------------------------------

def plot_identity_frontier(
    frontier_data: List[Dict[str, float]],
    output_path: Optional[Path] = None,
) -> Optional[Any]:
    """Plot identity regime Pareto frontier (deliverable 6).

    Args:
        frontier_data: Output from compute_identity_frontier().
        output_path: Optional path to save.

    Returns:
        matplotlib Figure if available, else None.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(8, 6))

    costs = [d["proof_cost"] for d in frontier_data]
    fraud = [d["fraud_rate"] for d in frontier_data]
    exclusion = [d["exclusion_error"] for d in frontier_data]

    # Plot fraud vs cost
    ax.plot(costs, fraud, "o-", color="red", label="Fraud Rate")
    ax.plot(costs, exclusion, "s-", color="blue", label="Exclusion Error")

    ax.set_xlabel("Proof-of-Personhood Cost")
    ax.set_ylabel("Rate")
    ax.set_title("Identity Regime Frontier: Fraud vs Inclusion vs Cost")
    ax.legend()
    ax.grid(True, alpha=0.3)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


# ---------------------------------------------------------------------------
# Combined report
# ---------------------------------------------------------------------------

def generate_all_plots(
    results: List[CSMEpisodeResult],
    output_dir: Path,
    identity_frontier: Optional[List[Dict[str, float]]] = None,
) -> List[Path]:
    """Generate all six deliverable plots.

    Args:
        results: Full benchmark results.
        output_dir: Directory to save plots.
        identity_frontier: Optional identity frontier data.

    Returns:
        List of paths to generated plot files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    plot_fns = [
        ("welfare_vs_adoption.png", plot_welfare_vs_adoption),
        ("congestion_vs_adoption.png", plot_congestion_vs_adoption),
        ("obfuscation_arms_race.png", plot_obfuscation_arms_race),
        ("byo_vs_bowling_shoe.png", plot_byo_vs_bowling_shoe),
        ("pref_dim_stress.png", plot_pref_dim_stress),
    ]

    for filename, plot_fn in plot_fns:
        path = output_dir / filename
        fig = plot_fn(results, output_path=path)
        if fig is not None:
            paths.append(path)
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except ImportError:
                pass

    # Identity frontier (separate data source)
    if identity_frontier:
        path = output_dir / "identity_frontier.png"
        fig = plot_identity_frontier(identity_frontier, output_path=path)
        if fig is not None:
            paths.append(path)
            try:
                import matplotlib.pyplot as plt
                plt.close(fig)
            except ImportError:
                pass

    return paths
