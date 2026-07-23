"""Governor showdown: legacy mean threshold vs. distributional tail-mass gate.

The same seeded population — honest, mediocre, and deceptive agents — runs
through the closed loop twice:

- **mean governor**: accept iff the belief mean clears 0.5. This is the scalar
  formalism's only lever; it sees exactly what ``p`` saw.
- **tail governor**: bound the downside tail mass ``P(v < tau)``, and DEFER
  diffuse beliefs to an audit that injects evidence before deciding.

Deceptive agents inflate their observables, so their belief *means* look fine.
What they cannot fake is evidence volume: their beliefs arrive diffuse. The
mean governor admits them until pooled reputation slowly drags the mean under
the bar — absorbing the damage meanwhile. The tail governor defers exactly
those diffuse beliefs to audit, which concentrates them around the (poor) truth
and rejects them from the first epoch.

Run:  python examples/governor_showdown.py
"""

from beta_swarm import (
    Archetype,
    Simulation,
    SimulationConfig,
    TailMassGovernor,
    make_population,
)


def run(governor: TailMassGovernor):
    population = make_population(
        {Archetype.HONEST: 4, Archetype.MEDIOCRE: 3, Archetype.DECEPTIVE: 3}
    )
    return Simulation(population, governor, SimulationConfig(seed=7)).run()


def main() -> None:
    mean_gov = TailMassGovernor(tau=0.4, max_tail_mass=1.0, min_mean=0.5)
    # The defer threshold must be reachable by one audit (roughly
    # base concentration + audit_strength), or every interaction loops
    # DEFER -> audit -> DEFER -> REJECT and the ecosystem shuts down.
    tail_gov = TailMassGovernor(
        tau=0.4, max_tail_mass=0.2, min_mean=None, defer_below_concentration=20.0
    )

    results = {"mean governor": run(mean_gov), "tail governor": run(tail_gov)}

    print("Same population, same seed — only the governance lever differs.\n")
    header = f"{'':24s}{'mean governor':>16s}{'tail governor':>16s}"
    print(header)
    print("-" * len(header))

    rows = [
        ("realized welfare", lambda r: f"{r.total_welfare:+.1f}"),
        ("realized toxicity", lambda r: f"{r.overall_toxicity:.3f}"),
        ("audits run", lambda r: str(sum(e.n_audits for e in r.epoch_reports))),
        ("mean CRPS (calibration)", _mean_crps),
    ]
    for label, fmt in rows:
        print(f"{label:24s}{fmt(results['mean governor']):>16s}{fmt(results['tail governor']):>16s}")

    print("\nAcceptance rate by archetype:")
    present = {i.metadata["archetype"] for i in results["mean governor"].interactions}
    for archetype in Archetype:
        if archetype.value not in present:
            continue
        print(
            f"  {archetype.value:12s}"
            f"{results['mean governor'].acceptance_rate(archetype):>14.0%}"
            f"{results['tail governor'].acceptance_rate(archetype):>16.0%}"
        )

    tail_first = results["tail governor"].epoch_reports[0]
    mean_first = results["mean governor"].epoch_reports[0]
    print(
        f"\nEpoch 0 deceptive acceptance — mean governor: "
        f"{mean_first.acceptance_by_archetype.get('deceptive', 0.0):.0%}, "
        f"tail governor: {tail_first.acceptance_by_archetype.get('deceptive', 0.0):.0%}."
    )
    print(
        "The mean governor learns who the deceptive agents are only by paying "
        "for their work;\nthe tail governor never has to — diffuse beliefs are "
        "audited before they are trusted."
    )


def _mean_crps(result) -> str:
    scores = [e.crps for e in result.epoch_reports if e.crps is not None]
    return f"{sum(scores) / len(scores):.4f}" if scores else "n/a"


if __name__ == "__main__":
    main()
