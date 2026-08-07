"""Collusion ring detection: the fingerprint is in the belief shape.

A three-agent ring runs inside a population of honest and mediocre agents,
plus one lone deceiver, under a permissive mean-threshold governor (the scalar
formalism's world — no audits, no tail gates). The ring lies to everyone
equally, so the classic scalar signal — high mean inside the ring, low mean
outside — is nearly zero. But ring partners corroborate each other's inflated
signals with faked engagement, which is *evidence*: within-ring beliefs arrive
sharper than the same agents' beliefs toward outsiders. The detector reads
that shape asymmetry (Wasserstein-1, concentration lift) alongside the graph
signal (frequency lift).

Run:  python examples/collusion_ring.py
"""

from beta_swarm import (
    Archetype,
    CollusionDetector,
    Simulation,
    SimulationConfig,
    TailMassGovernor,
    make_population,
)


def main() -> None:
    population = make_population(
        {
            Archetype.HONEST: 4,
            Archetype.MEDIOCRE: 2,
            Archetype.DECEPTIVE: 1,
            Archetype.COLLUDER: 3,
        }
    )
    governor = TailMassGovernor(tau=0.4, max_tail_mass=1.0, min_mean=0.5)
    result = Simulation(population, governor, SimulationConfig(seed=7)).run()

    detector = CollusionDetector()
    reports = detector.analyze(result.interactions)
    flagged = detector.flag(result.interactions)

    print(
        f"{len(result.interactions)} interactions, "
        f"{len(reports)} pairs scored, {len(flagged)} flagged.\n"
    )
    header = (
        f"{'pair':28s}{'n':>4s}{'freq':>7s}{'mean gap':>10s}"
        f"{'W1 gap':>8s}{'conc':>7s}{'suspicion':>11s}"
    )
    print(header)
    print("-" * len(header))
    for r in reports[:8]:
        print(
            f"{r.initiator + ' -> ' + r.counterparty:28s}{r.n_interactions:>4d}"
            f"{r.frequency_lift:>6.1f}x{r.mean_gap:>+10.3f}{r.w1_gap:>8.3f}"
            f"{r.concentration_lift:>6.2f}x{r.suspicion:>11.3f}"
        )

    ring_pairs = {
        (r.initiator, r.counterparty)
        for r in reports
        if r.initiator.startswith("colluder") and r.counterparty.startswith("colluder")
    }
    caught = {(r.initiator, r.counterparty) for r in flagged}
    print(
        f"\nRing pairs flagged: {len(caught & ring_pairs)}/{len(ring_pairs)}; "
        f"false positives: {len(caught - ring_pairs)}."
    )
    if reports:
        top = reports[0]
        print(
            f"\nThe scalar tell is gone — the top pair's mean gap is "
            f"{top.mean_gap:+.3f} (they inflate toward everyone). The shape "
            f"tells: within-ring beliefs are {top.concentration_lift:.2f}x more "
            f"concentrated,\nbecause corroborated engagement is evidence a lone "
            f"deceiver cannot fake."
        )


if __name__ == "__main__":
    main()
