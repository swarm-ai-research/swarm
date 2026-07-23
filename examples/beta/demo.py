"""Demo: the one thing scalar p cannot do.

Two interactions have the *same* mean quality (0.5). One is a confident,
sharp belief ("this is reliably mediocre"); the other is diffuse ("could be
great, could be terrible"). The scalar-p formalism assigns both p = 0.5 and
treats them identically. beta-swarm separates them on every axis: tail mass,
governance decision, distributional payoff under a convex downside, and the
value-at-risk floor.

Run:  python examples/demo.py
"""

from beta_swarm import (
    BetaBelief,
    DistributionalPayoffEngine,
    PayoffConfig,
    TailMassGovernor,
    ValueAtRiskGovernor,
)


def main() -> None:
    confident_mediocre = BetaBelief.from_mean_concentration(mean=0.5, concentration=200.0)
    uncertain = BetaBelief.from_mean_concentration(mean=0.5, concentration=2.0)

    print("Two beliefs, identical mean (= the legacy point estimate p):\n")
    for name, b in [("confident-mediocre", confident_mediocre), ("uncertain", uncertain)]:
        print(f"  {name:18s}  {b}")
    print()

    tau = 0.25
    print(f"Downside tail mass  P(v < {tau}):")
    print(f"  confident-mediocre  {confident_mediocre.tail_mass(tau):.3f}")
    print(f"  uncertain           {uncertain.tail_mass(tau):.3f}")
    print()

    gov = TailMassGovernor(tau=tau, max_tail_mass=0.15, min_mean=None)
    print(f"Tail-mass governor (reject if P(v<{tau}) > 0.15):")
    for name, b in [("confident-mediocre", confident_mediocre), ("uncertain", uncertain)]:
        v = gov.evaluate(b)
        print(f"  {name:18s}  {v.decision.value.upper():7s}  ({v.reason})")
    print()

    # Convex downside makes the diffuse belief strictly worse, same mean.
    eng = DistributionalPayoffEngine(PayoffConfig(downside_curvature=2.0))
    print("Expected surplus with a convex downside penalty (gamma=2):")
    print(f"  confident-mediocre  {eng.expected_surplus(confident_mediocre):+.3f}")
    print(f"  uncertain           {eng.expected_surplus(uncertain):+.3f}")
    print()

    var_gov = ValueAtRiskGovernor(alpha=0.05, min_var_outcome=0.3)
    print("5%-Value-at-Risk outcome (the quality you can count on 95% of the time):")
    print(f"  confident-mediocre  {var_gov.value_at_risk(confident_mediocre):.3f}")
    print(f"  uncertain           {var_gov.value_at_risk(uncertain):.3f}")


if __name__ == "__main__":
    main()
