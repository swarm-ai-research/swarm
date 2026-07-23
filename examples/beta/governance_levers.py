"""Composable governance levers: accountability beyond the admission gate.

The tail-mass governor decides admission *ex ante*, from the belief. But a
forged belief can slip past it (see examples/redteam_report.py). Levers add the
surrounding machinery the parent framework leans on — and their distributional
generalizations act *ex post*, on realized outcomes, where no forged belief can
hide:

- StakingLever: participation costs a stake, slashed when accepted work's
  realized outcome lands in the downside tail.
- TailCircuitBreaker: freeze an agent whose recent *realized* bad-outcome rate
  exceeds a bound.
- RiskTaxLever: price the belief's tail mass into a transaction tax.

Because they judge ground truth rather than the projected belief, a staking +
circuit-breaker stack contains volume forgery and reputation farming even with
no belief-level fix (no concentration cap). The catch — shown at the end — is
that this accountability is not free: it also costs honest agents who
occasionally underperform.

Run:  python examples/governance_levers.py
"""

from beta_swarm import (
    Archetype,
    AttackLibrary,
    GovernanceStack,
    RiskTaxLever,
    Simulation,
    SimulationConfig,
    StakingLever,
    TailCircuitBreaker,
    TailMassGovernor,
    make_population,
    run_attack,
)


def tail_gov() -> TailMassGovernor:
    return TailMassGovernor(
        tau=0.4, max_tail_mass=0.2, min_mean=None, defer_below_concentration=20.0
    )


def stack() -> GovernanceStack:
    return GovernanceStack(
        [
            StakingLever(min_stake=1.0, slash_rate=0.5, tau=0.4),
            TailCircuitBreaker(max_tail_exposure=0.5, window=6, freeze_epochs=3, tau=0.4),
            RiskTaxLever(rate=1.0, tau=0.4),
        ],
        initial_resources=5.0,
    )


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Levers contain the attacks that beat the belief-level gate
    # ------------------------------------------------------------------
    print("A staking + circuit-breaker + risk-tax stack vs. the attack library")
    print("(no concentration cap — belief admission is the vulnerable version):\n")
    print(f"  {'attack':20s}{'evasion':>18s}{'blocked':>9s}{'damage':>16s}")
    print("  " + "-" * 61)
    for scenario in AttackLibrary.all():
        base = run_attack(scenario, tail_gov(), SimulationConfig(seed=7))
        lev = run_attack(scenario, tail_gov(), SimulationConfig(seed=7), levers=stack())
        print(
            f"  {scenario.attack_id:20s}"
            f"{base.evasion_rate:>8.0%} -> {lev.evasion_rate:<6.0%}"
            f"{lev.blocked_rate:>8.0%}"
            f"{base.realized_damage:>8.0f} -> {lev.realized_damage:<6.0f}"
        )
    print(
        "\n  The stack judges realized outcomes, not the projected belief, so it "
        "reaches\n  volume forgery (damage 349 -> 20) where the belief gate could "
        "not. Staking is\n  the workhorse: skin in the game runs out for an agent "
        "whose accepted work\n  keeps coming up bad. Reputation farming keeps a "
        "high evasion rate — its\n  pre-flip work is genuinely good — but its "
        "post-flip damage collapses 7x.\n"
    )

    # ------------------------------------------------------------------
    # 2. The cost: accountability is not free
    # ------------------------------------------------------------------
    population = {Archetype.HONEST: 4, Archetype.MEDIOCRE: 3, Archetype.DECEPTIVE: 3}
    print("The tradeoff — the same stack on a mixed population with no attack:\n")
    print(f"  {'':12s}{'honest':>9s}{'mediocre':>10s}{'deceptive':>11s}{'welfare':>10s}")
    print("  " + "-" * 52)
    for label, levers in [("no levers", None), ("with stack", stack())]:
        result = Simulation(
            make_population(population), tail_gov(), SimulationConfig(seed=7), levers=levers
        ).run()
        print(
            f"  {label:12s}"
            f"{result.acceptance_rate(Archetype.HONEST):>9.0%}"
            f"{result.acceptance_rate(Archetype.MEDIOCRE):>10.0%}"
            f"{result.acceptance_rate(Archetype.DECEPTIVE):>11.0%}"
            f"{result.total_welfare:>+10.0f}"
        )
    print(
        "\n  Staking is a blunt instrument: an honest agent that occasionally "
        "draws poor\n  work gets slashed too, so honest throughput dips (99% -> "
        "92%) and welfare with\n  it. The circuit breaker (which needs a "
        "*sustained* bad rate) is more targeted.\n  Choosing the mix is the "
        "governance design problem the levers exist to study."
    )


if __name__ == "__main__":
    main()
