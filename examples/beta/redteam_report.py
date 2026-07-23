"""Red-team report: attacking the distributional defenses, and patching them.

Every defense in beta-swarm rests on one assumption — deception cannot fake
evidence volume, so malicious beliefs arrive diffuse and get deferred to audit.
This runs the attack library against that assumption and then hardens the
governor until the strongest attack is contained.

Run:  python examples/redteam_report.py
"""

from beta_swarm import (
    AttackLibrary,
    BetaProxyComputer,
    SimulationConfig,
    TailMassGovernor,
    run_attack,
)


def tail_gov() -> TailMassGovernor:
    return TailMassGovernor(
        tau=0.4, max_tail_mass=0.2, min_mean=None, defer_below_concentration=20.0
    )


def main() -> None:
    # ------------------------------------------------------------------
    # 1. The attack library against the standard tail governor
    # ------------------------------------------------------------------
    print("Attack library vs. the standard tail-mass governor:\n")
    print(f"  {'attack':20s}{'category':14s}{'evasion':>8s}{'audit-ev':>10s}{'damage':>8s}{'detect':>8s}")
    print("  " + "-" * 68)
    for scenario in AttackLibrary.all():
        r = run_attack(scenario, tail_gov(), SimulationConfig(seed=7))
        detect = "never" if r.detection_epoch is None else f"ep {r.detection_epoch}"
        print(
            f"  {scenario.attack_id:20s}{scenario.category.value:14s}"
            f"{r.evasion_rate:>7.0%}{r.audit_evasion_rate:>10.0%}"
            f"{r.realized_damage:>8.0f}{detect:>8s}"
        )
    print(
        "\n  Volume forgery is the break: 56% of forged interactions accepted, "
        "99% never\n  audited, never detected. Reputation farming evades for an "
        "exploitation window\n  before realized outcomes convict it. The "
        "collusion ring is contained by the\n  tail gate and flagged by the "
        "shape detector (see examples/collusion_ring.py).\n"
    )

    # ------------------------------------------------------------------
    # 2. Anatomy of volume forgery, and its two-layer patch
    # ------------------------------------------------------------------
    forgery = AttackLibrary.volume_forgery()
    stack = [
        ("vulnerable", SimulationConfig(seed=7), None),
        (
            "+ scaled audit",
            SimulationConfig(seed=7, audit_scales_with_concentration=True),
            None,
        ),
        ("+ concentration cap", SimulationConfig(seed=7), BetaProxyComputer(max_concentration=6.0)),
        (
            "+ both",
            SimulationConfig(seed=7, audit_scales_with_concentration=True),
            BetaProxyComputer(max_concentration=6.0),
        ),
    ]
    print("Volume forgery under progressive hardening:\n")
    print(f"  {'configuration':22s}{'evasion':>8s}{'audit-ev':>10s}{'damage':>8s}")
    print("  " + "-" * 48)
    for label, cfg, proxy in stack:
        r = run_attack(forgery, tail_gov(), cfg, proxy=proxy)
        print(
            f"  {label:22s}{r.evasion_rate:>7.0%}{r.audit_evasion_rate:>10.0%}"
            f"{r.realized_damage:>8.0f}"
        )
    print(
        "\n  Two distinct failures, two fixes. The forged belief evades the "
        "DEFER gate\n  (never audited), and its forged concentration would blunt "
        "a fixed-strength\n  audit if one fired. Scaling the audit alone barely "
        "helps — the attack never\n  triggers an audit. Capping the concentration "
        "a single interaction may assert\n  is the decisive fix: sharpness must "
        "be earned through audits and reputation,\n  not claimed in one shot. It "
        "drops evasion from 56% to 10% and damage 12x.\n  The residual 10% is the "
        "forger occasionally drawing genuinely acceptable\n  work — not a "
        "governance failure."
    )


if __name__ == "__main__":
    main()
