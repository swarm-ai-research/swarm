"""Calibration reliability: is the belief an honest forecast of the outcome?

Four measurements, in the spirit of the main swarm's proxy-fidelity arm:

1. The default proxy on faithful traffic — underconfident, and it *overstates*
   tail risk on honest work (diffuse beliefs put phantom mass in the left tail).
2. A paired sweep of ``evidence_scale`` — the concentration knob the scalar
   formalism never had — locating the calibrated optimum ~10x above default.
3. The same proxy on deceptive traffic — calibration collapses, with nearly
   all PIT mass in the bottom bin: the truth lands below the belief's entire
   support. Localized miscalibration is itself a detection signal.
4. The governance loop as calibrator: the raw proxy is equally miscalibrated
   under both showdown governors, but audits + reputation pooling repair the
   tail governor's decision beliefs to a nearly honest lever.

Run:  python examples/calibration_reliability.py
"""

from beta_swarm import (
    Archetype,
    Simulation,
    SimulationConfig,
    TailMassGovernor,
    calibrate,
    calibrate_interactions,
    fidelity_run,
    make_population,
    sweep_evidence_scale,
)


def pit_bar(freqs: list[float], width: int = 24) -> list[str]:
    peak = max(freqs) or 1.0
    return [f"{f:5.2f} {'#' * round(width * f / peak)}" for f in freqs]


def main() -> None:
    # ------------------------------------------------------------------
    # 1. Default proxy, faithful traffic
    # ------------------------------------------------------------------
    beliefs, outcomes = fidelity_run(3000, seed=0)
    report = calibrate(beliefs, outcomes)
    print("Default proxy on faithful (honest + mediocre) traffic:")
    print(
        f"  CRPS {report.crps:.4f}   PIT deviation {report.pit_deviation:.3f}   "
        f"tail ECE {report.tail_ece:.3f}   sharpness {report.sharpness:.1f}\n"
    )
    print("  Tail-mass reliability (the governor's lever, read vs. reality):")
    print(f"  {'predicted P(v<tau)':>20s}{'observed bad rate':>19s}{'n':>7s}")
    for b in report.tail_bins[:5]:
        print(f"  {b.mean_predicted:>20.3f}{b.observed_rate:>19.3f}{b.n:>7d}")
    print(
        "\n  The default proxy is underconfident: where the lever reads 15% "
        "downside risk,\n  the realized bad rate is under 1%. Honest agents "
        "pay for phantom risk.\n"
    )

    # ------------------------------------------------------------------
    # 2. Sweep the concentration knob
    # ------------------------------------------------------------------
    print("Paired sweep of evidence_scale (same draws, only sharpening differs):")
    print(f"  {'scale':>7s}{'CRPS':>9s}{'PIT dev':>9s}{'tail ECE':>10s}{'sharpness':>11s}")
    swept = sweep_evidence_scale([1.5, 6.0, 14.0, 20.0, 50.0], n=3000)
    best_scale, best = min(swept, key=lambda sr: sr[1].crps)
    for scale, rep in swept:
        marker = "  <- default" if scale == 1.5 else ("  <- optimum" if scale == best_scale else "")
        print(
            f"  {scale:>7.1f}{rep.crps:>9.4f}{rep.pit_deviation:>9.3f}"
            f"{rep.tail_ece:>10.3f}{rep.sharpness:>11.1f}{marker}"
        )
    print(
        f"\n  Calibration is maximized around evidence_scale ~{best_scale:.0f} — "
        f"the default is ~10x too\n  timid — and degrades past it as "
        f"overconfidence sets in. The residual PIT\n  deviation is mean bias, "
        f"which only the sigmoid calibration can fix.\n"
    )

    # ------------------------------------------------------------------
    # 3. Deceptive traffic
    # ------------------------------------------------------------------
    beliefs, outcomes = fidelity_run(3000, archetypes=[Archetype.DECEPTIVE], seed=0)
    adversarial = calibrate(beliefs, outcomes)
    print("Same proxy on deceptive traffic — PIT histogram (uniform iff calibrated):")
    for line in pit_bar(adversarial.pit_freqs):
        print(f"    {line}")
    print(
        f"\n  CRPS {adversarial.crps:.4f} (vs {report.crps:.4f} faithful), "
        f"tail ECE {adversarial.tail_ece:.3f}. All the PIT mass\n  piles into "
        "the bottom bin: outcomes land below the belief's entire support.\n"
        "  Miscalibration this localized is a fingerprint, not noise.\n"
    )

    # ------------------------------------------------------------------
    # 4. Governance as calibrator
    # ------------------------------------------------------------------
    population = {Archetype.HONEST: 4, Archetype.MEDIOCRE: 3, Archetype.DECEPTIVE: 3}
    mean_gov = TailMassGovernor(tau=0.4, max_tail_mass=1.0, min_mean=0.5)
    tail_gov = TailMassGovernor(
        tau=0.4, max_tail_mass=0.2, min_mean=None, defer_below_concentration=20.0
    )
    print("Calibration of the showdown logs (mixed traffic, same seed):")
    print(f"  {'':12s}{'raw proxy tail ECE':>20s}{'decision tail ECE':>19s}")
    for name, gov in [("mean gov", mean_gov), ("tail gov", tail_gov)]:
        result = Simulation(
            make_population(population), gov, SimulationConfig(seed=7)
        ).run()
        decision = calibrate_interactions(result.interactions)
        raw = calibrate_interactions(result.interactions, use_proxy_belief=True)
        print(f"  {name:12s}{raw.tail_ece:>20.3f}{decision.tail_ece:>19.3f}")
    print(
        "\n  The raw proxy is equally miscalibrated in both worlds. Audits and "
        "reputation\n  pooling are what turn it into an honest lever — "
        "governance doesn't just use\n  the beliefs, it calibrates them."
    )


if __name__ == "__main__":
    main()
