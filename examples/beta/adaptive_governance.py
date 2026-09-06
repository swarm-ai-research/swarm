"""Adaptive governance: tune the risk budget to a policy target, reversibly.

A fixed ``max_tail_mass`` is a guess, and a wrong guess is costly in both
directions: too loose and realized downside risk runs over your policy target;
too tight and you turn away good work for no safety gain. The adaptive
controller treats the threshold as something to *learn* from realized outcomes,
and treats every change as a reversible experiment — apply it, then crystallize
it only if outcomes sustain, else revert.

Here a risk appetite is fixed as a target on realized toxicity
(``E[1 - v_true | accepted]``, ground truth). The population has honest agents,
a lone deceiver, and a spread of *borderline* agents whose true quality ranges
from poor to good — so how strict the budget is smoothly controls how much
realized risk gets in. The controller is dropped onto two badly misconfigured
governors and one already-correct one, all with the same target.

Run:  python examples/adaptive_governance.py
"""

import numpy as np

from beta_swarm import (
    Archetype,
    Simulation,
    SimulationConfig,
    TailMassGovernor,
    make_population,
)
from beta_swarm.adaptive import AdaptiveController
from beta_swarm.agents import SimAgent

TARGET = 0.37
SEED = 7
EPOCHS = 70


def population() -> list[SimAgent]:
    """Honest + a deceiver + 8 borderline agents spanning quality 0.35–0.68."""
    pop = make_population({Archetype.HONEST: 4, Archetype.DECEPTIVE: 2})
    for i, mean in enumerate(np.linspace(0.35, 0.68, 8)):
        conc = 12.0
        pop.append(SimAgent(f"borderline-{i}", Archetype.MEDIOCRE, mean * conc, (1 - mean) * conc))
    return pop


def governor(max_tail_mass: float) -> TailMassGovernor:
    return TailMassGovernor(
        tau=0.4, max_tail_mass=max_tail_mass, min_mean=None, defer_below_concentration=20.0
    )


def late_toxicity(result) -> float:
    return float(np.mean([e.realized_toxicity for e in result.epoch_reports[-15:]]))


def main() -> None:
    print(f"Policy target: realized toxicity = {TARGET:.2f}. Same target, three starts.\n")
    print(f"  {'start':22s}{'max_tail_mass':>16s}{'realized toxicity':>20s}")
    print("  " + "-" * 58)

    for start, label in [(0.55, "misconfigured loose"), (0.05, "misconfigured tight"), (0.12, "already correct")]:
        # Static: leave the threshold fixed at the (mis)configured value.
        static = Simulation(population(), governor(start), SimulationConfig(seed=SEED, n_epochs=EPOCHS)).run()

        # Adaptive: let the controller retune it toward the target.
        gov = governor(start)
        controller = AdaptiveController(target_toxicity=TARGET, band=0.01)
        adaptive = Simulation(
            population(), gov, SimulationConfig(seed=SEED, n_epochs=EPOCHS), controller=controller
        ).run()

        print(
            f"  {label:22s}{start:>7.2f}{' (static)':>9s}{late_toxicity(static):>20.3f}"
        )
        print(
            f"  {'-> adaptive':22s}{gov.max_tail_mass:>7.2f}{' (tuned)':>9s}"
            f"{late_toxicity(adaptive):>20.3f}"
            f"    [{controller.n_crystallized} kept, {controller.n_reverted} reverted]"
        )
        print()

    print(
        "  A static budget misses the target in both directions — too loose sits "
        "at 0.40\n  (over the risk appetite), too tight at 0.34 (turning away "
        "good work for no gain).\n  The controller pulls both to the ~0.37 target "
        "from opposite starts, and leaves the\n  already-correct governor near "
        "where it began. The reverted proposals are the\n  discipline that makes "
        "it safe: a step that overshoots the target is rolled back,\n  so the "
        "threshold only keeps changes that earned their keep."
    )


if __name__ == "__main__":
    main()
