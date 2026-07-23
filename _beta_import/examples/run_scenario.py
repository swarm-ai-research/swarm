"""Run a scenario file — a whole governed experiment specified in YAML.

A scenario file wires up the population, proxy, payoff, governor, optional lever
stack, and optional adaptive controller. Identical files with identical seeds
produce identical runs, so an experiment is version-controllable and replayable.

Usage:
    python examples/run_scenario.py [path/to/scenario.yaml]

Defaults to scenarios/volume_forgery_defended.yaml.
"""

import sys
from pathlib import Path

import numpy as np

from beta_swarm import Archetype, load_scenario

DEFAULT = Path(__file__).resolve().parent.parent / "scenarios" / "volume_forgery_defended.yaml"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    config = load_scenario(path)
    simulation = config.build_simulation()
    result = simulation.run()

    print(f"scenario: {config.scenario_id}")
    if config.description:
        print(f"  {config.description.strip()}")
    print(f"  seed {config.seed}, {config.simulation.n_epochs} epochs, "
          f"{len(simulation.population)} agents\n")

    accepted = [i for i in result.interactions if i.accepted]
    print(f"  interactions: {len(result.interactions)}  accepted: {len(accepted)}")
    print(f"  total welfare: {result.total_welfare:+.1f}")
    print(f"  realized toxicity (accepted): {result.overall_toxicity:.3f}")

    present = {
        i.metadata.get("archetype")
        for i in result.interactions
        if not i.metadata.get("blocked")
    }
    print("\n  acceptance by archetype:")
    for archetype in Archetype:
        if archetype.value in present:
            print(f"    {archetype.value:12s}{result.acceptance_rate(archetype):>6.0%}")

    if config.governance_stack is not None:
        blocked = sum(i.metadata.get("blocked", False) for i in result.interactions)
        print(f"\n  lever stack active: {blocked} interactions blocked "
              f"({len(config.governance_stack.levers)} levers)")
    if config.adaptive is not None:
        late = np.mean([e.realized_toxicity for e in result.epoch_reports[-15:]])
        print(f"\n  adaptive controller: governor.{config.adaptive.parameter} tuned to "
              f"{getattr(simulation.governor, config.adaptive.parameter):.3f}, "
              f"late toxicity {late:.3f} (target {config.adaptive.target_toxicity})")


if __name__ == "__main__":
    main()
