"""Demo: neurosymbolic behaviour classification (neural perception + Scallop rules).

Runs three hand-built trajectories — a chaser, a fleer, and a forager — through
the neural layer (probabilistic atoms) and the Scallop layer (Datalog rules with
recursion) and prints the resulting behaviour probabilities and the temporal
episodes that support them.

    python examples/neurosymbolic_behavior.py
"""

from __future__ import annotations

from swarm.neurosymbolic import (
    Trajectory,
    classify_behaviors,
    to_scallop_program,
)


def _chaser() -> Trajectory:
    # Walks straight at a stationary target, closing distance each step.
    return Trajectory(
        agent="hunter",
        positions=[(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)],
        targets={"prey": [(10.0, 0.0)]},
    )


def _fleer() -> Trajectory:
    # A nearby threat is detected; the agent runs directly away from it.
    return Trajectory(
        agent="rabbit",
        positions=[(0.0, 0.0), (-1.0, 0.0), (-2.0, 0.0), (-3.0, 0.0), (-4.0, 0.0)],
        targets={"fox": [(1.0, 0.0)]},
    )


def _forager() -> Trajectory:
    # Alternates wandering (search) with darts toward a food cache (approach).
    return Trajectory(
        agent="bee",
        positions=[
            (0.0, 0.0), (0.5, 1.0), (2.0, 1.5), (1.0, 2.0), (3.0, 2.5), (2.0, 3.0)
        ],
        targets={"flower": [(4.0, 4.0)]},
    )


def main() -> None:
    for traj in (_chaser(), _fleer(), _forager()):
        scores = classify_behaviors(traj)
        top, prob = scores.top()
        print(f"\n{traj.agent}: top behaviour = {top} ({prob:.2f})")
        for behaviour, p in sorted(scores.scores.items(), key=lambda kv: -kv[1]):
            line = f"  {behaviour:<10} {p:.3f}"
            episodes = scores.episodes.get(behaviour, [])
            if episodes:
                ep = episodes[0]
                span = f"[{ep.start}->{ep.end}]" if ep.end != ep.start else f"[{ep.start}]"
                line += f"   best episode {span} p={ep.probability:.3f}"
            print(line)

    print("\n--- equivalent native Scallop program ---")
    print(to_scallop_program())


if __name__ == "__main__":
    main()
