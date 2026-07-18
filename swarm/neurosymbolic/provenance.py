"""Probability-propagation provenance for the neurosymbolic engine.

A *provenance* defines how probabilities flow through Datalog inference:

- ``times`` combines the probabilities of body atoms in a single rule
  application (logical AND — a conjunction of evidence).
- ``plus`` combines the probabilities of *alternative* derivations of the
  same fact (logical OR — independent ways to prove the same thing).

The recursive fixpoint in :mod:`swarm.neurosymbolic.engine` requires ``plus``
to be **idempotent** (``plus(a, a) == a``) so that re-deriving a fact across
iterations does not inflate its probability without bound. The default
:class:`MaxTimesProvenance` satisfies this — it is the *top-1 proof* /
Viterbi semiring, equivalent to Scallop's ``topkproofs`` with ``k = 1``:
each fact's probability is the probability of its single most-likely proof.

For combining *distinct* proofs at read-out time (where each proof is
enumerated exactly once, so idempotency is not needed) use :func:`noisy_or`,
which assumes the proofs are independent — the same assumption Scallop's
``addmult`` provenance makes.

Probabilities are kept in ``[0, 1]`` throughout, matching the framework-wide
invariant that any surfaced ``p`` is a valid probability.
"""

from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable


def clamp01(p: float) -> float:
    """Clamp a probability into ``[0, 1]`` (defensive; preserves the invariant)."""
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


@runtime_checkable
class Provenance(Protocol):
    """How probabilities combine during inference.

    Implementations must keep results in ``[0, 1]`` and ``plus`` must be
    idempotent, commutative and monotone non-decreasing for the engine's
    least-fixpoint iteration to converge.
    """

    def one(self) -> float:
        """Conjunction identity — the probability of the empty (always-true) body."""

    def zero(self) -> float:
        """Disjunction identity — the probability of a fact with no derivation."""

    def times(self, a: float, b: float) -> float:
        """Combine two body-atom probabilities (logical AND)."""

    def plus(self, a: float, b: float) -> float:
        """Combine two alternative-derivation probabilities (logical OR)."""


class MaxTimesProvenance:
    """Top-1-proof (Viterbi) provenance: ``times`` = product, ``plus`` = max.

    Each fact's probability becomes the probability of its single most-likely
    proof. ``plus`` (``max``) is idempotent, so this is the safe default for
    the recursive fixpoint. Equivalent to Scallop ``topkproofs`` with ``k=1``.
    """

    def one(self) -> float:
        return 1.0

    def zero(self) -> float:
        return 0.0

    def times(self, a: float, b: float) -> float:
        return clamp01(a * b)

    def plus(self, a: float, b: float) -> float:
        return a if a >= b else b


def noisy_or(probs: Iterable[float]) -> float:
    """Independent-OR over a set of *distinct* proof probabilities.

    ``noisy_or([p1, p2, ...]) = 1 - prod(1 - pi)``.

    Assumes the proofs are independent. Use only over an explicitly enumerated
    set where each proof appears exactly once (e.g. read-out aggregation across
    distinct temporal runs) — it is **not** idempotent and must not be used as
    the engine's fixpoint ``plus``.
    """
    acc = 1.0
    for p in probs:
        acc *= (1.0 - clamp01(p))
    return clamp01(1.0 - acc)


DEFAULT_PROVENANCE = MaxTimesProvenance()
