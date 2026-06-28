"""A small probabilistic Datalog engine (the "Scallop layer").

This is a self-contained, dependency-free implementation of the slice of
Scallop that the SWARM framework needs: probabilistic facts, rules with
variable-binding joins, **recursion**, and probability propagation through a
pluggable :class:`~swarm.neurosymbolic.provenance.Provenance`.

The neural layer (see :mod:`swarm.neurosymbolic.perceiver`) emits noisy
*probabilistic atomic facts* — ``near(a, t)::0.8`` — and rules defined here
compose them into higher-level behaviour relations, with probabilities flowing
through every join and recursion.

Design
------
- A **term** is either a :class:`Var` (logic variable) or a ground constant
  (``str``/``int``/``float``). ``Var("_")`` is an anonymous wildcard: it
  matches anything and never binds, and two ``"_"`` occurrences are
  independent.
- A **rule** is ``head :- body[0], body[1], ...`` — a conjunction. Every
  variable in ``head`` must appear in ``body`` (range restriction / safety).
- Evaluation is a naive least-fixpoint: rules fire until no fact's probability
  increases by more than ``epsilon``. With the idempotent default provenance
  (``plus = max``) this converges to a unique least fixpoint even with
  recursion.

The engine is deliberately small and readable rather than fast; behaviour
programs here operate on short trajectories (tens to hundreds of timesteps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple, Union

from swarm.neurosymbolic.provenance import (
    DEFAULT_PROVENANCE,
    Provenance,
    clamp01,
)

# A ground constant.
Const = Union[str, int, float]


@dataclass(frozen=True)
class Var:
    """A logic variable. ``Var("_")`` is an anonymous wildcard."""

    name: str

    @property
    def is_wildcard(self) -> bool:
        return self.name == "_"


Term = Union[Var, Const]
GroundTuple = Tuple[Const, ...]


@dataclass(frozen=True)
class Atom:
    """A relation applied to terms, e.g. ``near(Var('A'), Var('T'), 3)``."""

    relation: str
    terms: Tuple[Term, ...]

    def __init__(self, relation: str, *terms: Term) -> None:
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "terms", tuple(terms))

    def vars(self) -> List[Var]:
        return [t for t in self.terms if isinstance(t, Var) and not t.is_wildcard]


@dataclass(frozen=True)
class Rule:
    """``head :- body`` — head is implied by the conjunction of body atoms."""

    head: Atom
    body: Tuple[Atom, ...]

    def __init__(self, head: Atom, *body: Atom) -> None:
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "body", tuple(body))


class FactSet:
    """The set of derived facts: ``relation -> {ground_tuple: probability}``."""

    def __init__(self) -> None:
        self._facts: Dict[str, Dict[GroundTuple, float]] = {}

    def get(self, relation: str) -> Dict[GroundTuple, float]:
        return self._facts.get(relation, {})

    def items(self) -> Iterator[Tuple[str, GroundTuple, float]]:
        for relation, rows in self._facts.items():
            for gt, p in rows.items():
                yield relation, gt, p

    def prob(self, relation: str, *terms: Const) -> float:
        """Probability of a ground fact, or ``0.0`` if not derived."""
        return self._facts.get(relation, {}).get(tuple(terms), 0.0)

    def relations(self) -> List[str]:
        return list(self._facts.keys())

    def _bump(
        self,
        relation: str,
        gt: GroundTuple,
        p: float,
        prov: Provenance,
        epsilon: float = 0.0,
    ) -> bool:
        """Combine ``p`` into an existing fact via ``prov.plus``.

        Returns True if the fact's probability increased by more than
        ``epsilon`` (used to detect fixpoint convergence).
        """
        rows = self._facts.setdefault(relation, {})
        old = rows.get(gt, prov.zero())
        new = clamp01(prov.plus(old, p))
        rows[gt] = new
        return new > old + epsilon


class Program:
    """A probabilistic Datalog program: extensional facts + rules.

    Example
    -------
    >>> prog = Program()
    >>> prog.fact("near", "a", "t", p=0.8)
    >>> prog.rule(Atom("close", Var("X"), Var("Y")), Atom("near", Var("X"), Var("Y")))
    >>> db = prog.run()
    >>> round(db.prob("close", "a", "t"), 3)
    0.8
    """

    def __init__(self) -> None:
        self.edb = FactSet()
        self.rules: List[Rule] = []

    # -- construction -------------------------------------------------------
    def fact(self, relation: str, *terms: Const, p: float = 1.0) -> "Program":
        """Add an extensional (input) probabilistic fact."""
        self.edb._bump(relation, tuple(terms), clamp01(p), DEFAULT_PROVENANCE)
        return self

    def rule(self, head: Atom, *body: Atom) -> "Program":
        """Add a rule ``head :- body``. Raises if the rule is unsafe."""
        body_vars = {v.name for atom in body for v in atom.vars()}
        for v in head.vars():
            if v.name not in body_vars:
                raise ValueError(
                    f"Unsafe rule: head variable {v.name!r} does not appear in the body"
                )
        self.rules.append(Rule(head, *body))
        return self

    # -- evaluation ---------------------------------------------------------
    def run(
        self,
        provenance: Optional[Provenance] = None,
        *,
        max_iter: int = 1000,
        epsilon: float = 1e-9,
    ) -> FactSet:
        """Compute the least fixpoint and return all derived facts.

        Seeds the result with the extensional facts, then fires every rule to
        convergence. ``provenance`` must have an idempotent ``plus`` (the
        default :class:`MaxTimesProvenance` does) for recursion to terminate
        at a unique fixpoint.
        """
        prov = provenance or DEFAULT_PROVENANCE
        db = FactSet()
        # Seed with extensional facts.
        for relation, gt, p in self.edb.items():
            db._bump(relation, gt, p, prov)

        for _ in range(max_iter):
            # Read-then-write: collect every derivation against the current db
            # snapshot first, then apply. This keeps recursive rules from
            # mutating a relation while a join is iterating it.
            derived: List[Tuple[str, GroundTuple, float]] = []
            for r in self.rules:
                for binding, prob in self._join(r.body, db, prov):
                    gt = tuple(_subst(t, binding) for t in r.head.terms)
                    derived.append((r.head.relation, gt, prob))
            changed = False
            for relation, gt, prob in derived:
                if db._bump(relation, gt, prob, prov, epsilon):
                    changed = True
            if not changed:
                break
        else:  # pragma: no cover - safety valve only
            raise RuntimeError(
                "Datalog fixpoint did not converge; check provenance idempotency"
            )
        return db

    def _join(
        self,
        body: Tuple[Atom, ...],
        db: FactSet,
        prov: Provenance,
    ) -> Iterator[Tuple[Dict[str, Const], float]]:
        """Yield ``(binding, probability)`` for every way to satisfy ``body``."""

        def recurse(
            i: int, binding: Dict[str, Const], acc: float
        ) -> Iterator[Tuple[Dict[str, Const], float]]:
            if i == len(body):
                yield dict(binding), acc
                return
            atom = body[i]
            for gt, fact_p in db.get(atom.relation).items():
                new_binding = _unify(atom.terms, gt, binding)
                if new_binding is None:
                    continue
                yield from recurse(i + 1, new_binding, prov.times(acc, fact_p))

        yield from recurse(0, {}, prov.one())


def _subst(term: Term, binding: Dict[str, Const]) -> Const:
    """Resolve a (necessarily bound) term to a ground constant."""
    if isinstance(term, Var):
        return binding[term.name]
    return term


def _unify(
    terms: Tuple[Term, ...],
    gt: GroundTuple,
    binding: Dict[str, Const],
) -> Optional[Dict[str, Const]]:
    """Try to match a pattern against a ground tuple, extending ``binding``.

    Returns the extended binding, or ``None`` on mismatch. The input binding is
    not mutated.
    """
    if len(terms) != len(gt):
        return None
    out = dict(binding)
    for term, value in zip(terms, gt, strict=True):
        if isinstance(term, Var):
            if term.is_wildcard:
                continue
            bound = out.get(term.name)
            if bound is None:
                out[term.name] = value
            elif bound != value:
                return None
        elif term != value:
            return None
    return out
