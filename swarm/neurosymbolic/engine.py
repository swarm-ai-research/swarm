"""A small probabilistic Datalog engine (the "Scallop layer").

This is a self-contained, dependency-free implementation of the slice of
Scallop that the SWARM framework needs: probabilistic facts, rules with
variable-binding joins, **recursion**, **stratified negation**, **aggregation**,
and probability propagation through a pluggable
:class:`~swarm.neurosymbolic.provenance.Provenance`.

A perception layer emits noisy *probabilistic atomic facts* — for embodied
agents the :mod:`~swarm.neurosymbolic.perceiver` turns trajectories into
``near(a, t)::0.8``; for LLM agents the :mod:`~swarm.neurosymbolic.traces`
layer lifts tool calls / messages / errors into relations. Rules then compose
those atoms into higher-level behaviour relations, with probabilities flowing
through every join, recursion, and negation.

Design
------
- A **term** is either a :class:`Var` (logic variable) or a ground constant
  (``str``/``int``/``float``). ``Var("_")`` is an anonymous wildcard: it
  matches anything and never binds, and two ``"_"`` occurrences are
  independent.
- A **rule** is ``head :- l0, l1, ...`` where each literal is a positive
  :class:`Atom` or a negated :class:`Not`. Every variable in the head and in
  every negated literal must appear in some positive body atom (range
  restriction / safety).
- **Negation** ``Not(atom)`` contributes ``1 - p`` where ``p`` is the
  (finalised) probability of the ground negated atom.
- **Aggregation** (``count``/``sum``/``min``/``max``) summarises a body over
  groups, e.g. "how many times was this tool called with these args".
- Evaluation is **stratified**: relations that are negated or aggregated are
  computed to completion in a lower stratum before the relation that depends on
  them. Within a stratum the naive least-fixpoint iterates to convergence; with
  the idempotent default provenance (``plus = max``) recursion terminates at a
  unique fixpoint. Purely-positive non-aggregating programs form a single
  stratum and behave exactly like plain recursive Datalog.

The engine is deliberately small and readable rather than fast; programs here
operate on short traces (tens to hundreds of steps).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

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

    def has_wildcard(self) -> bool:
        return any(isinstance(t, Var) and t.is_wildcard for t in self.terms)


@dataclass(frozen=True)
class Not:
    """A negated body literal: holds with probability ``1 - p(atom)``.

    The negated atom must be ground once the positive body atoms are bound
    (all its variables must appear in a positive atom) and may not contain a
    wildcard — negation-as-failure is over a *specific* ground fact, not an
    existential. Express "no X exists such that ..." by defining a positive
    auxiliary relation and negating that.
    """

    atom: Atom

    def vars(self) -> List[Var]:
        return self.atom.vars()


BodyLiteral = Union[Atom, Not]


@dataclass(frozen=True)
class Rule:
    """``head :- body`` — head is implied by the conjunction of body literals.

    Positive atoms are stored before negated literals so that, during the join,
    every variable a negation refers to is already bound.
    """

    head: Atom
    body: Tuple[BodyLiteral, ...]

    def __init__(self, head: Atom, *body: BodyLiteral) -> None:
        object.__setattr__(self, "head", head)
        object.__setattr__(self, "body", _order_body(body))


@dataclass(frozen=True)
class Aggregate:
    """``out(group..., result) :- aggregate op of `over` over body, by group``.

    Produces, for each distinct binding of ``group_vars`` in the body's
    solutions, one fact ``out_relation(*group_values, result)`` with
    probability 1, where ``result`` is:

    - ``count`` — the number of distinct ``over`` values in the group,
    - ``sum`` / ``min`` / ``max`` — the corresponding reduction of the numeric
      ``over`` values.
    """

    out_relation: str
    op: str
    over: Optional[Var]
    group_vars: Tuple[Var, ...]
    body: Tuple[BodyLiteral, ...]


def _order_body(body: Sequence[BodyLiteral]) -> Tuple[BodyLiteral, ...]:
    """Positive atoms first, negated literals last (so negation vars are bound)."""
    positives = [b for b in body if isinstance(b, Atom)]
    negatives = [b for b in body if isinstance(b, Not)]
    return tuple(positives) + tuple(negatives)


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
    """A probabilistic Datalog program: extensional facts + rules + aggregates.

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
        self.aggregates: List[Aggregate] = []
        # Raw per-call contributions, kept so :meth:`run` can combine duplicate
        # extensional facts under the *run's* provenance rather than baking in a
        # fixed one at insertion time. ``self.edb`` is the combined input view.
        self._edb_contributions: List[Tuple[str, GroundTuple, float]] = []

    # -- construction -------------------------------------------------------
    def fact(self, relation: str, *terms: Const, p: float = 1.0) -> "Program":
        """Add an extensional (input) probabilistic fact."""
        gt = tuple(terms)
        pc = clamp01(p)
        self._edb_contributions.append((relation, gt, pc))
        self.edb._bump(relation, gt, pc, DEFAULT_PROVENANCE)
        return self

    def rule(self, head: Atom, *body: BodyLiteral) -> "Program":
        """Add a rule ``head :- body``. Raises if the rule is unsafe."""
        positive_vars = {
            v.name for lit in body if isinstance(lit, Atom) for v in lit.vars()
        }
        for v in head.vars():
            if v.name not in positive_vars:
                raise ValueError(
                    f"Unsafe rule: head variable {v.name!r} does not appear in a "
                    "positive body atom"
                )
        for lit in body:
            if isinstance(lit, Not):
                if lit.atom.has_wildcard():
                    raise ValueError(
                        "Unsafe rule: negated atom may not contain a wildcard"
                    )
                for v in lit.vars():
                    if v.name not in positive_vars:
                        raise ValueError(
                            f"Unsafe rule: negated variable {v.name!r} does not "
                            "appear in a positive body atom"
                        )
        self.rules.append(Rule(head, *body))
        return self

    def aggregate(
        self,
        out_relation: str,
        op: str,
        *body: BodyLiteral,
        group_by: Sequence[Var] = (),
        over: Optional[Var] = None,
    ) -> "Program":
        """Add an aggregation ``out_relation(group..., result) :- op over body``."""
        if op not in ("count", "sum", "min", "max"):
            raise ValueError(f"unknown aggregation op {op!r}")
        if op != "count" and over is None:
            raise ValueError(f"aggregation {op!r} requires an `over` variable")
        if op == "count" and over is None:
            raise ValueError("count requires an `over` variable to count distinctly")
        self.aggregates.append(
            Aggregate(out_relation, op, over, tuple(group_by), _order_body(body))
        )
        return self

    # -- evaluation ---------------------------------------------------------
    def run(
        self,
        provenance: Optional[Provenance] = None,
        *,
        max_iter: int = 1000,
        epsilon: float = 1e-9,
    ) -> FactSet:
        """Compute the stratified least fixpoint and return all derived facts.

        ``provenance`` must have an idempotent ``plus`` (the default
        :class:`MaxTimesProvenance` does) for recursion within a stratum to
        terminate at a unique fixpoint.
        """
        prov = provenance or DEFAULT_PROVENANCE
        strata = self._strata()
        max_s = max(strata.values(), default=0)

        db = FactSet()
        # Seed from the raw contributions so duplicate extensional facts are
        # combined with the requested provenance's ``plus`` (consistent with
        # how derived facts are combined), not a fixed default.
        for relation, gt, p in self._edb_contributions:
            db._bump(relation, gt, p, prov)

        for s in range(max_s + 1):
            # Aggregates first: their bodies live in strictly lower strata and
            # are already finalised, so each is a one-shot computation.
            for agg in self.aggregates:
                if strata[agg.out_relation] == s:
                    self._eval_aggregate(agg, db, prov)

            rules_s = [r for r in self.rules if strata[r.head.relation] == s]
            if not rules_s:
                continue

            for _ in range(max_iter):
                # Read-then-write: snapshot derivations, then apply, so recursive
                # rules don't mutate a relation while a join iterates it.
                derived: List[Tuple[str, GroundTuple, float]] = []
                for r in rules_s:
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

    def _strata(self) -> Dict[str, int]:
        """Assign each relation a stratum number.

        Positive dependencies require ``stratum(head) >= stratum(body)``;
        negation and aggregation require ``stratum(head) > stratum(body)``.
        Raises if the program is not stratifiable (a negation/aggregation cycle).
        """
        relations: set[str] = set(self.edb.relations())
        edges: List[Tuple[str, str, int]] = []  # (src, dst, weight)

        def note(rel: str) -> None:
            relations.add(rel)

        for r in self.rules:
            note(r.head.relation)
            for lit in r.body:
                if isinstance(lit, Atom):
                    note(lit.relation)
                    edges.append((lit.relation, r.head.relation, 0))
                else:
                    note(lit.atom.relation)
                    edges.append((lit.atom.relation, r.head.relation, 1))
        for agg in self.aggregates:
            note(agg.out_relation)
            for lit in agg.body:
                rel = lit.relation if isinstance(lit, Atom) else lit.atom.relation
                note(rel)
                edges.append((rel, agg.out_relation, 1))

        stratum: Dict[str, int] = dict.fromkeys(relations, 0)
        for _ in range(len(relations) + 1):
            changed = False
            for src, dst, w in edges:
                if stratum[dst] < stratum[src] + w:
                    stratum[dst] = stratum[src] + w
                    changed = True
            if not changed:
                break
        else:
            raise ValueError(
                "program is not stratifiable: negation/aggregation forms a cycle"
            )
        return stratum

    def _eval_aggregate(
        self, agg: Aggregate, db: FactSet, prov: Provenance
    ) -> None:
        groups: Dict[GroundTuple, List[Const]] = {}
        for binding, _prob in self._join(agg.body, db, prov):
            key = tuple(_subst(v, binding) for v in agg.group_vars)
            val = _subst(agg.over, binding) if agg.over is not None else None
            groups.setdefault(key, []).append(val)  # type: ignore[arg-type]
        for key, vals in groups.items():
            result = _apply_agg(agg.op, vals)
            db._bump(agg.out_relation, tuple(key) + (result,), 1.0, prov)

    def _join(
        self,
        body: Tuple[BodyLiteral, ...],
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
            lit = body[i]
            if isinstance(lit, Not):
                gt = tuple(_subst(t, binding) for t in lit.atom.terms)
                factor = clamp01(1.0 - db.prob(lit.atom.relation, *gt))
                if factor > 0.0:
                    yield from recurse(i + 1, binding, prov.times(acc, factor))
                return
            for gt, fact_p in db.get(lit.relation).items():
                new_binding = _unify(lit.terms, gt, binding)
                if new_binding is None:
                    continue
                yield from recurse(i + 1, new_binding, prov.times(acc, fact_p))

        yield from recurse(0, {}, prov.one())


def _apply_agg(op: str, vals: Sequence[Const]) -> Const:
    if op == "count":
        return len(set(vals))
    # sum/min/max require numeric values; fail loudly rather than silently
    # dropping non-numerics or surfacing an opaque min()/max() error.
    non_numeric = [v for v in vals if not isinstance(v, (int, float))]
    if non_numeric:
        raise ValueError(
            f"aggregation {op!r} requires numeric `over` values, got non-numeric "
            f"{non_numeric[0]!r}"
        )
    if not vals:
        raise ValueError(f"aggregation {op!r} has no values to reduce")
    numeric = list(vals)
    if op == "sum":
        return sum(numeric)  # type: ignore[arg-type]
    if op == "min":
        return min(numeric)
    if op == "max":
        return max(numeric)
    raise ValueError(f"unknown aggregation op {op!r}")  # pragma: no cover


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
