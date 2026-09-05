"""Tests for WardSet: bounded-by-construction and never-widen composition."""

from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from swarm.agentgit.identity import AgentKeypair, DelegationChain, sign_link
from swarm.agentgit.wards import (
    KNOWN_DONE_GATES,
    WardSet,
    compose,
    compose_chain,
    from_chain,
    never_widens,
)

PERMS = ["read", "write", "test", "lint", "vcs", "open_pr"]


def _root(**kw) -> WardSet:
    base = {
        "max_turns": 8,
        "max_depth": 2,
        "max_children": 4,
        "timeout_s": 600.0,
        "permissions": frozenset(PERMS),
        "negative_spec": ("single-seed result presented as general",),
    }
    base.update(kw)
    return WardSet(**base)


class TestBoundedByConstruction:
    def test_defaults_are_unbounded(self):
        errors = WardSet().errors()
        assert any("max_turns" in e for e in errors)
        assert any("max_depth" in e for e in errors)
        assert not WardSet().is_bounded

    def test_root_is_bounded(self):
        assert _root().errors() == []

    def test_unknown_done_gate_rejected(self):
        errors = _root(done_gate="vibes").errors()
        assert any("done_gate" in e for e in errors)

    def test_empty_done_gate_rejected(self):
        assert any("done_gate" in e for e in _root(done_gate="").errors())

    def test_negative_numbers_rejected(self):
        errors = _root(max_turns=0, max_depth=-1, max_children=-2, timeout_s=-1.0).errors()
        assert len(errors) == 4

    def test_allowlist_cannot_exceed_own_permissions(self):
        ws = _root(permissions=frozenset({"read"}), child_permission_allowlist=frozenset({"read", "vcs"}))
        assert any("allowlist" in e for e in ws.errors())

    def test_roundtrip(self):
        ws = _root(child_permission_allowlist=frozenset({"read"}), child_max_depth_ceiling=1)
        assert WardSet.from_dict(ws.to_dict()) == ws


class TestCompose:
    def test_numeric_min_boolean_or_set_intersection(self):
        parent = _root(require_artifact_on_done=False)
        child = WardSet(
            max_turns=20,
            max_depth=5,
            max_children=1,
            timeout_s=30.0,
            require_artifact_on_done=True,
            permissions=frozenset({"read", "test"}),
            negative_spec=("mock numbers without cordon",),
        )
        r = compose(parent, child)
        assert r.accepted, r.errors
        w = r.wards
        assert w.max_turns == 8
        assert w.max_depth == 1  # min(parent-1, child)
        assert w.max_children == 1
        assert w.timeout_s == 30.0
        assert w.require_artifact_on_done is True
        assert w.permissions == frozenset({"read", "test"})
        assert w.negative_spec == (
            "single-seed result presented as general",
            "mock numbers without cordon",
        )

    def test_depth_is_consumed_per_level(self):
        leaf = WardSet(max_turns=2, max_depth=9, permissions=frozenset({"read"}))
        r = compose_chain(_root(max_depth=2), [leaf, leaf])
        assert r.accepted, r.errors
        assert r.wards.max_depth == 0
        assert not r.wards.can_spawn
        r3 = compose_chain(_root(max_depth=2), [leaf, leaf, leaf])
        assert not r3.accepted
        assert any("may not spawn" in e for e in r3.errors)

    def test_widened_permissions_rejected_not_rewritten(self):
        parent = _root(permissions=frozenset({"read"}))
        child = WardSet(max_turns=1, max_depth=0, permissions=frozenset({"read", "vcs"}))
        r = compose(parent, child)
        assert not r.accepted and r.wards is None
        assert any("parent lacks" in e for e in r.errors)

    def test_allowlist_and_denylist(self):
        parent = _root(
            child_permission_allowlist=frozenset({"read", "test"}),
            child_permission_denylist=frozenset({"vcs"}),
        )
        ok = compose(parent, WardSet(max_turns=1, max_depth=0, permissions=frozenset({"read"})))
        assert ok.accepted
        bad = compose(parent, WardSet(max_turns=1, max_depth=0, permissions=frozenset({"read", "lint"})))
        assert any("allowlist" in e for e in bad.errors)
        denied = compose(
            _root(child_permission_denylist=frozenset({"vcs"})),
            WardSet(max_turns=1, max_depth=0, permissions=frozenset({"vcs"})),
        )
        assert any("denylisted" in e for e in denied.errors)

    def test_ceilings_reject_rather_than_clamp(self):
        parent = _root(child_max_turns_ceiling=3, child_max_depth_ceiling=0)
        over = compose(parent, WardSet(max_turns=4, max_depth=0, permissions=frozenset({"read"})))
        assert any("exceeds ceiling" in e for e in over.errors)
        deep = compose(parent, WardSet(max_turns=3, max_depth=1, permissions=frozenset({"read"})))
        assert any("exceeds ceiling" in e for e in deep.errors)
        at = compose(parent, WardSet(max_turns=3, max_depth=0, permissions=frozenset({"read"})))
        assert at.accepted and at.wards.max_turns == 3

    def test_unbounded_parent_or_child_rejected(self):
        assert not compose(WardSet(), _root()).accepted
        r = compose(_root(), WardSet())
        assert any(e.startswith("child:") for e in r.errors)

    def test_leaf_parent_cannot_spawn(self):
        r = compose(_root(max_depth=0), WardSet(max_turns=1, max_depth=0))
        assert any("may not spawn" in e for e in r.errors)
        r2 = compose(_root(max_children=0), WardSet(max_turns=1, max_depth=0))
        assert any("may not spawn" in e for e in r2.errors)


# --- hypothesis: INV-6 -----------------------------------------------------

_opt_int = st.one_of(st.none(), st.integers(min_value=0, max_value=12))
_opt_float = st.one_of(st.none(), st.floats(min_value=0, max_value=1e4, allow_nan=False))
_perms = st.frozensets(st.sampled_from(PERMS))
_spec = st.lists(st.sampled_from(["a", "b", "c", "d"]), max_size=3).map(tuple)


@st.composite
def ward_sets(draw):
    perms = draw(_perms)
    allow = draw(st.one_of(st.none(), st.frozensets(st.sampled_from(sorted(perms) or ["read"]))))
    if allow is not None:
        allow = allow & perms
    return WardSet(
        max_turns=draw(st.one_of(st.none(), st.integers(min_value=1, max_value=12))),
        max_depth=draw(_opt_int),
        max_children=draw(_opt_int),
        timeout_s=draw(_opt_float),
        done_gate=draw(st.sampled_from(sorted(KNOWN_DONE_GATES))),
        require_artifact_on_done=draw(st.booleans()),
        permissions=perms,
        negative_spec=draw(_spec),
        child_permission_allowlist=allow,
        child_permission_denylist=draw(_perms),
        child_max_turns_ceiling=_opt_int_val(draw),
        child_max_depth_ceiling=_opt_int_val(draw),
    )


def _opt_int_val(draw):
    return draw(_opt_int)


class TestNeverWidens:
    @settings(max_examples=400)
    @given(parent=ward_sets(), child=ward_sets())
    def test_composed_never_widens_parent(self, parent, child):
        r = compose(parent, child)
        if not r.accepted:
            assert r.wards is None
            return
        assert never_widens(parent, r.wards) == [], never_widens(parent, r.wards)
        assert r.wards.is_bounded

    @settings(max_examples=200)
    @given(root=ward_sets(), decls=st.lists(ward_sets(), min_size=1, max_size=4))
    def test_chain_never_widens_root(self, root, decls):
        r = compose_chain(root, decls)
        if r.accepted:
            assert never_widens(root, r.wards) == []

    def test_predicate_detects_widening(self):
        parent = _root()
        wider = _root(max_turns=9, permissions=frozenset(PERMS) | {"deploy"}, negative_spec=())
        found = never_widens(parent, wider)
        assert any(f.startswith("max_turns") for f in found)
        assert any(f.startswith("permissions") for f in found)
        assert any(f.startswith("negative_spec") for f in found)


class TestFromChain:
    def test_valid_chain_supplies_permissions(self):
        human, agent = AgentKeypair.generate(), AgentKeypair.generate()
        link = sign_link(human, subject_did=agent.did, permissions=["read", "test"])
        ws = from_chain(DelegationChain([link]), max_turns=3, max_depth=0)
        assert ws.permissions == frozenset({"read", "test"})
        assert ws.is_bounded

    def test_expired_chain_grants_nothing(self):
        human, agent = AgentKeypair.generate(), AgentKeypair.generate()
        past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        link = sign_link(human, subject_did=agent.did, permissions=["read"], not_after=past)
        ws = from_chain(DelegationChain([link]), max_turns=3, max_depth=0)
        assert ws.permissions == frozenset()
