"""Tests for WardSet: bounded-by-construction and never-widen composition."""

import json
from datetime import datetime, timedelta, timezone

from hypothesis import given, settings
from hypothesis import strategies as st

from swarm.agentgit import __main__ as cli
from swarm.agentgit.coordination import read_claim_marker, read_claim_wards
from swarm.agentgit.identity import AgentKeypair, DelegationChain, sign_link
from swarm.agentgit.wards import (
    GATE_CLASS_DEFAULTS,
    KNOWN_DONE_GATES,
    STAMP_PREFIX,
    WardSet,
    claim_gate,
    compose,
    compose_chain,
    for_gate_class,
    format_stamp,
    from_chain,
    latest_stamp,
    never_widens,
    parse_stamp,
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


# --- stamps and the claim gate (illq.2) --------------------------------------



class TestStamps:
    def test_gate_class_defaults_are_bounded(self):
        for gc in GATE_CLASS_DEFAULTS:
            ws = for_gate_class(gc, negative_spec=("x",))
            assert ws.is_bounded, (gc, ws.errors())
        assert for_gate_class("G2").max_children == 0
        assert not for_gate_class("G2").can_spawn

    def test_unknown_gate_class(self):
        import pytest as _pytest

        with _pytest.raises(ValueError):
            for_gate_class("G9")

    def test_stamp_roundtrip_and_prose_survives(self):
        ws = for_gate_class("G0", negative_spec=("mock numbers without cordon",), permissions=("read",))
        text = "rationale: unblocks 3, betweenness 0.2\n" + format_stamp(ws) + "\nmore prose"
        assert text.count(STAMP_PREFIX) == 1
        assert parse_stamp(text) == ws

    def test_parse_takes_last_and_skips_malformed(self):
        a = for_gate_class("G0")
        b = for_gate_class("G2", negative_spec=("n",))
        text = "\n".join([format_stamp(a), "WARDS: {not json", "WARDS: [1,2]", format_stamp(b)])
        assert parse_stamp(text) == b
        assert parse_stamp("no stamp here") is None
        assert latest_stamp(["prose", format_stamp(a), "prose", format_stamp(b)]) == b
        assert latest_stamp([]) is None


class TestClaimGate:
    def test_no_stamp_passes_declaration_through(self):
        assert claim_gate(None, None).errors == []
        assert claim_gate(None, None).wards is None
        d = for_gate_class("G1")
        assert claim_gate(None, d).wards == d

    def test_no_declaration_inherits_track(self):
        t = for_gate_class("G1", negative_spec=("n",))
        r = claim_gate(t, None)
        assert r.errors == [] and r.wards == t

    def test_peer_claim_does_not_consume_depth(self):
        t = for_gate_class("G1")  # max_depth 1
        r = claim_gate(t, WardSet(max_turns=8, max_depth=1))
        assert r.accepted, r.errors
        assert r.wards.max_depth == 1

    def test_widening_declaration_refused(self):
        t = for_gate_class("G2", permissions=("read",))
        r = claim_gate(t, WardSet(max_turns=99, max_depth=0, permissions=frozenset({"read", "vcs"})))
        assert not r.accepted
        assert any("parent lacks" in e for e in r.errors)

    def test_narrowing_declaration_composes(self):
        t = for_gate_class("G0", negative_spec=("a",), permissions=("read", "test"))
        r = claim_gate(t, WardSet(max_turns=3, max_depth=0, permissions=frozenset({"read"}), negative_spec=("b",)))
        assert r.accepted
        assert r.wards.max_turns == 3 and r.wards.permissions == frozenset({"read"})
        assert r.wards.negative_spec == ("a", "b")


class TestClaimCLI:
    def _env(self, monkeypatch, tmp_path, track):
        monkeypatch.setenv("MAIN_REPO_ROOT", str(tmp_path))
        monkeypatch.delenv("AGENT_WARDS", raising=False)
        monkeypatch.setattr(cli, "load_bead_wards", lambda task_id, **kw: track)
        # never touch the real bd from tests
        monkeypatch.setattr(cli.subprocess, "run", lambda *a, **k: type("P", (), {"returncode": 0, "stdout": "", "stderr": ""})())
        return str(tmp_path / "coord.db")

    def test_unstamped_bead_claims_as_before(self, monkeypatch, tmp_path):
        db = self._env(monkeypatch, tmp_path, None)
        assert cli.main(["claim", "claim", "bead-1", "--agent", "s1", "--db", db]) == 0
        assert read_claim_marker(tmp_path)["task_id"] == "bead-1"
        assert read_claim_wards(tmp_path) is None

    def test_stamped_bead_inherits_track_into_marker(self, monkeypatch, tmp_path):
        track = for_gate_class("G1", negative_spec=("single-seed as general",))
        db = self._env(monkeypatch, tmp_path, track)
        assert cli.main(["claim", "claim", "bead-2", "--agent", "s1", "--db", db]) == 0
        wards = read_claim_wards(tmp_path)
        assert wards is not None
        assert WardSet.from_dict(wards) == track

    def test_widening_declaration_is_refused_before_claim(self, monkeypatch, tmp_path, capsys):
        track = for_gate_class("G2", permissions=("read",))
        db = self._env(monkeypatch, tmp_path, track)
        declared = json.dumps({"max_turns": 50, "max_depth": 0, "permissions": ["read", "vcs"]})
        rc = cli.main(["claim", "claim", "bead-3", "--agent", "s1", "--db", db, "--wards", declared])
        assert rc == 3
        assert "REFUSED" in capsys.readouterr().out
        # The claim was never taken, so a second session can still claim it.
        assert cli.main(["claim", "claim", "bead-3", "--agent", "s2", "--db", db]) == 0

    def test_declaration_file_is_read(self, monkeypatch, tmp_path):
        track = for_gate_class("G0", permissions=("read", "test"))
        db = self._env(monkeypatch, tmp_path, track)
        (tmp_path / ".agentgit").mkdir()
        (tmp_path / ".agentgit" / "wards.json").write_text(
            json.dumps({"max_turns": 2, "max_depth": 0, "permissions": ["read"]})
        )
        assert cli.main(["claim", "claim", "bead-4", "--agent", "s1", "--db", db]) == 0
        assert read_claim_wards(tmp_path)["max_turns"] == 2

    def test_no_wards_flag_skips_gate(self, monkeypatch, tmp_path):
        track = for_gate_class("G2")
        db = self._env(monkeypatch, tmp_path, track)
        declared = json.dumps({"max_turns": 50, "max_depth": 0})
        assert cli.main(["claim", "claim", "bead-5", "--agent", "s1", "--db", db, "--wards", declared, "--no-wards"]) == 0

    def test_wards_stamp_and_check(self, monkeypatch, tmp_path, capsys):
        self._env(monkeypatch, tmp_path, None)
        rc = cli.main(["wards", "stamp", "--gate-class", "G0", "--negative", "mock without cordon", "--permission", "read"])
        assert rc == 0
        line = capsys.readouterr().out.strip().splitlines()[-1]
        parsed = parse_stamp(line)
        assert parsed is not None and parsed.negative_spec == ("mock without cordon",)
        monkeypatch.setattr(cli, "load_bead_wards", lambda task_id, **kw: parsed)
        assert cli.main(["wards", "check", "bead-9", "--wards", json.dumps({"max_turns": 1, "max_depth": 0, "permissions": ["read"]})]) == 0
        assert cli.main(["wards", "check", "bead-9", "--wards", json.dumps({"max_turns": 1, "max_depth": 0, "permissions": ["vcs"]})]) == 3
        assert cli.main(["wards", "show", "bead-9"]) == 0
