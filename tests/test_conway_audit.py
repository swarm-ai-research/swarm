"""Tests for scripts/conway_audit.py (beads-0tda).

Conway's homomorphism as a check: an import edge between modules built
under beads with no path between them in the dispatch graph is an
unnegotiated coupling. These tests pin the three classifications and the
parsing that feeds them.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conway_audit import (  # noqa: E402
    build_dispatch_graph,
    classify,
    imports_of,
    module_name,
    parent_id,
    parse_bead_refs,
    resolve_module,
    short_id,
)

KNOWN = {s: f"x-{s}" for s in ("vw8g", "illq", "illq.2", "naa8", "b61g")}


class TestParsing:
    def test_short_and_parent(self):
        assert short_id("x-illq.2") == "illq.2"
        assert parent_id("x-illq.2") == "x-illq"
        assert parent_id("x-illq") is None

    def test_refs_only_known_tokens(self):
        refs = parse_bead_refs("fix(blog): thing (vw8g, deps) illq.2", KNOWN)
        assert refs == {"x-vw8g", "x-illq.2"}

    def test_refs_word_boundary(self):
        assert parse_bead_refs("naa8x and xnaa8", KNOWN) == set()

    def test_module_name(self):
        assert module_name("swarm/core/proxy.py", "swarm") == "swarm.core.proxy"
        assert module_name("swarm/core/__init__.py", "swarm") == "swarm.core"
        assert module_name("tests/test_x.py", "swarm") is None
        assert module_name("swarm/data.yaml", "swarm") is None


class TestImports:
    def test_absolute_and_relative(self):
        src = (
            "import swarm.core.payoff\n"
            "from swarm.models import interaction\n"
            "from . import proxy\n"
            "from ..metrics.soft_metrics import SoftMetrics\n"
            "import numpy\n"
        )
        found = imports_of(src, "swarm.core.thing", "swarm")
        assert "swarm.core.payoff" in found
        assert "swarm.models.interaction" in found
        assert "swarm.core.proxy" in found
        assert "swarm.metrics.soft_metrics" in found
        assert not any(n.startswith("numpy") for n in found)

    def test_syntax_error_is_empty(self):
        assert imports_of("def (:", "swarm.x", "swarm") == set()

    def test_resolve_longest_prefix(self):
        known = {"swarm.core", "swarm.core.payoff"}
        assert resolve_module("swarm.core.payoff.Engine", known) == "swarm.core.payoff"
        assert resolve_module("swarm.core.other", known) == "swarm.core"
        assert resolve_module("swarm.nope", known) is None


class TestClassify:
    def _uf(self, dep_edges=(), commits=()):
        return build_dispatch_graph(KNOWN.values(), dep_edges, commits)

    def test_unnegotiated_when_no_path(self):
        owners = {"swarm.a": {"x-vw8g"}, "swarm.b": {"x-naa8"}}
        rep = classify([("swarm.a", "swarm.b")], owners, set(), [], {}, self._uf())
        assert [e["from"] for e in rep["unnegotiated"]] == ["swarm.a"]

    def test_dep_edge_negotiates(self):
        owners = {"swarm.a": {"x-vw8g"}, "swarm.b": {"x-naa8"}}
        uf = self._uf(dep_edges=[("x-vw8g", "x-naa8")])
        rep = classify([("swarm.a", "swarm.b")], owners, set(), [], {}, uf)
        assert rep["unnegotiated"] == []

    def test_parent_child_negotiates(self):
        owners = {"swarm.a": {"x-illq"}, "swarm.b": {"x-illq.2"}}
        rep = classify([("swarm.a", "swarm.b")], owners, set(), [], {}, self._uf())
        assert rep["unnegotiated"] == []

    def test_same_commit_negotiates(self):
        owners = {"swarm.a": {"x-vw8g"}, "swarm.b": {"x-b61g"}}
        uf = self._uf(commits=[frozenset({"x-vw8g", "x-b61g"})])
        rep = classify([("swarm.a", "swarm.b")], owners, set(), [], {}, uf)
        assert rep["unnegotiated"] == []

    def test_unattributed_side(self):
        owners = {"swarm.a": {"x-vw8g"}}
        rep = classify(
            [("swarm.a", "swarm.b")], owners, {"swarm.b"}, [], {}, self._uf()
        )
        assert rep["unnegotiated"] == []
        assert rep["unattributed"][0]["owned_side"] == "swarm.a"

    def test_unrealised_dep_edge(self):
        owners = {"swarm.a": {"x-vw8g"}, "swarm.b": {"x-naa8"}}
        dep = [("x-vw8g", "x-naa8")]
        rep = classify([], owners, set(), dep, {}, self._uf(dep_edges=dep))
        assert rep["unrealised"][0]["bead"] == "vw8g"
        rep2 = classify(
            [("swarm.b", "swarm.a")], owners, set(), dep, {}, self._uf(dep_edges=dep)
        )
        assert rep2["unrealised"] == []
