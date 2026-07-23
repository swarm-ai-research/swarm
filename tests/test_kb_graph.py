"""Tests for the knowledge-graph builder analytics.

Covers the docs-graph analytics layer: PageRank centrality, stale-reference
detection (broken mkdocstrings directives + dead internal links), and the
overall shape of the graph json that the UI / CLI / CI ratchet consume.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = REPO_ROOT / "scripts"


def _load_builder():
    """Import scripts/build_kb_graph.py as a module (it isn't a package)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "build_kb_graph", SCRIPTS / "build_kb_graph.py"
    )
    mod = importlib.util.module_from_spec(spec)
    # Register before exec so dataclass annotation resolution (which looks the
    # class's module up in sys.modules) works under `from __future__ import
    # annotations`.
    sys.modules["build_kb_graph"] = mod
    spec.loader.exec_module(mod)
    return mod


kb = _load_builder()


class TestPageRank:
    def test_uniform_on_empty(self):
        assert kb._pagerank([], {}) == {}

    def test_sums_to_one(self):
        ids = ["a", "b", "c", "d"]
        out_by = {"a": ["b", "c"], "b": ["c"], "c": ["a"], "d": []}
        pr = kb._pagerank(ids, out_by)
        assert pytest.approx(sum(pr.values()), abs=1e-6) == 1.0
        assert all(v > 0 for v in pr.values())

    def test_hub_outranks_leaf(self):
        # Three pages all link to the hub; the hub links nowhere.
        ids = ["hub", "x", "y", "z"]
        out_by = {"x": ["hub"], "y": ["hub"], "z": ["hub"], "hub": []}
        pr = kb._pagerank(ids, out_by)
        assert pr["hub"] > pr["x"]
        assert pr["hub"] > pr["y"]
        assert pr["hub"] > pr["z"]

    def test_dangling_redistributes(self):
        # A single dangling node must not leak probability mass.
        ids = ["a", "b"]
        out_by = {"a": [], "b": []}
        pr = kb._pagerank(ids, out_by)
        assert pytest.approx(sum(pr.values()), abs=1e-6) == 1.0
        # Symmetric graph -> equal ranks.
        assert pytest.approx(pr["a"], abs=1e-6) == pr["b"]


class TestStaleDetection:
    def test_dead_link_flagged(self, tmp_path):
        # Doc with one dead link and one live link (to this very test file's
        # dir via a real repo file) — only the dead one should be reported.
        node = kb.Node(
            id="docs/example.md", kind="doc", title="Example",
            section="root",
            source_path="docs/example.md",
            text="See [gone](./does-not-exist-xyz.md) and "
                 "[real](../README.md).",
        )
        dead = kb._dead_links([node])
        targets = {raw for _src, raw in dead}
        assert "./does-not-exist-xyz.md" in targets
        assert "../README.md" not in targets

    def test_external_and_anchor_links_ignored(self):
        node = kb.Node(
            id="docs/x.md", kind="doc", title="X", section="root",
            source_path="docs/x.md",
            text="[ext](https://example.com/a.md) [anchor](#section) "
                 "[img](./pic.png)",
        )
        # External URL, pure anchor, and a non-.md/.yaml target are all skipped.
        assert kb._dead_links([node]) == []

    def test_broken_code_directive_flagged(self):
        good = kb.Node(
            id="api/proxy.md", kind="doc", title="Proxy", section="api",
            source_path="docs/api/proxy.md",
            text="::: swarm.core.proxy.ProxyComputer",
        )
        # _resolve_python_path falls back to the nearest existing package
        # prefix, so a directive is only "broken" when no prefix resolves at
        # all — i.e. the root module is gone, not merely a renamed symbol.
        # The mkdocstrings regex captures only the lowercase module path, so
        # use a lowercase dotted name whose root package does not exist.
        bad = kb.Node(
            id="api/ghost.md", kind="doc", title="Ghost", section="api",
            source_path="docs/api/ghost.md",
            text="::: nonexistent_top_pkg.sub_module",
        )
        _edges, broken = kb._code_pass([good, bad])
        broken_ids = {dotted for _src, dotted in broken}
        assert "nonexistent_top_pkg.sub_module" in broken_ids
        assert not any(d.startswith("swarm.core.proxy") for _s, d in broken)


class TestBuildGraphShape:
    """build_graph against the live corpus exposes the analytics fields."""

    @pytest.fixture(scope="class")
    def graph(self):
        return kb.build_graph()

    def test_nodes_have_pagerank(self, graph):
        assert graph["nodes"], "expected a non-empty corpus"
        for n in graph["nodes"]:
            assert "pagerank" in n
            assert n["pagerank"] >= 0.0

    def test_pagerank_distribution(self, graph):
        total = sum(n["pagerank"] for n in graph["nodes"])
        # Rounding to 6dp over hundreds of nodes keeps this very close to 1.
        assert pytest.approx(total, abs=1e-2) == 1.0

    def test_top_central_sorted(self, graph):
        top = graph["stats"]["top_central"]
        assert top, "expected some central pages"
        prs = [c["pagerank"] for c in top]
        assert prs == sorted(prs, reverse=True)

    def test_stale_refs_present_and_counted(self, graph):
        assert isinstance(graph["stale_refs"], list)
        assert graph["stats"]["stale_count"] == len(graph["stale_refs"])
        for r in graph["stale_refs"]:
            assert r["kind"] in {"code", "link"}
            assert r["id"] and r["source"] and r["target"]
