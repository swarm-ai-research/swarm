#!/usr/bin/env python3
"""Conway audit: does the artifact's coupling graph match the dispatch graph?

Conway (1968) is causal, not a slogan: every interface between two modules
implies a negotiation between the two groups that built them, so the system
graph is a homomorphic image of the organisation's communication graph.
For a swarm the "organisation" is the beads graph plus whatever agents
happened to share a commit. This tool compares the two
(distributional-agi-safety-0tda; docs/research/classic-essays-swarm-lessons.md):

  artifact graph   import edges between package modules touched in a window
  dispatch graph   beads dep edges (any type) + parent/child by id
                   + co-occurrence in one commit message

and reports three lists:

  UNNEGOTIATED  module A imports module B, A and B were built under beads
                that have no path between them in the dispatch graph.
                An agent invented a coupling nobody negotiated. Should be
                rare; every hit is a missing edge or a missing owner.
  UNATTRIBUTED  an import edge where one side was changed only by commits
                that name no bead. The coupling exists, the dispatch graph
                cannot see who made it.
  UNREALISED    a beads dep edge whose two beads both touched package
                code, with no import between their file sets in either
                direction. Informational: the negotiation happened, the
                artifact shows no trace of it (may be fine, e.g. docs).

Usage:
    python scripts/conway_audit.py --since 2026-07-18 --until 2026-07-20
    python scripts/conway_audit.py --since 2026-07-18 --json

Bead references are taken from commit subjects: any word-boundary token
that matches a known bead short id (``vw8g``, ``illq.2``). Unknown tokens
are ignored so ``(blog)`` and ``(deps)`` never count. A commit touching
more than ``--sweep-files`` files is a sweep (pg-cleanup passes, bulk
bead closes): its bead refs still count as negotiated with each other,
but it does not make those beads the owners of every file it brushed.
Import edges are read from the working tree at HEAD, not per commit.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

_TOKEN = re.compile(r"(?<![A-Za-z0-9])([a-z0-9]{3,5}(?:\.[0-9]+)?)(?![A-Za-z0-9])")


def short_id(bead_id: str) -> str:
    """``distributional-agi-safety-illq.2`` -> ``illq.2``."""
    return bead_id.rsplit("-", 1)[-1]


def parent_id(bead_id: str) -> Optional[str]:
    """``x-illq.2`` -> ``x-illq``; top-level beads have no parent."""
    head, sep, tail = bead_id.rpartition(".")
    if sep and tail.isdigit():
        return head
    return None


def parse_bead_refs(subject: str, known: Dict[str, str]) -> Set[str]:
    """Full ids of every known bead whose short id appears in ``subject``."""
    return {known[m] for m in _TOKEN.findall(subject) if m in known}


def module_name(path: str, package: str) -> Optional[str]:
    """``swarm/core/proxy.py`` -> ``swarm.core.proxy``; non-package -> None."""
    if not path.startswith(package + "/") or not path.endswith(".py"):
        return None
    parts = path[: -len(".py")].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def imports_of(source: str, this_module: str, package: str) -> Set[str]:
    """Dotted names imported by ``source`` that live inside ``package``.

    Relative imports resolve against ``this_module``. ``from a.b import c``
    yields ``a.b.c`` and ``a.b``; the caller collapses to known modules.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    found: Set[str] = set()
    pkg_parts = this_module.split(".")
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == package or alias.name.startswith(package + "."):
                    found.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg_parts[: len(pkg_parts) - node.level]
                if node.module:
                    base = base + node.module.split(".")
                stem = ".".join(base)
            else:
                stem = node.module or ""
            if not (stem == package or stem.startswith(package + ".")):
                continue
            found.add(stem)
            for alias in node.names:
                found.add(f"{stem}.{alias.name}")
    return found


def resolve_module(name: str, known_modules: Set[str]) -> Optional[str]:
    """Longest known module that is a prefix of ``name``."""
    parts = name.split(".")
    for cut in range(len(parts), 0, -1):
        candidate = ".".join(parts[:cut])
        if candidate in known_modules:
            return candidate
    return None


class UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb

    def connected(self, a: str, b: str) -> bool:
        return self.find(a) == self.find(b)


def build_dispatch_graph(
    beads: Iterable[str],
    dep_edges: Iterable[Tuple[str, str]],
    commit_refs: Iterable[FrozenSet[str]],
) -> UnionFind:
    """Union-find over beads: dep edges, parent/child, same-commit."""
    uf = UnionFind()
    for b in beads:
        uf.find(b)
        p = parent_id(b)
        if p:
            uf.union(b, p)
    for a, b in dep_edges:
        uf.union(a, b)
    for refs in commit_refs:
        refs = list(refs)
        for other in refs[1:]:
            uf.union(refs[0], other)
    return uf


def classify(
    import_edges: Iterable[Tuple[str, str]],
    owners: Dict[str, Set[str]],
    unattributed_modules: Set[str],
    dep_edges: Iterable[Tuple[str, str]],
    module_files: Dict[str, str],
    uf: UnionFind,
) -> Dict[str, List[dict]]:
    """Split the artifact/dispatch comparison into the three report lists."""
    unnegotiated: List[dict] = []
    unattributed: List[dict] = []
    edges = list(import_edges)
    for src, dst in edges:
        so, do = owners.get(src, set()), owners.get(dst, set())
        if so and do:
            if so & do:
                continue
            if any(uf.connected(a, b) for a in so for b in do):
                continue
            unnegotiated.append(
                {"from": src, "to": dst, "from_beads": sorted(map(short_id, so)),
                 "to_beads": sorted(map(short_id, do))}
            )
        elif (so or do) and (src in unattributed_modules or dst in unattributed_modules):
            unattributed.append(
                {"from": src, "to": dst,
                 "owned_side": src if so else dst,
                 "beads": sorted(map(short_id, so or do))}
            )
    bead_modules: Dict[str, Set[str]] = defaultdict(set)
    for mod, bs in owners.items():
        for b in bs:
            bead_modules[b].add(mod)
    edge_set = set(edges)
    unrealised: List[dict] = []
    for a, b in dep_edges:
        ma, mb = bead_modules.get(a), bead_modules.get(b)
        if not ma or not mb or ma & mb:
            continue
        touching = any((x, y) in edge_set or (y, x) in edge_set for x in ma for y in mb)
        if not touching:
            unrealised.append(
                {"bead": short_id(a), "depends_on": short_id(b),
                 "bead_modules": sorted(ma), "dep_modules": sorted(mb)}
            )
    return {"unnegotiated": unnegotiated, "unattributed": unattributed,
            "unrealised": unrealised}


# ---------------------------------------------------------------- collectors


def _run(cmd: Sequence[str], cwd: Path) -> str:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True).stdout


def load_beads(root: Path) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """(short id -> full id, dep edges) from bd, else from .beads/issues.jsonl."""
    try:
        issues = json.loads(_run(["bd", "list", "--json", "--status", "all", "--limit", "0"], root))
        ids = [i["id"] for i in issues]
        edges: List[Tuple[str, str]] = []
        for i in range(0, len(ids), 40):
            chunk = ids[i : i + 40]
            out = _run(["bd", "dep", "list", "--json", *chunk], root)
            for e in json.loads(out) or []:
                if e.get("issue_id") and e.get("depends_on_id"):
                    edges.append((e["issue_id"], e["depends_on_id"]))
        return {short_id(i): i for i in ids}, edges
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        pass
    jsonl = root / ".beads" / "issues.jsonl"
    known: Dict[str, str] = {}
    edges = []
    if jsonl.is_file():
        for line in jsonl.read_text(errors="ignore").splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "id" not in d:
                continue
            known[short_id(d["id"])] = d["id"]
            for dep in d.get("dependencies") or []:
                target = dep.get("depends_on_id") or dep.get("id")
                if target:
                    edges.append((d["id"], target))
    return known, edges


def load_commits(
    root: Path, since: str, until: Optional[str]
) -> List[Tuple[str, str, List[str]]]:
    """[(hash, subject, files)] for the window, oldest first."""
    cmd = ["git", "log", "--reverse", f"--since={since}", "--format=__C__%x00%h%x00%s",
           "--name-only"]
    if until:
        cmd.insert(3, f"--until={until}")
    commits: List[Tuple[str, str, List[str]]] = []
    for line in _run(cmd, root).splitlines():
        if line.startswith("__C__\x00"):
            _, h, subj = line.split("\x00", 2)
            commits.append((h, subj, []))
        elif line.strip() and commits:
            commits[-1][2].append(line.strip())
    return commits


def package_modules(root: Path, package: str) -> Dict[str, str]:
    """module name -> repo-relative path for every .py under ``package``."""
    out: Dict[str, str] = {}
    for p in (root / package).rglob("*.py"):
        rel = p.relative_to(root).as_posix()
        mod = module_name(rel, package)
        if mod:
            out[mod] = rel
    return out


def audit(
    root: Path, since: str, until: Optional[str], package: str, sweep_files: int = 20
) -> dict:
    known, dep_edges = load_beads(root)
    commits = load_commits(root, since, until)
    modules = package_modules(root, package)
    known_modules = set(modules)

    owners: Dict[str, Set[str]] = defaultdict(set)
    unattributed: Set[str] = set()
    commit_refs: List[FrozenSet[str]] = []
    touched: Set[str] = set()
    n_with_refs = 0
    n_sweeps = 0
    for _h, subj, files in commits:
        refs = parse_bead_refs(subj, known)
        if refs:
            n_with_refs += 1
            commit_refs.append(frozenset(refs))
        is_sweep = len(files) > sweep_files
        n_sweeps += is_sweep
        for f in files:
            mod = module_name(f, package)
            if not mod or mod not in known_modules:
                continue
            touched.add(mod)
            if refs and not is_sweep:
                owners[mod] |= refs
            else:
                unattributed.add(mod)
    unattributed -= set(owners)

    import_edges: Set[Tuple[str, str]] = set()
    for mod in touched:
        src = (root / modules[mod]).read_text(errors="ignore")
        for name in imports_of(src, mod, package):
            target = resolve_module(name, known_modules)
            if target and target != mod and target in touched:
                import_edges.add((mod, target))

    uf = build_dispatch_graph(known.values(), dep_edges, commit_refs)
    report = classify(sorted(import_edges), owners, unattributed, dep_edges, modules, uf)
    report["stats"] = {
        "window": {"since": since, "until": until},
        "commits": len(commits),
        "commits_with_bead_refs": n_with_refs,
        "sweep_commits": n_sweeps,
        "modules_touched": len(touched),
        "modules_owned": len(owners),
        "modules_unattributed": len(unattributed),
        "import_edges_among_touched": len(import_edges),
        "beads_known": len(known),
        "dep_edges": len(dep_edges),
    }
    return report


def render(report: dict, limit: int = 25) -> str:
    s = report["stats"]
    lines = [
        f"Conway audit  {s['window']['since']} .. {s['window']['until'] or 'HEAD'}",
        f"  commits {s['commits']}  with bead refs {s['commits_with_bead_refs']}"
        f"  sweeps {s['sweep_commits']}",
        f"  modules touched {s['modules_touched']}  owned {s['modules_owned']}"
        f"  unattributed {s['modules_unattributed']}",
        f"  import edges among touched {s['import_edges_among_touched']}"
        f"  beads {s['beads_known']}  dep edges {s['dep_edges']}",
        "",
        f"UNNEGOTIATED ({len(report['unnegotiated'])})",
    ]
    for e in report["unnegotiated"][:limit]:
        lines.append(f"  {e['from']} {e['from_beads']} -> {e['to']} {e['to_beads']}")
    _more(lines, report["unnegotiated"], limit)
    lines.append(f"UNATTRIBUTED ({len(report['unattributed'])})")
    for e in report["unattributed"][:limit]:
        lines.append(f"  {e['from']} -> {e['to']}  owned side {e['owned_side']} {e['beads']}")
    _more(lines, report["unattributed"], limit)
    lines.append(f"UNREALISED ({len(report['unrealised'])})")
    for e in report["unrealised"][:limit]:
        lines.append(f"  {e['bead']} depends on {e['depends_on']}: no import between"
                     f" {e['bead_modules']} and {e['dep_modules']}")
    _more(lines, report["unrealised"], limit)
    return "\n".join(lines)


def _more(lines: List[str], items: list, limit: int) -> None:
    if len(items) > limit:
        lines.append(f"  ... {len(items) - limit} more (use --json for all)")


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--since", required=True, help="git --since date")
    ap.add_argument("--until", default=None, help="git --until date")
    ap.add_argument("--package", default="swarm")
    ap.add_argument("--root", default=".", type=Path)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--sweep-files", type=int, default=20,
                    help="commits touching more files than this are sweeps")
    args = ap.parse_args(argv)
    report = audit(args.root.resolve(), args.since, args.until, args.package,
                   args.sweep_files)
    print(json.dumps(report, indent=1) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
