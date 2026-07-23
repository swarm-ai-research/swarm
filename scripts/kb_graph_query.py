#!/usr/bin/env python3
"""Query the SWARM knowledge graph from the command line.

Reads docs/assets/kb_graph.json (rebuilds it if missing) and answers the
structural questions a human or LLM agent would ask of the corpus:

    find <query>                  — search nodes by title/id substring
    info <id-or-query>            — full info: kind, description, in/out counts
    backlinks <id-or-query>       — what links *to* this node (explicit only)
    outbound <id-or-query>        — what this node links *to* (explicit only)
    related <id-or-query>         — all neighbors, grouped by edge kind
                                    (includes semantic suggestions)
    path <from> <to>              — BFS shortest link path (explicit only)
    orphans [--kind K]            — pages with no inbound links
    central [--kind K] [-n N]     — most central pages by PageRank (key concepts)
    stale                         — broken code directives + dead internal links

IDs accept fuzzy lookup: if the exact id isn't found, the script falls back to
title match, then substring match. Output is text only; downstream agents that
need structured data should read `docs/assets/kb_graph.json` directly.
"""
from __future__ import annotations

import json
import sys
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GRAPH_PATH = REPO_ROOT / "docs" / "assets" / "kb_graph.json"

EXPLICIT_KINDS = {"link", "wikilink", "mention", "slashcmd", "code"}


def _load() -> dict:
    if not GRAPH_PATH.exists():
        sys.path.insert(0, str(Path(__file__).parent))
        import build_kb_graph
        g = build_kb_graph.build_graph()
        build_kb_graph.write_graph(g)
        return g
    return json.loads(GRAPH_PATH.read_text(encoding="utf-8"))


def _by_id(g: dict) -> dict[str, dict]:
    return {n["id"]: n for n in g["nodes"]}


def _resolve(g: dict, query: str) -> dict | None:
    """Exact id > exact title > unique substring match. Returns None if ambiguous."""
    by_id = _by_id(g)
    if query in by_id:
        return by_id[query]
    q = query.lower()
    title_hits = [n for n in g["nodes"] if n["title"].lower() == q]
    if len(title_hits) == 1:
        return title_hits[0]
    sub_hits = [n for n in g["nodes"]
                if q in n["title"].lower() or q in n["id"].lower()]
    if len(sub_hits) == 1:
        return sub_hits[0]
    if len(sub_hits) > 1:
        print(f"Ambiguous: {len(sub_hits)} matches for {query!r}. Try one of:",
              file=sys.stderr)
        for n in sub_hits[:8]:
            print(f"  {n['id']}  ({n['title']!r})", file=sys.stderr)
        return None
    return None


def _fmt_node(n: dict) -> str:
    return f"[{n['kind']}] {n['title']}  ({n['id']})"


def cmd_find(g: dict, args: list[str]) -> int:
    if not args:
        print("usage: find <query>", file=sys.stderr)
        return 2
    q = " ".join(args).lower()
    hits = [n for n in g["nodes"]
            if q in n["title"].lower() or q in n["id"].lower()]
    if not hits:
        print(f"No matches for {q!r}.")
        return 0
    hits.sort(key=lambda n: (n["kind"], n["title"].lower()))
    for n in hits[:25]:
        print(_fmt_node(n))
    if len(hits) > 25:
        print(f"... and {len(hits) - 25} more")
    return 0


def cmd_info(g: dict, args: list[str]) -> int:
    if not args:
        print("usage: info <id-or-query>", file=sys.stderr)
        return 2
    n = _resolve(g, " ".join(args))
    if not n:
        return 1
    print(_fmt_node(n))
    print(f"  section:  {n['section']}")
    print(f"  source:   {n['source_path']}  ({n['source_repo']})")
    if n["url"]:
        print(f"  url:      {n['url']}")
    if n["external_url"]:
        print(f"  github:   {n['external_url']}")
    if n["description"]:
        print(f"  desc:     {n['description'][:240]}")
    if n["tags"]:
        print(f"  tags:     {', '.join(n['tags'][:10])}")
    print(f"  inbound:  {n['indegree']} explicit links")
    print(f"  outbound: {n['outdegree']} explicit links")
    if n["orphan"]:
        print("  ORPHAN: nothing in the corpus links here.")
    return 0


def _edges_into(g: dict, nid: str, kinds: set[str] | None = None) -> list[dict]:
    return [e for e in g["edges"] if e["target"] == nid
            and (kinds is None or e["kind"] in kinds)]


def _edges_from(g: dict, nid: str, kinds: set[str] | None = None) -> list[dict]:
    return [e for e in g["edges"] if e["source"] == nid
            and (kinds is None or e["kind"] in kinds)]


def cmd_backlinks(g: dict, args: list[str]) -> int:
    if not args:
        print("usage: backlinks <id-or-query>", file=sys.stderr)
        return 2
    n = _resolve(g, " ".join(args))
    if not n:
        return 1
    by_id = _by_id(g)
    edges = _edges_into(g, n["id"], EXPLICIT_KINDS)
    if not edges:
        print(f"No explicit inbound links to {n['id']}.")
        return 0
    edges.sort(key=lambda e: (by_id[e["source"]]["kind"], e["kind"]))
    print(f"{len(edges)} inbound to {_fmt_node(n)}:")
    for e in edges:
        src = by_id[e["source"]]
        print(f"  ← [{e['kind']:8s}] {_fmt_node(src)}")
    return 0


def cmd_outbound(g: dict, args: list[str]) -> int:
    if not args:
        print("usage: outbound <id-or-query>", file=sys.stderr)
        return 2
    n = _resolve(g, " ".join(args))
    if not n:
        return 1
    by_id = _by_id(g)
    edges = _edges_from(g, n["id"], EXPLICIT_KINDS)
    if not edges:
        print(f"{n['id']} has no explicit outbound links.")
        return 0
    edges.sort(key=lambda e: (e["kind"], by_id[e["target"]]["title"].lower()))
    print(f"{len(edges)} outbound from {_fmt_node(n)}:")
    for e in edges:
        print(f"  → [{e['kind']:8s}] {_fmt_node(by_id[e['target']])}")
    return 0


def cmd_related(g: dict, args: list[str]) -> int:
    if not args:
        print("usage: related <id-or-query>", file=sys.stderr)
        return 2
    n = _resolve(g, " ".join(args))
    if not n:
        return 1
    by_id = _by_id(g)
    print(f"Related to {_fmt_node(n)}:")
    for label, edges in (("Outbound", _edges_from(g, n["id"])),
                         ("Inbound",  _edges_into(g, n["id"]))):
        by_kind: dict[str, list[dict]] = {}
        for e in edges:
            by_kind.setdefault(e["kind"], []).append(e)
        if not by_kind:
            continue
        print(f"\n  {label}:")
        key_name = "target" if label == "Outbound" else "source"
        for kind in sorted(by_kind):
            print(f"    [{kind}] ({len(by_kind[kind])})")
            for e in sorted(by_kind[kind],
                            key=lambda x: by_id[x[key_name]]["title"].lower())[:8]:
                other = by_id[e[key_name]]
                print(f"      {_fmt_node(other)}")
            if len(by_kind[kind]) > 8:
                print(f"      ... and {len(by_kind[kind]) - 8} more")
    return 0


def cmd_path(g: dict, args: list[str]) -> int:
    # Two positional args, each a single token. Multi-word queries must be
    # quoted by the shell so they arrive as a single argv entry; we do NOT
    # join trailing args, since that parses asymmetrically with the first.
    if len(args) != 2:
        print("usage: path <from> <to>   (each argument is a single id or "
              "quoted query)", file=sys.stderr)
        return 2
    a, b = _resolve(g, args[0]), _resolve(g, args[1])
    if not a or not b:
        return 1
    by_id = _by_id(g)
    adj: dict[str, list[str]] = {n["id"]: [] for n in g["nodes"]}
    for e in g["edges"]:
        if e["kind"] in EXPLICIT_KINDS:
            adj[e["source"]].append(e["target"])

    if a["id"] == b["id"]:
        print("(same node)")
        return 0
    q = deque([a["id"]])
    prev: dict[str, str | None] = {a["id"]: None}
    while q:
        cur = q.popleft()
        for nxt in adj[cur]:
            if nxt in prev:
                continue
            prev[nxt] = cur
            if nxt == b["id"]:
                path = [nxt]
                while prev[path[-1]] is not None:
                    path.append(prev[path[-1]])  # type: ignore[arg-type]
                path.reverse()
                print(f"{len(path) - 1} hop(s):")
                for nid in path:
                    print(f"  {_fmt_node(by_id[nid])}")
                return 0
            q.append(nxt)
    print(f"No explicit-link path from {a['id']} to {b['id']}.")
    print("(Try the reverse direction, or use `related` to find a bridge.)")
    return 1


def cmd_orphans(g: dict, args: list[str]) -> int:
    kind = None
    if "--kind" in args:
        i = args.index("--kind")
        if i + 1 < len(args):
            kind = args[i + 1]
    orphans = [n for n in g["nodes"] if n["orphan"]
               and (kind is None or n["kind"] == kind)]
    orphans.sort(key=lambda n: (n["kind"], n["section"], n["id"]))
    print(f"{len(orphans)} orphan(s)" + (f" of kind={kind}" if kind else "") + ":")
    for n in orphans[:50]:
        print(f"  {_fmt_node(n)}  (out:{n['outdegree']})")
    if len(orphans) > 50:
        print(f"... and {len(orphans) - 50} more")
    return 0


def cmd_central(g: dict, args: list[str]) -> int:
    kind = None
    n_show = 20
    if "--kind" in args:
        i = args.index("--kind")
        if i + 1 < len(args):
            kind = args[i + 1]
    if "-n" in args:
        i = args.index("-n")
        if i + 1 < len(args):
            raw = args[i + 1]
            try:
                parsed = int(raw)
            except ValueError:
                print(f"warning: -n expected an integer, got {raw!r}; using default {n_show}", file=sys.stderr)
            else:
                if parsed < 1:
                    print(f"warning: -n must be >= 1, got {parsed}; using default {n_show}", file=sys.stderr)
                else:
                    n_show = parsed
    ranked = [n for n in g["nodes"] if kind is None or n["kind"] == kind]
    # pagerank is written by build_kb_graph; fall back to indegree if absent
    # (e.g. an older cached graph json) so the command still does something.
    ranked.sort(key=lambda n: (n.get("pagerank", 0.0), n["indegree"]), reverse=True)
    label = f" of kind={kind}" if kind else ""
    print(f"Top {min(n_show, len(ranked))} central page(s){label} by PageRank:")
    for n in ranked[:n_show]:
        print(f"  {n.get('pagerank', 0.0):.4f}  in:{n['indegree']:<3d} {_fmt_node(n)}")
    return 0


def cmd_stale(g: dict, args: list[str]) -> int:
    stale = g.get("stale_refs", [])
    if not stale:
        print("No stale references — all code directives and internal links resolve. ✓")
        return 0
    by_kind: dict[str, list[dict]] = {}
    for r in stale:
        by_kind.setdefault(r["kind"], []).append(r)
    print(f"{len(stale)} stale reference(s):")
    for kind in sorted(by_kind):
        rows = by_kind[kind]
        head = "broken code directives" if kind == "code" else "dead internal links"
        print(f"\n  [{kind}] {head} ({len(rows)}):")
        for r in sorted(rows, key=lambda x: x["id"]):
            print(f"    ! {r['source']}  →  {r['target']}")
    return 0


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd, args = argv[0], argv[1:]
    g = _load()

    dispatch = {
        "find": cmd_find, "info": cmd_info,
        "backlinks": cmd_backlinks, "outbound": cmd_outbound,
        "related": cmd_related, "path": cmd_path,
        "orphans": cmd_orphans, "central": cmd_central, "stale": cmd_stale,
    }
    if cmd not in dispatch:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print(__doc__, file=sys.stderr)
        return 2
    return dispatch[cmd](g, args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
