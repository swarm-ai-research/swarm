---
name: kb-query
description: Query the SWARM knowledge graph structurally — find pages, list backlinks, follow link paths, and surface related/semantic neighbors across docs, scenarios, slash commands, agents, roles, and code references. Prefer this over grep when you need *connected* answers ("what links to X", "how does X relate to Y", "what's similar to X").
version: "1.0"
metadata:
  author: swarm-research-os
  generated_from: feat/kb-knowledge-graph
allowed-tools: Bash Read
---

## EXECUTE NOW

**Query: $ARGUMENTS**

Parse the query type and run the matching subcommand. The script lives at
`scripts/kb_graph_query.py` and reads `docs/assets/kb_graph.json` (it will
rebuild the graph automatically if the JSON is missing).

| Query shape | Subcommand | Example |
|---|---|---|
| "find X", "search X" | `find <terms>` | `find soft labels` |
| "info on X", "describe X" | `info <id-or-title>` | `info concepts/soft-labels.md` |
| "what links to X", "who uses X" | `backlinks <X>` | `backlinks concepts/governance.md` |
| "what does X link to" | `outbound <X>` | `outbound cmd/ship` |
| "X is related to what" | `related <X>` | `related agent/auditor` |
| "how does X reach Y", "connect X and Y" | `path <X> <Y>` | `path concepts/soft-labels.md papers/delegation_games.md` |
| "what's dead/unused/disconnected" | `orphans [--kind K]` | `orphans --kind command` |
| "key/central pages", "where do I start", "what's load-bearing" | `central [--kind K] [-n N]` | `central -n 10` |
| "broken/stale links", "dead code refs" | `stale` | `stale` |

For free-text questions ("what is X about", "what's similar to X"), prefer
`info` followed by `related` — info gives the description, related gives the
neighborhood across all edge kinds including TF-IDF semantic suggestions.

---

## Execution

Run the relevant subcommand as a single Bash invocation and return its output
verbatim. Do not synthesize or summarize unless the user asked you to — the
caller usually wants the raw structure.

```bash
python scripts/kb_graph_query.py <subcommand> <args>
```

If the user gave a free-text query ("tell me about delegation games"):

```bash
python scripts/kb_graph_query.py info delegation_games
python scripts/kb_graph_query.py related delegation_games
```

If the query is ambiguous (multiple title matches), the script prints the
candidates and exits non-zero. Ask the user which one they meant, or pick the
clearest exact match by id.

---

## When to use this skill (and when not to)

**Use kb-query when:**
- You need to know **what links to** a doc/scenario/command/agent before
  renaming or deleting it.
- You want the **shortest conceptual path** between two ideas (introductions,
  related-work sections, onboarding paths).
- You're looking for **semantically related** pages that aren't explicitly
  linked yet (densification candidates, duplicate-content candidates).
- You want **orphan pages** for triage — by kind, by section.
- You want to navigate the corpus **structurally** instead of by full-text grep.

**Don't use it when:**
- You need the contents of a specific file you already know — just Read it.
- You want full-text occurrence counts of a phrase — use Grep.
- You need *live* state of a service or test run — the graph is a static
  snapshot of the markdown/code corpus.

---

## Edge kinds in the graph

- `link` — explicit `[text](path.md)` markdown link
- `wikilink` — `[[name]]` resolved by filename stem
- `mention` — bare repo-relative path mention (`scenarios/foo.yaml`)
- `slashcmd` — `/command-name` reference resolving to a `.claude/commands/` node
- `code` — mkdocstrings `::: swarm.x.y` directive in api docs → source file
- `semantic` — TF-IDF cosine ≥ 0.18 suggestion; **not** a real corpus link.
  `path` and `backlinks` deliberately ignore semantic edges; `related`
  includes them so you can find topical kin.

---

## See also

- `scripts/build_kb_graph.py` — regenerate the graph (run after large doc
  changes; the mkdocs hook does this automatically per build).
- `scripts/build_kb_graph.py --check` — CI gate that fails on new orphans
  not in `.kb-graph-orphans`.
- The `/graph` page in the rendered docs site — the interactive viz of the
  same data.
