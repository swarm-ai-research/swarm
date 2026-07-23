---
title: Knowledge Graph
description: "Interactive map of the SWARM knowledge base spanning docs, scenarios, slash commands, agents, roles, and (locally) artifacts research. Traverse by links, filter by kind, search, and find the shortest link path between any two concepts."
hide:
  - toc
---

# Knowledge Graph

Every page in this repo is a node; every link, `[[wikilink]]`, `/slash-command`
mention, or referenced file path is an edge. The graph spans docs, scenarios,
slash commands, agents, role memory, and — when the `swarm-artifacts` repo is
checked out alongside this one — its papers, research notes, and study writeups.

**Drag** to pan, **scroll** to zoom, **hover** a node to see its description in
the side panel, and **click** to open it (or focus it, in speedrun mode).
Dashed nodes are orphans — nothing links to them yet. **Node size reflects
PageRank**: bigger pages are more central to the corpus. Hit **Key concepts**
to spotlight the load-bearing pages, or **Show orphans** for the disconnected
ones.

<div id="kb-stats" class="kb-stats"></div>

<div class="kb-controls" markdown="0">
  <div class="kb-row">
    <input id="kb-search" type="search" placeholder="Search nodes…" autocomplete="off" />
    <label class="kb-radio"><input type="radio" name="kb-mode" id="kb-mode-browse" checked /> Browse</label>
    <label class="kb-radio" title="Click stays on the graph; the side panel shows outgoing links to click next."><input type="radio" name="kb-mode" id="kb-mode-speedrun" /> Speedrun</label>
    <button id="kb-show-orphans" class="kb-mini-btn">Show orphans</button>
    <button id="kb-show-key" class="kb-mini-btn" title="Spotlight the most central pages by PageRank — the corpus spine and the pages whose edits ripple widest.">Key concepts</button>
    <button id="kb-toggle-semantic" class="kb-mini-btn on" title="Dashed violet edges = TF-IDF similarity suggestions, no explicit link in the corpus.">Suggestions: on</button>
  </div>
  <div class="kb-pathfinder">
    <span class="kb-pf-label">Shortest path:</span>
    <select id="kb-from" aria-label="From"></select>
    <span class="kb-arrow">→</span>
    <select id="kb-to" aria-label="To"></select>
    <button id="kb-find-path">Find</button>
  </div>
  <div id="kb-path-result" class="kb-path-result"></div>
  <div id="kb-trail" class="kb-trail"></div>
  <div id="kb-legend" class="kb-legend"></div>
</div>

<div class="kb-stage">
  <div id="kb-graph" class="kb-graph-canvas"></div>
  <aside id="kb-info" class="kb-info"></aside>
</div>

<script src="https://cdn.jsdelivr.net/npm/cytoscape@3.30.2/dist/cytoscape.min.js"></script>
<script src="../javascripts/kb_graph.js"></script>

> The graph is regenerated on every docs build from the live corpus across
> docs, scenarios, `.claude/commands`, `.claude/agents`, `agents/`, and
> (locally) `swarm-artifacts`. To rebuild it and see what's weakly linked,
> run `python scripts/build_kb_graph.py`.
