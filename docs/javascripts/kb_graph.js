/* Multi-source knowledge-graph viewer for SWARM docs.
 *
 * Loads assets/kb_graph.json and renders a force-directed Cytoscape graph.
 * Nodes span multiple kinds: docs, scenarios, slash commands, agents, roles,
 * and (when checked out locally) artifacts papers/research/notes.
 *
 * Modes:
 *   - Browse mode  : clicking a node opens its page (docs) or GitHub source.
 *   - Speedrun mode: clicking a node focuses it in a side panel showing its
 *                    outgoing links — you can keep clicking to rack up hops
 *                    without ever leaving the /graph page. A visited-trail
 *                    breadcrumb tracks the run.
 *
 * Features: kind filter chips, search, BFS shortest-path pathfinder, hover
 * info panel, and an orphan highlighter.
 */
(function () {
  "use strict";

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  // Escape any corpus-derived string before concatenating it into innerHTML.
  // Node titles and descriptions come from arbitrary markdown frontmatter and
  // H1s; without this a doc whose title contains "<script>..." would execute
  // on /graph and on every page that links back to it.
  var ESC_MAP = {"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"};
  function esc(s) {
    return String(s == null ? "" : s).replace(/[&<>"']/g, function (c) { return ESC_MAP[c]; });
  }

  function baseUrl() {
    var marker = "/graph/";
    var p = window.location.pathname;
    var i = p.indexOf(marker);
    return i >= 0 ? p.slice(0, i + 1) : "/";
  }

  // Color by node kind — far more useful for a multi-source graph than
  // coloring by mkdocs section (which only meaningfully partitions docs).
  var KIND_COLORS = {
    "doc":          "#2563eb",  // blue
    "scenario":     "#059669",  // green
    "command":      "#d97706",  // amber
    "agent":        "#7c3aed",  // violet
    "role":         "#db2777",  // pink
    "paper-art":    "#dc2626",  // red
    "research-art": "#0891b2",  // cyan
    "note-art":     "#65a30d",  // lime
    "code":         "#475569",  // slate
  };
  var KIND_LABELS = {
    "doc":          "docs",
    "scenario":     "scenarios",
    "command":      "/commands",
    "agent":        "agents",
    "role":         "roles",
    "paper-art":    "papers (artifacts)",
    "research-art": "research (artifacts)",
    "note-art":     "notes (artifacts)",
    "code":         "source code",
  };

  function bfsPath(adj, start, goal) {
    if (start === goal) return [start];
    var queue = [start];
    var prev = {}; prev[start] = null;
    while (queue.length) {
      var cur = queue.shift();
      var nbrs = adj[cur] || [];
      for (var k = 0; k < nbrs.length; k++) {
        var n = nbrs[k];
        if (!(n in prev)) {
          prev[n] = cur;
          if (n === goal) {
            var path = [n]; var c = cur;
            while (c !== null) { path.unshift(c); c = prev[c]; }
            return path;
          }
          queue.push(n);
        }
      }
    }
    return null;
  }

  ready(function () {
    var container = document.getElementById("kb-graph");
    if (!container || typeof cytoscape === "undefined") return;
    var base = baseUrl();

    fetch(base + "assets/kb_graph.json")
      .then(function (r) { return r.json(); })
      .then(function (graph) { init(graph, base); })
      .catch(function (e) {
        container.innerHTML = '<p style="padding:1rem">Could not load graph data: ' + e + "</p>";
      });

    function init(graph, base) {
      var stats = document.getElementById("kb-stats");
      if (stats) {
        var byKind = graph.stats.by_kind || {};
        var parts = Object.keys(byKind).sort().map(function (k) {
          return '<span class="kb-stat-pill" style="border-color:' +
            (KIND_COLORS[k] || "#999") + '">' +
            (KIND_LABELS[k] || k) + ": " + byKind[k] + "</span>";
        });
        var staleN = graph.stats.stale_count || 0;
        stats.innerHTML = "<b>" + graph.stats.node_count + " nodes</b> · <b>" +
          graph.stats.edge_count + " edges</b> · " +
          (graph.stats.orphan_count || 0) + " orphans · " +
          staleN + " stale refs · " + parts.join(" ");
      }

      // Node size is driven by PageRank (corpus centrality) rather than raw
      // degree — a page linked from a few very central pages should look more
      // important than one linked from many leaf pages. Scale relative to the
      // max so the dynamic range is stable regardless of corpus size.
      var maxPr = 0;
      graph.nodes.forEach(function (n) { if ((n.pagerank || 0) > maxPr) maxPr = n.pagerank; });
      if (maxPr <= 0) maxPr = 1;

      // Top-N central node ids for the "Key concepts" highlighter, taken from
      // the precomputed stats (falls back to empty if an older graph json).
      var centralIds = {};
      (graph.stats.top_central || []).forEach(function (c) { centralIds[c.id] = true; });

      var adj = {};        // all edges
      var adjReal = {};    // explicit edges only (BFS uses this — semantic edges are suggestions)
      var byId = {};
      graph.nodes.forEach(function (n) { adj[n.id] = []; adjReal[n.id] = []; byId[n.id] = n; });
      graph.edges.forEach(function (e) {
        adj[e.source].push(e.target);
        if (e.kind !== "semantic") adjReal[e.source].push(e.target);
      });

      var elements = [];
      graph.nodes.forEach(function (n) {
        elements.push({ data: {
          id: n.id, label: n.title, section: n.section, kind: n.kind,
          color: KIND_COLORS[n.kind] || "#888",
          deg: (n.indegree + n.outdegree),
          pr: Math.round((n.pagerank || 0) / maxPr * 100),
          orphan: n.orphan
        }});
      });
      graph.edges.forEach(function (e) {
        elements.push({ data: { id: e.source + "->" + e.target,
          source: e.source, target: e.target, kind: e.kind }});
      });

      var cy = cytoscape({
        container: container,
        elements: elements,
        wheelSensitivity: 0.2,
        style: [
          { selector: "node", style: {
            "background-color": "data(color)",
            "label": "data(label)",
            "font-size": 5,
            "width": "mapData(pr, 0, 100, 6, 40)",
            "height": "mapData(pr, 0, 100, 6, 40)",
            "text-wrap": "wrap", "text-max-width": 60,
            "color": "#334155", "min-zoomed-font-size": 8,
            "border-width": 0
          }},
          { selector: "node[?orphan]", style: {
            "border-width": 1.5, "border-color": "#94a3b8", "border-style": "dashed"
          }},
          { selector: "edge", style: {
            "width": 0.5, "line-color": "#cbd5e1",
            "target-arrow-color": "#cbd5e1", "target-arrow-shape": "triangle",
            "arrow-scale": 0.4, "curve-style": "bezier", "opacity": 0.4
          }},
          { selector: 'edge[kind = "slashcmd"]', style: { "line-color": "#d97706", "target-arrow-color": "#d97706" }},
          { selector: 'edge[kind = "mention"]',  style: { "line-style": "dashed" }},
          { selector: 'edge[kind = "code"]',     style: { "line-color": "#475569", "target-arrow-color": "#475569", "width": 1.0 }},
          { selector: 'edge[kind = "semantic"]', style: { "line-color": "#a78bfa", "target-arrow-color": "#a78bfa", "line-style": "dashed", "opacity": 0.25, "width": 0.5 }},
          { selector: 'edge[kind = "semantic"].semantic-off', style: { "display": "none" }},
          { selector: ".faded", style: { "opacity": 0.06 }},
          { selector: ".highlight", style: { "opacity": 1 }},
          { selector: "node.focal", style: { "border-width": 4, "border-color": "#111827", "z-index": 99 }},
          { selector: "node.path", style: { "border-width": 3, "border-color": "#111827", "z-index": 99 }},
          { selector: "edge.path", style: { "width": 3, "line-color": "#111827",
            "target-arrow-color": "#111827", "opacity": 1, "z-index": 99 }},
          { selector: "node.orphan-hit", style: { "border-width": 3, "border-color": "#dc2626" }},
          { selector: "node.key-hit", style: { "border-width": 3, "border-color": "#f59e0b" }}
        ],
        // cose layout tuned for the larger multi-source graph
        layout: { name: "cose", animate: false, nodeRepulsion: 5000,
                  idealEdgeLength: 50, padding: 30, randomize: false }
      });

      // ---- modes ----
      // (no need to hold a ref to the "browse" radio — speedrun is the only
      // mode flag the click/hover handlers branch on)
      var modeSpeedrun = document.getElementById("kb-mode-speedrun");
      var trailEl = document.getElementById("kb-trail");
      var trail = JSON.parse(sessionStorage.getItem("kb-trail") || "[]");

      function isSpeedrun() { return modeSpeedrun && modeSpeedrun.checked; }
      function saveTrail() { sessionStorage.setItem("kb-trail", JSON.stringify(trail)); }
      function renderTrail() {
        if (!trailEl) return;
        if (!trail.length) {
          trailEl.innerHTML = '<span class="kb-trail-empty">No hops yet — click a node in speedrun mode.</span>';
          return;
        }
        var pieces = trail.map(function (id, i) {
          var n = byId[id]; if (!n) return "";
          return '<a href="#" data-id="' + esc(id) + '" class="kb-trail-hop">' + esc(n.title) + "</a>";
        });
        trailEl.innerHTML = '<b>Your run (' + (trail.length - 1) + ' hops):</b> ' +
          pieces.join(' <span class="kb-arrow">→</span> ') +
          ' <button id="kb-trail-reset" class="kb-mini-btn">reset</button>';
        document.getElementById("kb-trail-reset").onclick = function () {
          trail = []; saveTrail(); renderTrail();
          cy.elements().removeClass("focal");
        };
        Array.prototype.forEach.call(trailEl.querySelectorAll("a.kb-trail-hop"), function (a) {
          a.onclick = function (e) { e.preventDefault(); focus(a.getAttribute("data-id")); };
        });
      }
      renderTrail();

      // ---- info panel ----
      var info = document.getElementById("kb-info");
      function renderInfo(node, mode) {
        if (!info || !node) { if (info) info.innerHTML = ""; return; }
        var color = KIND_COLORS[node.kind] || "#888";
        var kindLabel = KIND_LABELS[node.kind] || node.kind;
        var outs = (node.out || []).map(function (id) {
          var n = byId[id]; if (!n) return "";
          var color2 = KIND_COLORS[n.kind] || "#888";
          return '<li><a href="#" data-id="' + esc(id) + '" class="kb-info-link">' +
            '<span class="kb-dot" style="background:' + color2 + '"></span>' +
            esc(n.title) + ' <span class="kb-info-kind">' +
            esc(KIND_LABELS[n.kind] || n.kind) + '</span></a></li>';
        }).join("");
        var openLink = "";
        if (node.url) openLink = '<a class="kb-info-open" href="' + esc(base + node.url) + '">Open page →</a>';
        else if (node.external_url) openLink = '<a class="kb-info-open" target="_blank" rel="noopener" href="' + esc(node.external_url) + '">Open source on GitHub →</a>';
        info.innerHTML =
          '<div class="kb-info-head">' +
          '<span class="kb-dot" style="background:' + color + '"></span>' +
          '<h3>' + esc(node.title) + '</h3>' +
          '<span class="kb-info-kind">' + esc(kindLabel) + '</span></div>' +
          (node.description ? '<p class="kb-info-desc">' + esc(node.description) + '</p>' : '') +
          '<p class="kb-info-metrics">PageRank ' +
            (node.pagerank != null ? node.pagerank.toFixed(4) : "—") +
            (centralIds[node.id] ? ' <span class="kb-key-badge">key concept</span>' : '') +
            ' · in ' + (node.indegree || 0) + ' · out ' + (node.outdegree || 0) + '</p>' +
          openLink +
          '<h4>Out (' + (node.out || []).length + ')</h4>' +
          (outs ? '<ul class="kb-info-outs">' + outs + '</ul>' : '<p class="kb-info-empty">No outgoing links.</p>');
        Array.prototype.forEach.call(info.querySelectorAll("a.kb-info-link"), function (a) {
          a.onclick = function (e) { e.preventDefault(); focus(a.getAttribute("data-id")); };
        });
      }

      function focus(id) {
        var node = byId[id]; if (!node) return;
        cy.elements().removeClass("focal").removeClass("faded").removeClass("highlight");
        var cyNode = cy.getElementById(id);
        cyNode.addClass("focal");
        cy.elements().addClass("faded");
        cyNode.closedNeighborhood().removeClass("faded").addClass("highlight");
        cy.animate({ center: { eles: cyNode }, zoom: Math.max(cy.zoom(), 1.2) }, { duration: 300 });
        renderInfo(node);
        if (!trail.length || trail[trail.length - 1] !== id) {
          trail.push(id); saveTrail(); renderTrail();
        }
      }

      // ---- click / hover behavior ----
      cy.on("tap", "node", function (evt) {
        var id = evt.target.id();
        var node = byId[id];
        if (isSpeedrun()) { focus(id); return; }
        if (node.url) window.location.href = base + node.url;
        else if (node.external_url) window.open(node.external_url, "_blank", "noopener");
      });
      cy.on("mouseover", "node", function (evt) {
        if (isSpeedrun()) return;  // speedrun keeps the focal selection sticky
        var n = evt.target;
        cy.elements().addClass("faded");
        n.closedNeighborhood().removeClass("faded").addClass("highlight");
        renderInfo(byId[n.id()]);
      });
      cy.on("mouseout", "node", function () {
        if (isSpeedrun()) return;
        cy.elements().removeClass("faded").removeClass("highlight");
      });

      // ---- kind filter chips ----
      var legend = document.getElementById("kb-legend");
      var hiddenKinds = {};
      Object.keys(KIND_COLORS).forEach(function (k) {
        if (!(graph.stats.by_kind || {})[k]) return;  // skip kinds not present
        var chip = document.createElement("button");
        chip.className = "kb-chip";
        chip.style.borderColor = KIND_COLORS[k];
        chip.innerHTML = '<span class="kb-dot" style="background:' + KIND_COLORS[k] + '"></span>' +
          (KIND_LABELS[k] || k) + ' (' + (graph.stats.by_kind[k] || 0) + ')';
        chip.onclick = function () {
          hiddenKinds[k] = !hiddenKinds[k];
          chip.classList.toggle("off", hiddenKinds[k]);
          cy.nodes('[kind = "' + k + '"]').style("display", hiddenKinds[k] ? "none" : "element");
        };
        legend.appendChild(chip);
      });

      // ---- search ----
      var search = document.getElementById("kb-search");
      search.addEventListener("input", function () {
        var q = search.value.trim().toLowerCase();
        if (!q) { cy.elements().removeClass("faded").removeClass("highlight"); return; }
        var matches = cy.nodes().filter(function (n) {
          var d = n.data();
          return d.label.toLowerCase().indexOf(q) >= 0 || d.id.toLowerCase().indexOf(q) >= 0;
        });
        cy.elements().addClass("faded");
        matches.removeClass("faded").addClass("highlight");
        if (matches.length) cy.animate({ fit: { eles: matches, padding: 80 } }, { duration: 300 });
      });

      // ---- BFS speedrun pathfinder ----
      var fromSel = document.getElementById("kb-from");
      var toSel = document.getElementById("kb-to");
      var pathOut = document.getElementById("kb-path-result");
      graph.nodes.slice().sort(function (a, b) {
        return a.title.localeCompare(b.title);
      }).forEach(function (n) {
        [fromSel, toSel].forEach(function (sel) {
          var o = document.createElement("option");
          o.value = n.id;
          o.textContent = n.title + "  (" + (KIND_LABELS[n.kind] || n.kind) + ")";
          sel.appendChild(o);
        });
      });

      document.getElementById("kb-find-path").onclick = function () {
        cy.elements().removeClass("path");
        var a = fromSel.value, b = toSel.value;
        var path = bfsPath(adjReal, a, b);
        if (!path) {
          pathOut.innerHTML = '<span class="kb-nopath">No link path from <b>' +
            esc(byId[a].title) + "</b> to <b>" + esc(byId[b].title) +
            "</b> (try the reverse, or densify links).</span>";
          return;
        }
        for (var i = 0; i < path.length; i++) {
          cy.getElementById(path[i]).addClass("path");
          if (i > 0) cy.getElementById(path[i - 1] + "->" + path[i]).addClass("path");
        }
        cy.animate({ fit: { eles: cy.elements(".path"), padding: 80 } }, { duration: 400 });
        var hops = path.map(function (id) {
          var n = byId[id];
          var href = n.url ? base + n.url : (n.external_url || "#");
          var ext = n.url ? "" : ' target="_blank" rel="noopener"';
          return '<a href="' + esc(href) + '"' + ext + '>' + esc(n.title) + "</a>";
        }).join(' <span class="kb-arrow">→</span> ');
        pathOut.innerHTML = "<b>" + (path.length - 1) + " hop" +
          (path.length === 2 ? "" : "s") + ":</b> " + hops;
      };

      // ---- semantic edge toggle ----
      var semBtn = document.getElementById("kb-toggle-semantic");
      var semOn = true;
      if (semBtn) semBtn.onclick = function () {
        semOn = !semOn;
        semBtn.classList.toggle("on", semOn);
        semBtn.textContent = "Suggestions: " + (semOn ? "on" : "off");
        cy.edges('[kind = "semantic"]').toggleClass("semantic-off", !semOn);
      };

      // ---- orphan highlighter ----
      var orphanBtn = document.getElementById("kb-show-orphans");
      var orphansOn = false;
      orphanBtn.onclick = function () {
        orphansOn = !orphansOn;
        orphanBtn.classList.toggle("on", orphansOn);
        if (keyOn) { keyOn = false; keyBtn.classList.remove("on"); keyBtn.textContent = "Key concepts"; }
        if (orphansOn) {
          cy.nodes().addClass("faded").removeClass("orphan-hit");
          cy.nodes("[?orphan]").removeClass("faded").addClass("orphan-hit");
          orphanBtn.textContent = "Show all (" + cy.nodes("[?orphan]").length + " orphans)";
        } else {
          cy.nodes().removeClass("faded").removeClass("orphan-hit");
          orphanBtn.textContent = "Show orphans";
        }
      };

      // ---- key-concept (PageRank) highlighter ----
      // Spotlights the most central pages — the corpus spine you'd want a
      // newcomer to read first, and the pages whose edits ripple widest.
      var keyBtn = document.getElementById("kb-show-key");
      var keyOn = false;
      if (keyBtn) keyBtn.onclick = function () {
        keyOn = !keyOn;
        keyBtn.classList.toggle("on", keyOn);
        if (orphansOn) { orphansOn = false; orphanBtn.classList.remove("on"); orphanBtn.textContent = "Show orphans"; }
        cy.nodes().removeClass("orphan-hit");
        if (keyOn) {
          var key = cy.nodes().filter(function (nd) { return centralIds[nd.id()]; });
          cy.nodes().addClass("faded").removeClass("key-hit");
          key.removeClass("faded").addClass("key-hit");
          keyBtn.textContent = "Show all (" + key.length + " key concepts)";
        } else {
          cy.nodes().removeClass("faded").removeClass("key-hit");
          keyBtn.textContent = "Key concepts";
        }
      };
    }
  });
})();
