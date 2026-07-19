"""Collusion graph and agent interaction network renderer for SWARM.

Provides force-directed graph layouts and network visualizations for
detecting coordinated behaviour (collusion) and inspecting interaction
frequency between agents.  No external graph library (e.g. networkx) is
required -- spring layout is computed with a simple repulsion/attraction
loop.

Every public function returns ``(fig, ax)``.

Usage::

    from swarm.analysis.network import plot_collusion_network

    edges = [("a0", "a1", 0.85), ("a1", "a2", 0.4)]
    fig, ax = plot_collusion_network(
        edges,
        node_types={"a0": "honest", "a1": "deceptive", "a2": "adversarial"},
        detected_coalitions=[{"a0", "a1"}],
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from swarm.analysis.theme import (
    COLORS,
    agent_color,
    swarm_theme,
)

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

def compute_spring_layout(
    nodes: List[str],
    edges: List[Tuple[str, str, float]],
    *,
    iterations: int = 50,
    seed: Optional[int] = None,
) -> Dict[str, Tuple[float, float]]:
    """Simple force-directed (spring) layout -- no networkx dependency.

    Uses Fruchterman-Reingold-style forces:

    * **Repulsion** between every pair of nodes (``1 / distance**2``).
    * **Attraction** along edges proportional to ``weight * distance``.

    Args:
        nodes: List of node identifier strings.
        edges: List of ``(source, target, weight)`` tuples.
        iterations: Number of simulation steps (default 50).
        seed: Optional random seed for reproducibility.

    Returns:
        Dictionary mapping each node id to an ``(x, y)`` position.
    """
    rng = np.random.default_rng(seed)
    n = len(nodes)
    if n == 0:
        return {}

    idx = {node: i for i, node in enumerate(nodes)}
    pos = rng.uniform(-1.0, 1.0, size=(n, 2))

    # Ideal spring length scales with sqrt(area / n)
    k = np.sqrt(4.0 / max(n, 1))

    for it in range(iterations):
        disp = np.zeros_like(pos)
        temperature = max(0.1, 1.0 - it / iterations)

        # --- Repulsive forces between all node pairs ---
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[i] - pos[j]
                dist = float(np.linalg.norm(delta))
                if dist < 1e-6:
                    delta = rng.uniform(-0.01, 0.01, size=2)
                    dist = float(np.linalg.norm(delta))
                force = (k * k) / (dist * dist)
                direction = delta / dist
                disp[i] += direction * force
                disp[j] -= direction * force

        # --- Attractive forces along edges ---
        for src, tgt, weight in edges:
            if src not in idx or tgt not in idx:
                continue
            i, j = idx[src], idx[tgt]
            delta = pos[i] - pos[j]
            dist = float(np.linalg.norm(delta))
            if dist < 1e-6:
                continue
            w = max(abs(weight), 0.1)
            force = w * dist / k
            direction = delta / dist
            disp[i] -= direction * force
            disp[j] += direction * force

        # --- Apply displacements with cooling ---
        lengths = np.linalg.norm(disp, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-6)
        disp = disp / lengths * np.minimum(lengths, temperature)
        pos += disp

    # Normalise to [0, 1] for plotting convenience
    mins = pos.min(axis=0)
    maxs = pos.max(axis=0)
    span = maxs - mins
    span[span < 1e-9] = 1.0
    pos = (pos - mins) / span

    return {node: (float(pos[i, 0]), float(pos[i, 1])) for i, node in enumerate(nodes)}


# ---------------------------------------------------------------------------
# Convex hull helper
# ---------------------------------------------------------------------------

def _draw_convex_hull(
    ax: matplotlib.axes.Axes,
    points: np.ndarray,
    *,
    color: str = COLORS.TOXICITY,
    alpha: float = 0.12,
) -> None:
    """Draw a shaded convex hull around a set of 2-D points.

    Uses :class:`scipy.spatial.ConvexHull` when available; falls back to a
    simple angle-sorted polygon otherwise.

    Args:
        ax: Matplotlib axes to draw on.
        points: ``(N, 2)`` array of x/y coordinates.
        color: Fill colour for the hull.
        alpha: Fill opacity.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        # Not enough points for a polygon -- draw a circle around the pair
        if len(pts) == 2:
            cx, cy = pts.mean(axis=0)
            r = float(np.linalg.norm(pts[0] - pts[1])) / 2.0 + 0.04
            circle = plt.Circle((cx, cy), r, facecolor=color, alpha=alpha,
                                linewidth=1.2, linestyle="--",
                                edgecolor=color, fill=True, zorder=1)
            ax.add_patch(circle)
        return

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
    except (ImportError, Exception):
        # Fallback: sort by angle from centroid
        centroid = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centroid[1], pts[:, 0] - centroid[0])
        order = np.argsort(angles)
        hull_pts = pts[order]

    # Close the polygon and add padding
    centroid = hull_pts.mean(axis=0)
    padded = centroid + (hull_pts - centroid) * 1.15

    polygon = plt.Polygon(
        padded,
        closed=True,
        facecolor=color,
        edgecolor=color,
        alpha=alpha,
        linewidth=1.4,
        linestyle="--",
        zorder=1,
    )
    ax.add_patch(polygon)


def _unique_nodes_from_edges(edges: List[Tuple[str, str, float]]) -> List[str]:
    """Extract unique nodes from edge list, preserving order.

    Args:
        edges: List of (source, target, weight) tuples.

    Returns:
        Ordered list of unique node identifiers.
    """
    seen: dict[str, None] = {}
    for src, tgt, _ in edges:
        seen.setdefault(src, None)
        seen.setdefault(tgt, None)
    return list(seen)


# ---------------------------------------------------------------------------
# Collusion network
# ---------------------------------------------------------------------------

def plot_collusion_network(
    edges: List[Tuple[str, str, float]],
    *,
    node_types: Optional[Dict[str, str]] = None,
    suspicion_scores: Optional[Dict[str, float]] = None,
    detection_threshold: float = 0.7,
    detected_coalitions: Optional[List[set]] = None,
    title: str = "Collusion Network",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Full collusion-network visualisation.

    Nodes are coloured by agent type (via :func:`agent_color`).  Edges are
    drawn with thickness proportional to the pairwise correlation.  Edges
    whose correlation exceeds *detection_threshold* are highlighted in
    :pydata:`COLORS.TOXICITY` red.  Optionally, convex hulls are drawn
    around each set in *detected_coalitions*.

    Args:
        edges: ``(source, target, correlation)`` tuples.
        node_types: Mapping from node id to agent-type string for colouring.
        suspicion_scores: Mapping from node id to a ``[0, 1]`` suspicion
            value.  When provided, node border width scales with suspicion.
        detection_threshold: Correlation above which an edge is highlighted
            as suspicious (default ``0.7``).
        detected_coalitions: List of sets of node ids.  A shaded convex hull
            is drawn around each coalition.
        title: Figure title.
        mode: ``"dark"`` or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    node_types = node_types or {}
    suspicion_scores = suspicion_scores or {}

    # Collect unique nodes preserving order
    nodes = _unique_nodes_from_edges(edges)

    positions = compute_spring_layout(nodes, edges, iterations=60, seed=42)

    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(9, 8))

        # --- Detected coalition hulls ---
        if detected_coalitions:
            hull_colors = [
                COLORS.TOXICITY,
                COLORS.EVASION,
                COLORS.DECEPTIVE,
                COLORS.PLANNER,
                COLORS.ADVERSARIAL,
            ]
            for ci, coalition in enumerate(detected_coalitions):
                pts = np.array([positions[n] for n in coalition if n in positions])
                if len(pts) >= 2:
                    hc = hull_colors[ci % len(hull_colors)]
                    _draw_convex_hull(ax, pts, color=hc, alpha=0.10)

        # --- Edges ---
        max_corr = max((abs(w) for _, _, w in edges), default=1.0)
        for src, tgt, corr in edges:
            if src not in positions or tgt not in positions:
                continue
            x0, y0 = positions[src]
            x1, y1 = positions[tgt]
            normed = abs(corr) / max(max_corr, 1e-9)
            lw = 0.6 + normed * 3.5
            is_suspicious = abs(corr) >= detection_threshold
            ec = COLORS.TOXICITY if is_suspicious else COLORS.TEXT_MUTED
            ea = 0.85 if is_suspicious else 0.35 + normed * 0.35
            ax.plot(
                [x0, x1], [y0, y1],
                color=ec, linewidth=lw, alpha=ea, zorder=2,
                solid_capstyle="round",
            )

        # --- Nodes ---
        for node in nodes:
            x, y = positions[node]
            atype = node_types.get(node, "unknown")
            fc = agent_color(atype)
            susp = suspicion_scores.get(node, 0.0)
            border_width = 1.0 + susp * 3.0
            border_color = COLORS.TOXICITY if susp > 0.5 else (
                COLORS.TEXT_PRIMARY if mode == "dark" else COLORS.TEXT_DARK
            )

            circle = plt.Circle(
                (x, y), 0.028,
                facecolor=fc,
                edgecolor=border_color,
                linewidth=border_width,
                zorder=4,
            )
            ax.add_patch(circle)
            ax.text(
                x, y - 0.045,
                node,
                ha="center", va="top",
                fontsize=7, zorder=5,
            )

        # --- Legend for agent types ---
        used_types = sorted(set(node_types.values())) if node_types else []
        if used_types:
            handles = [
                mpatches.Patch(color=agent_color(t), label=t.title())
                for t in used_types
            ]
            # Threshold legend entry
            handles.append(
                mpatches.Patch(
                    facecolor="none", edgecolor=COLORS.TOXICITY,
                    linewidth=2, label=f"corr >= {detection_threshold}",
                )
            )
            ax.legend(handles=handles, loc="upper left", fontsize=8,
                      framealpha=0.7)

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13, pad=12)
        ax.axis("off")
        fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Interaction network (simpler variant)
# ---------------------------------------------------------------------------

def plot_interaction_network(
    edges: List[Tuple[str, str, float]],
    *,
    node_types: Optional[Dict[str, str]] = None,
    node_sizes: Optional[Dict[str, float]] = None,
    title: str = "Interaction Network",
    mode: str = "dark",
) -> Tuple[plt.Figure, matplotlib.axes.Axes]:
    """Simpler network showing interaction frequency between agents.

    Edge thickness is proportional to the interaction weight/count.  Nodes
    are optionally sized by a scalar (e.g. reputation).

    Args:
        edges: ``(source, target, weight)`` tuples where *weight* is an
            interaction count or frequency.
        node_types: Mapping from node id to agent-type string.
        node_sizes: Mapping from node id to a scalar controlling node
            radius (e.g. reputation).
        title: Figure title.
        mode: ``"dark"`` or ``"light"``.

    Returns:
        ``(fig, ax)`` tuple.
    """
    node_types = node_types or {}
    node_sizes = node_sizes or {}

    nodes = _unique_nodes_from_edges(edges)

    positions = compute_spring_layout(nodes, edges, iterations=50, seed=7)

    # Normalise node sizes to a reasonable radius range
    if node_sizes:
        raw = np.array([node_sizes.get(n, 1.0) for n in nodes], dtype=float)
        smin, smax = float(raw.min()), float(raw.max())
        span = smax - smin if smax > smin else 1.0
        radii = 0.018 + 0.025 * (raw - smin) / span
    else:
        radii = np.full(len(nodes), 0.025)

    with swarm_theme(mode):
        fig, ax = plt.subplots(figsize=(9, 8))

        # --- Edges ---
        max_w = max((abs(w) for _, _, w in edges), default=1.0)
        for src, tgt, weight in edges:
            if src not in positions or tgt not in positions:
                continue
            x0, y0 = positions[src]
            x1, y1 = positions[tgt]
            normed = abs(weight) / max(max_w, 1e-9)
            lw = 0.5 + normed * 4.0
            ax.plot(
                [x0, x1], [y0, y1],
                color=COLORS.WELFARE,
                linewidth=lw,
                alpha=0.25 + normed * 0.5,
                zorder=2,
                solid_capstyle="round",
            )

        # --- Nodes ---
        for i, node in enumerate(nodes):
            x, y = positions[node]
            atype = node_types.get(node, "unknown")
            fc = agent_color(atype)
            r = float(radii[i])

            border_color = (
                COLORS.TEXT_PRIMARY if mode == "dark" else COLORS.TEXT_DARK
            )
            circle = plt.Circle(
                (x, y), r,
                facecolor=fc,
                edgecolor=border_color,
                linewidth=1.0,
                zorder=4,
            )
            ax.add_patch(circle)
            ax.text(
                x, y - r - 0.015,
                node,
                ha="center", va="top",
                fontsize=7, zorder=5,
            )

        # --- Legend ---
        used_types = sorted(set(node_types.values())) if node_types else []
        if used_types:
            handles = [
                mpatches.Patch(color=agent_color(t), label=t.title())
                for t in used_types
            ]
            ax.legend(handles=handles, loc="upper left", fontsize=8,
                      framealpha=0.7)

        # Size legend (min / max) if we have varying sizes
        if node_sizes and len(set(node_sizes.values())) > 1:
            vals = sorted(node_sizes.values())
            lo, hi = vals[0], vals[-1]
            ax.text(
                0.99, 0.01,
                f"node size: {lo:.2f} -- {hi:.2f}",
                transform=ax.transAxes,
                ha="right", va="bottom",
                fontsize=7,
                alpha=0.6,
            )

        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=13, pad=12)
        ax.axis("off")
        fig.tight_layout()

    return fig, ax
