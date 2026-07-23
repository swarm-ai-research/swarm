#!/usr/bin/env python3
"""Visualize recursive subagent spawning in Claude Code.

Demonstrates that the Task tool only supports depth-1 fan-out:
the root agent can spawn N children in parallel, but children
cannot spawn grandchildren (they lack the Task spawning tool).

Generates:
  1. ASCII tree of actual vs attempted recursive structure
  2. Matplotlib figure showing both topologies
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Tree data model
# ---------------------------------------------------------------------------

@dataclass
class AgentNode:
    id: str
    depth: int
    parent: str | None = None
    model: str = "unknown"
    status: str = "ok"  # ok | refused | timeout
    message: str = ""
    children: list[AgentNode] = field(default_factory=list)
    agent_id: str | None = None  # Claude Code agent ID

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "depth": self.depth,
            "parent": self.parent,
            "model": self.model,
            "status": self.status,
            "message": self.message,
            "agent_id": self.agent_id,
            "children": [c.to_dict() for c in self.children],
        }


# ---------------------------------------------------------------------------
# Actual results from the spawning experiment
# ---------------------------------------------------------------------------

def build_actual_tree() -> AgentNode:
    """Build the tree that actually worked (depth-1 fan-out only)."""
    root = AgentNode(id="ROOT", depth=0, model="opus-4.6", status="ok",
                     message="Root agent (orchestrator)")

    leaves = [
        AgentNode(id="alpha", depth=1, parent="ROOT", model="haiku-4.5",
                  status="ok", message="LEAF|alpha|depth=1|parent=ROOT",
                  agent_id="a83d9a1"),
        AgentNode(id="beta", depth=1, parent="ROOT", model="haiku-4.5",
                  status="ok", message="LEAF|beta|depth=1|parent=ROOT",
                  agent_id="a826c18"),
        AgentNode(id="gamma", depth=1, parent="ROOT", model="haiku-4.5",
                  status="ok", message="LEAF|gamma|depth=1|parent=ROOT",
                  agent_id="a9f9b68"),
        AgentNode(id="delta", depth=1, parent="ROOT", model="haiku-4.5",
                  status="ok", message="LEAF|delta|depth=1|parent=ROOT",
                  agent_id="a659800"),
        AgentNode(id="epsilon", depth=1, parent="ROOT", model="haiku-4.5",
                  status="ok", message="LEAF|epsilon|depth=1|parent=ROOT",
                  agent_id="a1fe14b"),
        AgentNode(id="zeta", depth=1, parent="ROOT", model="haiku-4.5",
                  status="ok", message="LEAF|zeta|depth=1|parent=ROOT",
                  agent_id="a645f50"),
    ]
    root.children = leaves
    return root


def build_attempted_tree() -> AgentNode:
    """Build the tree we ATTEMPTED (depth-3 recursive, all refused at depth 1)."""
    root = AgentNode(id="ROOT", depth=0, model="opus-4.6", status="ok")

    for branch_id in ["A", "B", "C"]:
        branch = AgentNode(
            id=branch_id, depth=1, parent="ROOT", model="sonnet-4.5",
            status="refused",
            message="Refused: 'I don't have access to a Task tool that spawns child agents'",
        )
        # The children that WOULD have been spawned
        for i in range(1, 3):
            child = AgentNode(
                id=f"{branch_id}{i}", depth=2, parent=branch_id,
                model="haiku-4.5", status="never_spawned",
            )
            for suffix in ["a", "b"]:
                leaf = AgentNode(
                    id=f"{branch_id}{i}{suffix}", depth=3,
                    parent=f"{branch_id}{i}", model="haiku-4.5",
                    status="never_spawned",
                )
                child.children.append(leaf)
            branch.children.append(child)
        root.children.append(branch)

    return root


# ---------------------------------------------------------------------------
# ASCII tree renderer
# ---------------------------------------------------------------------------

STATUS_ICONS = {
    "ok": "\033[92m●\033[0m",        # green
    "refused": "\033[91m✗\033[0m",    # red
    "never_spawned": "\033[90m○\033[0m",  # gray
}

def render_ascii(node: AgentNode, prefix: str = "", is_last: bool = True, lines: list[str] | None = None) -> list[str]:
    if lines is None:
        lines = []

    connector = "└── " if is_last else "├── "
    icon = STATUS_ICONS.get(node.status, "?")

    if node.depth == 0:
        lines.append(f"{icon} {node.id} ({node.model})")
    else:
        model_tag = f" [{node.model}]" if node.model != "unknown" else ""
        status_tag = " \033[91m(REFUSED)\033[0m" if node.status == "refused" else ""
        status_tag = " \033[90m(never spawned)\033[0m" if node.status == "never_spawned" else status_tag
        lines.append(f"{prefix}{connector}{icon} {node.id}{model_tag}{status_tag}")

    extension = "    " if is_last else "│   "
    child_prefix = prefix + extension if node.depth > 0 else prefix

    for i, child in enumerate(node.children):
        render_ascii(child, child_prefix, i == len(node.children) - 1, lines)

    return lines


# ---------------------------------------------------------------------------
# Matplotlib visualization
# ---------------------------------------------------------------------------

def _layout_tree(node: AgentNode, x: float = 0, y: float = 0,
                 x_spacing: float = 1.5, y_spacing: float = 1.2,
                 positions: dict | None = None) -> dict:
    """Assign (x, y) coordinates to each node."""
    if positions is None:
        positions = {}

    positions[node.id] = (x, -y)

    if not node.children:
        return positions

    total_width = (len(node.children) - 1) * x_spacing
    start_x = x - total_width / 2

    for i, child in enumerate(node.children):
        child_x = start_x + i * x_spacing
        # Narrow spacing for deeper nodes
        child_spacing = x_spacing / max(len(child.children), 2)
        _layout_tree(child, child_x, y + y_spacing, child_spacing, y_spacing, positions)

    return positions


def plot_trees(actual: AgentNode, attempted: AgentNode, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot generation")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle("Recursive Subagent Spawning in Claude Code", fontsize=16, fontweight="bold")

    color_map = {
        "ok": "#4CAF50",
        "refused": "#F44336",
        "never_spawned": "#9E9E9E",
    }
    model_markers = {
        "opus-4.6": "s",      # square
        "sonnet-4.5": "D",    # diamond
        "haiku-4.5": "o",     # circle
    }

    def draw_tree(ax, root, title, x_spacing):
        ax.set_title(title, fontsize=13, pad=15)
        positions = _layout_tree(root, x_spacing=x_spacing)

        # Draw edges
        def draw_edges(node):
            if node.id in positions:
                px, py = positions[node.id]
                for child in node.children:
                    if child.id in positions:
                        cx, cy = positions[child.id]
                        edge_color = "#BDBDBD" if child.status == "never_spawned" else "#616161"
                        linestyle = "--" if child.status == "never_spawned" else "-"
                        ax.plot([px, cx], [py, cy], color=edge_color,
                                linewidth=1.5, linestyle=linestyle, zorder=1)
                    draw_edges(child)
        draw_edges(root)

        # Draw nodes
        def draw_nodes(node):
            if node.id in positions:
                x, y = positions[node.id]
                color = color_map.get(node.status, "#9E9E9E")
                marker = model_markers.get(node.model, "o")
                size = 200 if node.depth == 0 else 120
                ax.scatter(x, y, c=color, marker=marker, s=size,
                           zorder=3, edgecolors="white", linewidths=1.5)
                offset_y = 0.15
                fontsize = 9 if node.depth > 0 else 11
                ax.annotate(node.id, (x, y - offset_y), ha="center", va="top",
                            fontsize=fontsize, fontweight="bold" if node.depth == 0 else "normal")
            for child in node.children:
                draw_nodes(child)
        draw_nodes(root)

        ax.set_aspect("equal")
        ax.axis("off")

    draw_tree(ax1, actual, "Actual: Depth-1 Fan-Out (6 leaves)", x_spacing=1.8)
    draw_tree(ax2, attempted, "Attempted: Depth-3 Recursive (refused at depth 1)", x_spacing=2.5)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor="#4CAF50", label="Success (responded)"),
        mpatches.Patch(facecolor="#F44336", label="Refused (no Task tool)"),
        mpatches.Patch(facecolor="#9E9E9E", label="Never spawned"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                    markersize=10, label="Opus 4.6 (root)"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                    markersize=10, label="Sonnet 4.5 (mid)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                    markersize=10, label="Haiku 4.5 (leaf)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=10, frameon=True, fancybox=True)

    plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\n  Plot saved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    actual = build_actual_tree()
    attempted = build_attempted_tree()

    # ASCII trees
    print("=" * 70)
    print("ACTUAL TREE (what worked: depth-1 fan-out)")
    print("=" * 70)
    for line in render_ascii(actual):
        print(line)

    print()
    print("=" * 70)
    print("ATTEMPTED TREE (depth-3 recursive — refused at depth 1)")
    print("=" * 70)
    for line in render_ascii(attempted):
        print(line)

    print()
    print("=" * 70)
    print("FINDINGS")
    print("=" * 70)
    print("""
  Architecture constraint discovered:

    The Task tool (subagent spawner) is ONLY available to the
    root orchestrator agent. Spawned subagents receive:
      - TaskCreate, TaskUpdate, TaskList, TaskGet (task tracking)
      - Read, Write, Edit, Glob, Grep, Bash, etc.
      - BUT NOT the Task spawning tool itself

    This means:
      Max recursion depth: 1 (root → leaves)
      Max fan-out: unlimited (tested 6 parallel)
      Topology: star/fan, not tree

    Implication for multi-agent systems:
      Claude Code's agent architecture is a single-level
      orchestrator pattern, not a recursive hierarchy.
      The root agent is the ONLY coordinator.
""")

    # Stats
    actual_nodes = 1 + len(actual.children)
    attempted_nodes = 1 + 3 + 6 + 12  # root + 3 branches + 6 mid + 12 leaves
    print(f"  Actual agents spawned:    {actual_nodes} (1 root + {len(actual.children)} leaves)")
    print(f"  Attempted agents:         {attempted_nodes} (would need recursive Task tool)")
    print(f"  Success rate depth-1:     {len(actual.children)}/{len(actual.children)} (100%)")
    print("  Success rate depth-2+:    0/3 (0% — tool not available)")
    print()

    # Save JSON
    out_dir = Path(__file__).resolve().parent.parent / "runs" / "agent_tree_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    tree_data = {
        "experiment": "recursive_subagent_spawning",
        "finding": "Task spawning tool only available to root orchestrator",
        "max_depth_achieved": 1,
        "max_fanout_tested": 6,
        "actual_tree": actual.to_dict(),
        "attempted_tree": attempted.to_dict(),
    }
    json_path = out_dir / "agent_tree.json"
    json_path.write_text(json.dumps(tree_data, indent=2))
    print(f"  JSON saved: {json_path}")

    # Plot
    plot_trees(actual, attempted, out_dir / "agent_tree.png")


if __name__ == "__main__":
    main()
