"""Paper synthesis for SWARM Track A runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd

from swarm.research.platforms import Paper
from swarm.research.swarm_papers.memory import MemoryArtifact


@dataclass
class PaperContext:
    title: str
    abstract: str
    run_id: str
    task_count: int
    conditions: list[dict]
    related_work: list["RelatedWorkItem"]
    memory_items: list[MemoryArtifact]
    critique_summary: "CritiqueSummary"
    family_metrics: dict[str, dict[str, dict[str, float]]]
    bib: str = ""
    figures: list["PaperFigure"] = field(default_factory=list)
    images: dict[str, str] = field(default_factory=dict)


@dataclass
class CritiqueSummary:
    total_flags: int
    flag_rate: float
    top_reasons: list[str]


@dataclass
class RelatedWorkItem:
    title: str
    cite_key: str
    paper_id: str = ""


@dataclass
class PaperFigure:
    filename: str
    caption: str


# Abbreviations for adversarial conditions
CONDITION_ABBREV = {
    "adv_noise": "nse",
    "adv_confident": "cnf",
    "adv_strategic": "str",
    "adv_sycophant": "syc",
    "adv_coordinated": "crd",
    "adv_majority": "maj",
    "adv_memory": "mem",
}


class PaperBuilder:
    """Builds a LaTeX paper from Track A run summaries."""

    def build(self, context: PaperContext) -> Paper:
        latex = self._render_latex(context)
        return Paper(
            title=context.title,
            abstract=context.abstract,
            source=latex,
            bib=context.bib,
            images=context.images,
            authors=["SWARM Research Agents"],
            categories=["swarm", "track-a", "agentrxiv"],
        )

    def _render_latex(self, context: PaperContext) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        conditions_table = self._render_conditions_table(context.conditions)
        related_work = self._render_related(context.related_work)
        memory_section = self._render_memory(context.memory_items)
        critique_section = self._render_critique(context.critique_summary)
        family_tables = self._render_family_tables(context.family_metrics, context.conditions)
        figures_section = self._render_figures(context.figures)

        return f"""\\documentclass{{article}}
\\usepackage{{booktabs}}
\\usepackage{{graphicx}}
\\usepackage{{geometry}}
\\usepackage{{adjustbox}}
\\geometry{{margin=1in}}
\\title{{{_escape_latex(context.title)}}}
\\date{{{_escape_latex(now)}}}
\\begin{{document}}
\\maketitle
\\begin{{abstract}}
{_escape_latex(context.abstract)}
\\end{{abstract}}

\\section{{Introduction}}
We evaluate SWARM-style coordination mechanisms on Track A (verifiable reasoning),
using controlled arithmetic and word-problem tasks with deterministic checks.
Each condition corresponds to a coordination policy (divergence, critique,
reconciliation, memory). This paper summarizes one full run (ID: {_escape_latex(context.run_id)}).

\\section{{Methods}}
Tasks: {context.task_count} total, generated with fixed random seed and difficulty calibration.

\\subsection{{Conditions}}
{conditions_table}

\\section{{Results}}
Across conditions, we report accuracy (correct/total), disagreement rate when
multiple solvers are active, and reconciliation frequency when enabled.

{critique_section}

{family_tables}

{figures_section}

\\section{{Related Work (AgentRxiv)}}
\\begin{{itemize}}
{related_work}
\\end{{itemize}}

{memory_section}

\\section{{Limitations}}
We treat confidence as a reported scalar and rely on simple divergence heuristics.
Future runs should incorporate stronger validators and richer task suites.

{self._render_bibliography(context.bib)}

\\end{{document}}
"""

    def _render_conditions_table(self, conditions: Iterable[dict]) -> str:
        """Render conditions summary table using pandas."""
        rows = []
        for cond in conditions:
            rows.append({
                "Condition": cond.get("name", ""),
                "Accuracy": cond.get("accuracy", 0.0),
                "Tokens": cond.get("avg_tokens", 0.0),
                "Notes": cond.get("note", ""),
            })

        if not rows:
            return "No conditions recorded."

        df = pd.DataFrame(rows)
        df["Accuracy"] = df["Accuracy"].map(lambda x: f"{x:.3f}")

        # Hide Tokens column when all values are zero (no LLM solvers)
        has_tokens = any(r["Tokens"] > 0 for r in rows)
        if has_tokens:
            df["Tokens"] = df["Tokens"].map(lambda x: f"{x:.1f}")
            col_format = "llrl"
        else:
            df = df.drop(columns=["Tokens"])
            col_format = "lll"

        latex = df.to_latex(
            index=False,
            escape=True,
            column_format=col_format,
            position="h",
        )
        return str(latex)

    def _render_memory(self, items: list[MemoryArtifact]) -> str:
        if not items:
            return "\\section{Memory Artifacts}\nNo memory artifacts were accepted in this run."
        lines = ["\\section{Memory Artifacts}", "\\begin{itemize}"]
        for item in items:
            text = f"{item.title}: {item.summary}"
            lines.append(f"\\item {_escape_latex(text)}")
        lines.append("\\end{itemize}")
        return "\n".join(lines)

    def _render_related(self, items: Iterable[RelatedWorkItem]) -> str:
        items = list(items)
        if not items:
            return "\\item None."
        lines = []
        for item in items:
            title = _escape_latex(item.title)
            cite = f"\\cite{{{item.cite_key}}}" if item.cite_key else ""
            paper_id = (
                f" (AgentRxiv ID: {_escape_latex(item.paper_id)})"
                if item.paper_id
                else ""
            )
            lines.append(f"\\item {title} {cite}{paper_id}")
        return "\n".join(lines)

    def _render_figures(self, figures: list[PaperFigure]) -> str:
        if not figures:
            return ""
        lines = ["\\section{Figures}"]
        for fig in figures:
            lines.append("\\begin{figure}[h]")
            lines.append("\\centering")
            lines.append(f"\\includegraphics[width=0.85\\linewidth]{{{_escape_latex(fig.filename)}}}")
            lines.append(f"\\caption{{{_escape_latex(fig.caption)}}}")
            lines.append("\\end{figure}")
        return "\n".join(lines)

    def _render_critique(self, summary: CritiqueSummary) -> str:
        if summary.total_flags == 0:
            return "\\paragraph{Critique Summary} No critic flags were recorded."
        lines = [
            "\\paragraph{Critique Summary}",
            f"Critic flags: {summary.total_flags} "
            f"({summary.flag_rate:.1%} of episodes).",
        ]
        if summary.top_reasons:
            lines.append("\\begin{itemize}")
            for reason in summary.top_reasons:
                lines.append(f"\\item {_escape_latex(reason)}")
            lines.append("\\end{itemize}")
        return "\n".join(lines)

    def _render_bibliography(self, bib: str) -> str:
        if not bib.strip():
            return ""
        return "\\bibliographystyle{plain}\n\\bibliography{references}"

    def _render_family_tables(
        self, family_metrics: dict[str, dict[str, dict[str, float]]], conditions: list[dict]
    ) -> str:
        """Render per-family accuracy tables using pandas."""
        if not family_metrics:
            return ""

        condition_names = [cond.get("name", "") for cond in conditions]
        families = _order_families(family_metrics)

        # Split into baseline and adversarial conditions
        baseline_conds = [c for c in condition_names if not c.startswith("adv_")]
        adv_conds = [c for c in condition_names if c.startswith("adv_")]

        sections = []

        # Baseline accuracy table
        if baseline_conds:
            sections.append(self._render_family_df(
                title="Per-Family Accuracy (Baseline)",
                metric_key="accuracy",
                family_metrics=family_metrics,
                families=families,
                condition_names=baseline_conds,
                abbreviate=False,
            ))

        # Adversarial accuracy table
        if adv_conds:
            sections.append(self._render_family_df(
                title="Per-Family Accuracy (Adversarial)",
                metric_key="accuracy",
                family_metrics=family_metrics,
                families=families,
                condition_names=adv_conds,
                abbreviate=True,
            ))

        # Token efficiency - only if any non-zero values
        has_tokens = any(
            family_metrics.get(cond, {}).get(fam, {}).get("token_eff", 0.0) > 0
            for cond in condition_names
            for fam in families
        )
        if has_tokens:
            if baseline_conds:
                sections.append(self._render_family_df(
                    title="Per-Family Token Efficiency (Baseline)",
                    metric_key="token_eff",
                    family_metrics=family_metrics,
                    families=families,
                    condition_names=baseline_conds,
                    abbreviate=False,
                ))
            if adv_conds:
                sections.append(self._render_family_df(
                    title="Per-Family Token Efficiency (Adversarial)",
                    metric_key="token_eff",
                    family_metrics=family_metrics,
                    families=families,
                    condition_names=adv_conds,
                    abbreviate=True,
                ))

        return "\n\n".join(sections)

    def _render_family_df(
        self,
        *,
        title: str,
        metric_key: str,
        family_metrics: dict[str, dict[str, dict[str, float]]],
        families: list[str],
        condition_names: list[str],
        abbreviate: bool = False,
    ) -> str:
        """Render a family metrics table using pandas DataFrame."""
        # Build data dictionary
        data: dict[str, list[Any]] = {"Family": []}
        display_names = []

        for cond in condition_names:
            display = CONDITION_ABBREV.get(cond, cond) if abbreviate else cond
            display_names.append(display)
            data[display] = []

        for family in families:
            data["Family"].append(_family_label(family))
            for cond, display in zip(condition_names, display_names, strict=False):
                value = family_metrics.get(cond, {}).get(family, {}).get(metric_key, 0.0)
                data[display].append(value)

        df = pd.DataFrame(data)

        # Format numeric columns
        for col in display_names:
            df[col] = df[col].map(lambda x: f"{x:.2f}")

        # Generate LaTeX with adjustbox for wide tables
        latex = df.to_latex(
            index=False,
            escape=True,
            column_format="l" + "r" * len(display_names),
        )

        # Wrap in adjustbox if abbreviating (likely wide table)
        if abbreviate:
            legend = (
                "\\noindent\\textit{Legend: "
                + ", ".join(f"{v}={k.replace('adv_', '')}" for k, v in CONDITION_ABBREV.items() if k in condition_names)
                + "}"
            )
            return f"""\\subsection{{{_escape_latex(title)}}}
\\begin{{adjustbox}}{{max width=\\textwidth}}
{latex}\\end{{adjustbox}}

{legend}"""
        else:
            return f"\\subsection{{{_escape_latex(title)}}}\n{latex}"


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters."""
    replacements = {
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "\\": r"\textbackslash{}",
    }
    return "".join(replacements.get(c, c) for c in text)


def _order_families(family_metrics: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    """Order families with preferred ones first."""
    preferred = ["arithmetic", "algebra", "logic_grid", "symbolic", "word"]
    families: set[str] = set()
    for metrics in family_metrics.values():
        families.update(metrics.keys())
    ordered = [fam for fam in preferred if fam in families]
    remainder = sorted(fam for fam in families if fam not in ordered)
    return ordered + remainder


def _family_label(family: str) -> str:
    """Convert family key to display label."""
    return "logic" if family == "logic_grid" else family
