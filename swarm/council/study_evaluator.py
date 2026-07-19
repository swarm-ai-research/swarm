"""LLM Council study evaluator.

Uses the council deliberation protocol to evaluate study results
with three expert perspectives: mechanism designer, statistician,
and red-teamer.
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.agents.llm_config import LLMConfig, LLMProvider
from swarm.council.config import CouncilConfig, CouncilMemberConfig
from swarm.council.protocol import Council, CouncilResult, QueryFn

logger = logging.getLogger(__name__)

# ── Evaluator personas ──────────────────────────────────────────────

MECHANISM_DESIGNER_PROMPT = """\
You are a mechanism design expert evaluating simulation study results.
Focus on: incentive compatibility, Nash equilibria, welfare properties,
mechanism monotonicity, and whether the governance design achieves its
stated objectives. Flag any perverse incentives or unintended equilibria."""

STATISTICIAN_PROMPT = """\
You are a statistician evaluating simulation study results.
Focus on: sample sizes, effect sizes (Cohen's d), multiple comparisons
corrections (Bonferroni/Holm), confidence intervals, normality assumptions,
potential confounds, and statistical power. Flag any p-hacking risks or
overclaimed significance."""

RED_TEAMER_PROMPT = """\
You are a red-team analyst evaluating simulation study results.
Focus on: exploitable loopholes in the governance mechanism, adversarial
strategies not tested, parameter ranges that might break invariants,
gaming opportunities for strategic agents, and scenarios the study
did not consider. Suggest concrete attack vectors."""

PERSONAS: Dict[str, str] = {
    "mechanism_designer": MECHANISM_DESIGNER_PROMPT,
    "statistician": STATISTICIAN_PROMPT,
    "red_teamer": RED_TEAMER_PROMPT,
}


# ── Result types ────────────────────────────────────────────────────

@dataclass
class StudyEvaluation:
    """Result of a council study evaluation."""

    council_result: CouncilResult
    findings: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    evaluation_type: str = "sweep"
    run_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON output."""
        return {
            "evaluation_type": self.evaluation_type,
            "run_dir": self.run_dir,
            "synthesis": self.council_result.synthesis,
            "findings": self.findings,
            "concerns": self.concerns,
            "recommendations": self.recommendations,
            "responses": self.council_result.responses,
            "rankings": self.council_result.rankings,
            "aggregate_ranking": self.council_result.aggregate_ranking,
            "members_responded": self.council_result.members_responded,
            "members_total": self.council_result.members_total,
            "success": self.council_result.success,
            "error": self.council_result.error,
        }


# ── Parsing ─────────────────────────────────────────────────────────

def parse_synthesis_sections(synthesis: str) -> Dict[str, List[str]]:
    """Parse synthesis text into FINDINGS / CONCERNS / RECOMMENDATIONS sections.

    Looks for section headers (case-insensitive) and collects bullet points
    and continuation lines under each. Note: bare bullet markers with no text
    (e.g., a line containing only "-" or "*") produce empty strings in the section lists.
    """
    sections: Dict[str, List[str]] = {
        "findings": [],
        "concerns": [],
        "recommendations": [],
    }

    current_section: Optional[str] = None
    current_item: Optional[str] = None

    for line in synthesis.splitlines():
        stripped = line.strip()

        # Check for section headers
        lower = stripped.lower().rstrip(":")
        if lower in ("findings", "key findings"):
            if current_section and current_item is not None:
                sections[current_section].append(current_item)
            current_section = "findings"
            current_item = None
            continue
        elif lower in ("concerns", "key concerns"):
            if current_section and current_item is not None:
                sections[current_section].append(current_item)
            current_section = "concerns"
            current_item = None
            continue
        elif lower in ("recommendations", "key recommendations"):
            if current_section and current_item is not None:
                sections[current_section].append(current_item)
            current_section = "recommendations"
            current_item = None
            continue

        if current_section is None:
            continue

        if not stripped:
            # Blank line ends current item
            if current_item is not None:
                sections[current_section].append(current_item)
                current_item = None
            continue

        # Bullet point starts a new item
        if stripped.startswith(("-", "*", "•")) or (
            len(stripped) > 2 and stripped[0].isdigit() and stripped[1] in ".)"
        ):
            if current_item is not None:
                sections[current_section].append(current_item)
            # Strip the bullet marker
            text = stripped.lstrip("-*•").strip()
            if text and text[0].isdigit() and len(text) > 1 and text[1] in ".)":
                text = text[2:].strip()
            elif not text and stripped[0].isdigit():
                text = stripped[2:].strip()
            current_item = text
        elif current_item is not None:
            # Continuation line
            current_item += " " + stripped
        else:
            # Non-bullet text under a section header
            current_item = stripped

    # Flush last item
    if current_section and current_item is not None:
        sections[current_section].append(current_item)

    return sections


# ── Config builder ──────────────────────────────────────────────────

def default_evaluator_config(
    provider_configs: Optional[Dict[str, LLMConfig]] = None,
) -> CouncilConfig:
    """Build a CouncilConfig with the 3 evaluator roles.

    Args:
        provider_configs: Optional dict mapping role name to LLMConfig.
            If not provided, defaults to Anthropic claude-sonnet-4-20250514
            for all members.
    """
    default_llm = LLMConfig(
        provider=LLMProvider.ANTHROPIC,
        model="claude-sonnet-4-20250514",
        temperature=0.3,
        max_tokens=1024,
    )

    members = []
    for role in ("mechanism_designer", "statistician", "red_teamer"):
        llm = (provider_configs or {}).get(role, default_llm)
        members.append(CouncilMemberConfig(
            member_id=role,
            llm_config=llm,
            weight=1.5 if role == "mechanism_designer" else 1.0,
        ))

    chairman = members[0]  # mechanism_designer is chairman/synthesizer

    return CouncilConfig(
        members=members,
        chairman=chairman,
        min_members_required=2,
        timeout_per_member=60.0,
        seed=42,
    )


# ── Prompt formatting ──────────────────────────────────────────────

def _format_sweep_prompt(run_dir: Path) -> str:
    """Format a prompt from sweep results for council evaluation."""
    parts = []

    # Load summary.json
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary = json.load(f)
            parts.append("## Study Summary\n")
            parts.append(json.dumps(summary, indent=2, default=str))
        except (json.JSONDecodeError, IOError) as e:
            parts.append(f"## Study Summary\n(Error reading summary.json: {e})")
    else:
        parts.append("## Study Summary\n(summary.json not found)")

    # Load sweep_results.csv — summary stats only, not raw data
    csv_path = run_dir / "sweep_results.csv"
    if csv_path.exists():
        try:
            import csv as csv_mod

            with open(csv_path) as f:
                reader = csv_mod.DictReader(f)
                rows = list(reader)

            if rows:
                parts.append(f"\n## Sweep Results ({len(rows)} rows)\n")
                parts.append(f"Columns: {', '.join(rows[0].keys())}\n")

                # Compute summary stats for numeric columns
                numeric_cols: Dict[str, List[float]] = {}
                for row in rows:
                    for k, v in row.items():
                        try:
                            fv = float(v)
                        except (ValueError, TypeError):
                            continue
                        numeric_cols.setdefault(k, []).append(fv)

                if numeric_cols:
                    parts.append("### Column Statistics\n")
                    for col, vals in numeric_cols.items():
                        n = len(vals)
                        if n == 0:
                            continue
                        mean = sum(vals) / n
                        sorted_vals = sorted(vals)
                        median = sorted_vals[n // 2]
                        mn, mx = sorted_vals[0], sorted_vals[-1]
                        parts.append(
                            f"- {col}: mean={mean:.4f}, median={median:.4f}, "
                            f"min={mn:.4f}, max={mx:.4f}, n={n}"
                        )

                # Top rows by effect size if available
                effect_cols = [c for c in rows[0].keys() if "effect" in c.lower() or "cohen" in c.lower()]
                if effect_cols:
                    col = effect_cols[0]
                    try:
                        sorted_rows = sorted(rows, key=lambda r: abs(float(r.get(col, 0))), reverse=True)
                        parts.append(f"\n### Top 5 by |{col}|\n")
                        for row in sorted_rows[:5]:
                            parts.append(str({k: v for k, v in row.items() if v}))
                    except (ValueError, TypeError):
                        pass
        except Exception as e:
            parts.append(f"\n## Sweep Results\n(Error reading CSV: {e})")
    else:
        parts.append("\n## Sweep Results\n(sweep_results.csv not found)")

    return "\n".join(parts)


def _format_scenario_prompt(yaml_path: Path) -> str:
    """Format a scenario YAML for pre-run design review."""
    parts = ["## Scenario Configuration\n"]

    if yaml_path.exists():
        parts.append(yaml_path.read_text())
    else:
        parts.append(f"(File not found: {yaml_path})")

    parts.append("\n\nEvaluate this scenario design BEFORE running it.")
    parts.append("Focus on: parameter choices, potential issues, missing controls.")
    return "\n".join(parts)


def _format_cross_study_prompt(run_dirs: List[Path]) -> str:
    """Format multiple study summaries for cross-study comparison."""
    parts = ["## Cross-Study Comparison\n"]

    for i, run_dir in enumerate(run_dirs, 1):
        parts.append(f"### Study {i}: {run_dir.name}\n")
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            parts.append(json.dumps(summary, indent=2, default=str))
        else:
            parts.append("(summary.json not found)")
        parts.append("")

    parts.append("Compare these studies. Identify consistent findings, contradictions,")
    parts.append("and gaps in coverage across the studies.")
    return "\n".join(parts)


# ── Main evaluator class ───────────────────────────────────────────

class StudyEvaluator:
    """Evaluates study results using a multi-LLM council.

    Wraps the Council protocol with expert personas (mechanism designer,
    statistician, red-teamer) to provide structured feedback on simulation
    study results.

    Usage with real LLM calls::

        # Requires: pip install -e ".[runtime]"
        # Requires: ANTHROPIC_API_KEY (or OPENAI_API_KEY, or Ollama running)

        from swarm.council.study_evaluator import StudyEvaluator

        evaluator = StudyEvaluator()  # auto-builds LLM query functions
        evaluation = evaluator.evaluate_sweep("runs/my_sweep")

    Custom providers::

        from swarm.agents.llm_config import LLMConfig, LLMProvider
        from swarm.council.study_evaluator import StudyEvaluator, default_evaluator_config

        config = default_evaluator_config(provider_configs={
            "mechanism_designer": LLMConfig(provider=LLMProvider.OPENAI, model="gpt-4o"),
            "statistician": LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-sonnet-4-20250514"),
            "red_teamer": LLMConfig(provider=LLMProvider.OLLAMA, model="llama3"),
        })
        evaluator = StudyEvaluator(config=config)
    """

    def __init__(
        self,
        config: Optional[CouncilConfig] = None,
        query_fns: Optional[Dict[str, QueryFn]] = None,
    ):
        """Initialize the study evaluator.

        Args:
            config: Council configuration. Defaults to 3-member evaluator council.
            query_fns: Dict of member_id -> async query function. If provided,
                used directly. If None, automatically builds query functions
                from each member's LLMConfig using LLMAgent.
        """
        self.config = config or default_evaluator_config()
        self._member_agents: Dict[str, Any] = {}

        if query_fns is not None:
            self._query_fns = query_fns
        else:
            self._query_fns = self._build_query_fns()

        self.council = Council(config=self.config, query_fns=self._query_fns)

    def _build_query_fns(self) -> Dict[str, QueryFn]:
        """Build async query functions from council member LLMConfigs.

        Creates an LLMAgent for each member and wraps its _call_llm_async
        method as a QueryFn closure, following the pattern in council_agent.py.
        """
        from swarm.agents.llm_agent import LLMAgent

        query_fns: Dict[str, QueryFn] = {}

        for member_cfg in self.config.members:
            llm_config = member_cfg.llm_config

            agent = LLMAgent(
                agent_id=f"evaluator_{member_cfg.member_id}",
                llm_config=llm_config,
                name=f"evaluator_{member_cfg.member_id}",
            )
            self._member_agents[member_cfg.member_id] = agent

            def _make_query_fn(a: LLMAgent) -> QueryFn:
                async def _query(sys: str, usr: str) -> str:
                    text, _, _ = await a._call_llm_async(sys, usr)
                    return str(text)
                return _query

            query_fns[member_cfg.member_id] = _make_query_fn(agent)

        return query_fns

    def _deliberate_sync(self, system_prompt: str, user_prompt: str) -> CouncilResult:
        """Run council deliberation synchronously.

        Bridges async Council.deliberate to sync context using asyncio.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # Already in an async context — create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.council.deliberate(system_prompt, user_prompt),
                )
                return future.result()
        else:
            return asyncio.run(
                self.council.deliberate(system_prompt, user_prompt)
            )

    def _build_evaluation(
        self,
        council_result: CouncilResult,
        evaluation_type: str,
        run_dir: Optional[str] = None,
    ) -> StudyEvaluation:
        """Build a StudyEvaluation from a CouncilResult."""
        sections = parse_synthesis_sections(council_result.synthesis)
        return StudyEvaluation(
            council_result=council_result,
            findings=sections["findings"],
            concerns=sections["concerns"],
            recommendations=sections["recommendations"],
            evaluation_type=evaluation_type,
            run_dir=run_dir,
        )

    def evaluate_sweep(self, run_dir: str | Path) -> StudyEvaluation:
        """Evaluate sweep results from a run directory.

        Reads summary.json and sweep_results.csv, formats a concise prompt
        with summary stats and top results, then runs council deliberation.

        Args:
            run_dir: Path to the run directory containing sweep results.
        """
        run_dir = Path(run_dir)
        user_prompt = _format_sweep_prompt(run_dir)

        system_prompt = (
            "You are evaluating the results of a multi-agent simulation parameter sweep. "
            "Analyze the study results and provide your assessment.\n\n"
            "Structure your response with these sections:\n"
            "FINDINGS:\n- Key findings from the study\n\n"
            "CONCERNS:\n- Statistical or methodological concerns\n\n"
            "RECOMMENDATIONS:\n- Suggested follow-up work or improvements\n"
        )

        result = self._deliberate_sync(system_prompt, user_prompt)
        return self._build_evaluation(result, "sweep", str(run_dir))

    def evaluate_scenario(self, yaml_path: str | Path) -> StudyEvaluation:
        """Pre-run design review of a scenario YAML.

        Args:
            yaml_path: Path to the scenario YAML file.
        """
        yaml_path = Path(yaml_path)
        user_prompt = _format_scenario_prompt(yaml_path)

        system_prompt = (
            "You are reviewing a simulation scenario design BEFORE it runs. "
            "Evaluate the scenario configuration for potential issues.\n\n"
            "Structure your response with these sections:\n"
            "FINDINGS:\n- Notable design choices\n\n"
            "CONCERNS:\n- Potential problems or missing controls\n\n"
            "RECOMMENDATIONS:\n- Suggested improvements before running\n"
        )

        result = self._deliberate_sync(system_prompt, user_prompt)
        return self._build_evaluation(result, "scenario", str(yaml_path))

    def evaluate_cross_study(self, run_dirs: List[str | Path]) -> StudyEvaluation:
        """Cross-study comparison from multiple run directories.

        Args:
            run_dirs: List of paths to run directories with summary.json files.
        """
        paths = [Path(d) for d in run_dirs]
        user_prompt = _format_cross_study_prompt(paths)

        system_prompt = (
            "You are comparing results across multiple simulation studies. "
            "Identify patterns, contradictions, and gaps.\n\n"
            "Structure your response with these sections:\n"
            "FINDINGS:\n- Consistent findings across studies\n\n"
            "CONCERNS:\n- Contradictions or methodological issues\n\n"
            "RECOMMENDATIONS:\n- Suggested follow-up studies\n"
        )

        result = self._deliberate_sync(system_prompt, user_prompt)
        return self._build_evaluation(result, "cross_study")


# ── Output ──────────────────────────────────────────────────────────

def save_evaluation(evaluation: StudyEvaluation, path: str | Path) -> Path:
    """Write full deliberation trace to JSON.

    Args:
        evaluation: The study evaluation to save.
        path: Output file path.

    Returns:
        The path written to.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(evaluation.to_dict(), f, indent=2, default=str)
    logger.info(f"Saved evaluation to {path}")
    return path
