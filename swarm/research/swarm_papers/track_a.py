"""Track A: Verifiable reasoning benchmark pipeline."""

from __future__ import annotations

import base64
import json
import logging
import math
import random
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from swarm.research.agents import Critique, PeerReview
from swarm.research.pdf_export import PDFExportError, paper_to_pdf
from swarm.research.platforms import Paper
from swarm.research.quality import QualityGates
from swarm.research.swarm_papers.agentrxiv_bridge import AgentRxivBridge
from swarm.research.swarm_papers.memory import (
    BasicAudit,
    MemoryArtifact,
    MemoryStore,
    RetrievalPolicy,
    WritePolicy,
    new_artifact_id,
    summarize_artifacts,
)
from swarm.research.swarm_papers.paper import (
    CritiqueSummary,
    PaperBuilder,
    PaperContext,
    PaperFigure,
    RelatedWorkItem,
)

logger = logging.getLogger(__name__)

# Optional dependency — may not be installed
try:
    from swarm.agents.llm_config import LLMConfig, LLMProvider
except ImportError:  # pragma: no cover - optional

    class LLMConfig:  # type: ignore[no-redef]
        """Stub when swarm.agents.llm_config is not installed."""

        def __init__(self, **kwargs: Any) -> None:
            for k, v in kwargs.items():
                setattr(self, k, v)

    class LLMProvider:  # type: ignore[no-redef]
        """Stub when swarm.agents.llm_config is not installed."""

        ANTHROPIC = "anthropic"
        OPENAI = "openai"
        OLLAMA = "ollama"


@dataclass
class ReasoningTask:
    task_id: str
    prompt: str
    answer: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Solution:
    solver_name: str
    role: str
    answer: str
    confidence: float
    raw_text: str
    tokens: int = 0


@dataclass
class EpisodeLog:
    episode_id: str
    condition: str
    task_id: str
    prompt: str
    ground_truth: str
    final_answer: str
    correct: bool
    confidence: float
    solver_outputs: list[dict]
    divergence_score: float
    reconciled: bool
    critic_flagged: bool
    critic_note: str
    adversary_flagged: bool
    adversary_selected: bool  # Was an adversary's answer selected?
    adversary_blocked: bool  # Was adversary detected and blocked?
    memory_used: list[str]
    tokens: int
    timestamp: str
    family: str

    def to_dict(self) -> dict:
        return {
            "episode_id": self.episode_id,
            "condition": self.condition,
            "task_id": self.task_id,
            "prompt": self.prompt,
            "ground_truth": self.ground_truth,
            "final_answer": self.final_answer,
            "correct": self.correct,
            "confidence": self.confidence,
            "solver_outputs": self.solver_outputs,
            "divergence_score": self.divergence_score,
            "reconciled": self.reconciled,
            "critic_flagged": self.critic_flagged,
            "critic_note": self.critic_note,
            "adversary_flagged": self.adversary_flagged,
            "adversary_selected": self.adversary_selected,
            "adversary_blocked": self.adversary_blocked,
            "memory_used": self.memory_used,
            "tokens": self.tokens,
            "timestamp": self.timestamp,
            "family": self.family,
        }


class AdversaryStrategy:
    """Adversary behavior strategies."""

    NOISE = "noise"  # Random noise with high confidence
    CONFIDENT_WRONG = "confident_wrong"  # Always wrong, max confidence
    SYCOPHANT = "sycophant"  # Agrees with first solver but subtly wrong
    STRATEGIC = "strategic"  # Targets specific task families
    COORDINATED = "coordinated"  # Multiple adversaries collude


@dataclass
class ConditionSpec:
    name: str
    description: str
    n_solvers: int = 1
    use_critic: bool = False
    use_reconcile: bool = False
    vote: bool = False
    use_memory: bool = False
    adversary: bool = False
    adversary_count: int = 0  # Number of adversary solvers (0 = use legacy bool)
    adversary_strategy: str = AdversaryStrategy.NOISE
    divergence_threshold: float = 0.35


@dataclass
class ConditionMetrics:
    name: str
    accuracy: float
    avg_tokens: float
    avg_confidence: float
    disagreement_rate: float
    reconcile_rate: float
    critic_flag_rate: float
    adversary_rate: float  # Rate adversary was flagged
    adversary_selection_rate: float = 0.0  # Rate adversary answer was selected
    adversary_block_rate: float = 0.0  # Rate adversary was blocked
    adversary_success_rate: float = 0.0  # Rate adversary caused wrong answer
    note: str = ""


@dataclass
class RunSummary:
    run_id: str
    task_count: int
    conditions: list[ConditionMetrics]
    memory_artifacts: list[MemoryArtifact]
    related_work: list["RelatedWorkItem"]
    bib_entries: str
    family_metrics: dict[str, dict[str, dict[str, float]]]
    critique_summary: CritiqueSummary
    reputation_summary: dict[str, dict[str, dict[str, float]]]
    reputation_drift: list[dict[str, Any]]
    peer_reviews: list[dict] = field(default_factory=list)
    review_gate_passed: bool | None = None


@dataclass
class TrackAConfig:
    n_tasks: int = 200
    seed: int = 42
    difficulty: float = 0.5
    output_dir: str = "runs/swarm_collate"
    conditions: list[ConditionSpec] = field(default_factory=list)
    retrieval_policy: RetrievalPolicy = field(default_factory=RetrievalPolicy)
    write_policy: WritePolicy = field(default_factory=WritePolicy)
    enable_agentrxiv: bool = True
    agentrxiv_url: str | None = None
    enable_pdf: bool = False
    publish_to_agentrxiv: bool = False
    enable_peer_review: bool = True
    query: str = "SWARM Track A verifiable reasoning"
    llm_enabled: bool = False
    llm_config: "LLMConfig | None" = None

    def __post_init__(self) -> None:
        if not self.conditions:
            self.conditions = default_conditions()


def default_conditions() -> list[ConditionSpec]:
    return [
        ConditionSpec(
            name="single",
            description="Single solver baseline",
            n_solvers=1,
        ),
        ConditionSpec(
            name="diverge",
            description="Two solvers, pick highest confidence",
            n_solvers=2,
        ),
        ConditionSpec(
            name="sda",
            description="Diverge + reconcile on disagreement",
            n_solvers=2,
            use_reconcile=True,
        ),
        ConditionSpec(
            name="critic",
            description="Diverge + critic + reconcile",
            n_solvers=2,
            use_critic=True,
            use_reconcile=True,
        ),
        ConditionSpec(
            name="memory",
            description="SDA + memory retrieval",
            n_solvers=2,
            use_reconcile=True,
            use_critic=True,
            use_memory=True,
        ),
    ]


def adversarial_conditions() -> list[ConditionSpec]:
    """Return adversarial condition variants for robustness testing.

    All conditions use voting (majority vote) for adversary resistance.
    """
    return [
        # Basic adversary injection - tests if high-confidence wrong answers get selected
        ConditionSpec(
            name="adv_noise",
            description="Two solvers + 1 noisy adversary + voting",
            n_solvers=2,
            adversary=True,
            adversary_count=1,
            adversary_strategy=AdversaryStrategy.NOISE,
            vote=True,
            use_critic=True,
            use_reconcile=True,
        ),
        # Confident wrong adversary - always wrong with 95% confidence
        ConditionSpec(
            name="adv_confident",
            description="Two solvers + 1 confident-wrong adversary + voting",
            n_solvers=2,
            adversary=True,
            adversary_count=1,
            adversary_strategy=AdversaryStrategy.CONFIDENT_WRONG,
            vote=True,
            use_critic=True,
            use_reconcile=True,
        ),
        # Strategic adversary - plausible errors that are hard to detect
        ConditionSpec(
            name="adv_strategic",
            description="Two solvers + 1 strategic adversary + voting",
            n_solvers=2,
            adversary=True,
            adversary_count=1,
            adversary_strategy=AdversaryStrategy.STRATEGIC,
            vote=True,
            use_critic=True,
            use_reconcile=True,
        ),
        # Sycophant adversary - agrees with peers but introduces subtle errors
        ConditionSpec(
            name="adv_sycophant",
            description="Two solvers + 1 sycophant adversary + voting",
            n_solvers=2,
            adversary=True,
            adversary_count=1,
            adversary_strategy=AdversaryStrategy.SYCOPHANT,
            vote=True,
            use_critic=True,
            use_reconcile=True,
        ),
        # Coordinated attack - multiple adversaries collude
        ConditionSpec(
            name="adv_coordinated",
            description="Two solvers + 2 coordinated adversaries + voting",
            n_solvers=2,
            adversary=True,
            adversary_count=2,
            adversary_strategy=AdversaryStrategy.COORDINATED,
            vote=True,
            use_critic=True,
            use_reconcile=True,
        ),
        # Majority adversary - adversaries outnumber honest solvers
        ConditionSpec(
            name="adv_majority",
            description="Two solvers + 3 adversaries (adversary majority) + voting",
            n_solvers=2,
            adversary=True,
            adversary_count=3,
            adversary_strategy=AdversaryStrategy.NOISE,
            vote=True,
            use_critic=True,
            use_reconcile=True,
        ),
        # Adversary with memory - can adversarial artifacts poison memory?
        ConditionSpec(
            name="adv_memory",
            description="Memory condition + 1 strategic adversary + voting",
            n_solvers=2,
            adversary=True,
            adversary_count=1,
            adversary_strategy=AdversaryStrategy.STRATEGIC,
            vote=True,
            use_critic=True,
            use_reconcile=True,
            use_memory=True,
        ),
    ]


def all_conditions() -> list[ConditionSpec]:
    """Return all conditions including adversarial variants."""
    return default_conditions() + adversarial_conditions()


class TrackATaskGenerator:
    """Generate Track A tasks with configurable difficulty and task families."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)
        self._people = ["Ava", "Ben", "Cleo"]
        self._pets = ["cat", "dog", "owl"]
        self._colors = ["red", "green", "blue"]
        # Extended for 4x4 grids
        self._people_4 = ["Ava", "Ben", "Cleo", "Dana"]
        self._pets_4 = ["cat", "dog", "owl", "fish"]
        self._colors_4 = ["red", "green", "blue", "yellow"]
        self._hobbies = ["chess", "piano", "gardening", "painting"]

    def _format_clues(self, clues: list[str]) -> str:
        """Format a list of clues into numbered text."""
        return " ".join(f"{i+1}) {clue}" for i, clue in enumerate(clues))

    def generate(self, n_tasks: int, difficulty: float) -> list[ReasoningTask]:
        tasks: list[ReasoningTask] = []
        for idx in range(n_tasks):
            roll = self._rng.random()
            # Original families (~65%)
            if roll < 0.15:
                task = self._expression_task(idx, difficulty)
            elif roll < 0.30:
                task = self._word_problem_task(idx, difficulty)
            elif roll < 0.42:
                task = self._algebra_task(idx, difficulty)
            elif roll < 0.54:
                task = self._symbolic_task(idx, difficulty)
            elif roll < 0.62:
                task = self._logic_grid_task(idx)
            # New harder families (~38%)
            elif roll < 0.70:
                task = self._system_of_equations_task(idx, difficulty)
            elif roll < 0.76:
                task = self._modular_arithmetic_task(idx, difficulty)
            elif roll < 0.82:
                task = self._inequality_task(idx, difficulty)
            elif roll < 0.88:
                task = self._logic_grid_4x4_task(idx)
            elif roll < 0.94:
                task = self._knights_knaves_task(idx)
            else:
                task = self._code_verify_task(idx, difficulty)
            tasks.append(task)
        return tasks

    def _expression_task(self, idx: int, difficulty: float) -> ReasoningTask:
        max_val = int(10 + 90 * difficulty)
        a = self._rng.randint(2, max_val)
        b = self._rng.randint(2, max_val)
        c = self._rng.randint(2, max_val)
        op = self._rng.choice(["+", "-", "*", "/"])
        if op == "/":
            b = self._rng.randint(2, max_val)
            a = b * self._rng.randint(2, max_val)
        expr = f"({a} {op} {b}) + {c}"
        answer = _safe_eval(expr)
        prompt = f"Compute the value of {expr}. Provide only the final numeric answer."
        return ReasoningTask(
            task_id=f"expr_{idx}",
            prompt=prompt,
            answer=_format_number(answer),
            metadata={"expression": expr, "family": "arithmetic"},
        )

    def _word_problem_task(self, idx: int, difficulty: float) -> ReasoningTask:
        max_val = int(12 + 60 * difficulty)
        a = self._rng.randint(3, max_val)
        b = self._rng.randint(3, max_val)
        c = self._rng.randint(2, 9)
        total = a + b
        if total % c != 0:
            total += c - (total % c)
        a = total - b
        prompt = (
            "A lab has {a} samples and receives {b} more. "
            "They split them evenly into {c} trays. "
            "How many samples are in each tray?"
        ).format(a=a, b=b, c=c)
        expr = f"({a} + {b}) / {c}"
        answer = _safe_eval(expr)
        return ReasoningTask(
            task_id=f"word_{idx}",
            prompt=prompt,
            answer=_format_number(answer),
            metadata={"expression": expr, "family": "word"},
        )

    def _algebra_task(self, idx: int, difficulty: float) -> ReasoningTask:
        max_val = int(8 + 70 * difficulty)
        if self._rng.random() < 0.5:
            a = self._rng.randint(2, max_val)
            x = self._rng.randint(2, max_val)
            b = self._rng.randint(1, max_val)
            c = a * x + b
            prompt = f"Solve for x: {a}x + {b} = {c}."
            expr = f"({c} - {b}) / {a}"
            lhs = f"{a}*x + {b}"
            rhs = f"{c}"
        else:
            a = self._rng.randint(2, max_val)
            x = self._rng.randint(2, max_val)
            b = self._rng.randint(1, max_val)
            c = a * (x + b)
            prompt = f"Solve for x: {a}(x + {b}) = {c}."
            expr = f"({c} / {a}) - {b}"
            lhs = f"{a}*(x + {b})"
            rhs = f"{c}"

        answer = _safe_eval(expr)
        return ReasoningTask(
            task_id=f"alg_{idx}",
            prompt=prompt,
            answer=_format_number(answer),
            metadata={"expression": expr, "family": "algebra", "lhs": lhs, "rhs": rhs},
        )

    def _symbolic_task(self, idx: int, difficulty: float) -> ReasoningTask:
        max_val = int(6 + 40 * difficulty)
        if self._rng.random() < 0.5:
            a = self._rng.randint(1, max_val)
            b = self._rng.randint(1, max_val)
            c = self._rng.randint(1, max_val)
            prompt = (
                f"Let a={a}, b={b}. Compute 2a^2 + 3b - {c}. "
                "Provide only the final numeric answer."
            )
            expr = f"2*({a}**2) + 3*{b} - {c}"
        else:
            a = self._rng.randint(1, max_val)
            b = self._rng.randint(1, max_val)
            c = self._rng.randint(1, max_val)
            d = self._rng.randint(1, max_val)
            k = self._rng.randint(1, max_val)
            prompt = (
                f"Let f(x) = {a}x + {b} and g(x) = {c}x^2 + {d}. "
                f"Compute g(f({k})). Provide only the final numeric answer."
            )
            expr = f"{c}*(({a}*{k}+{b})**2) + {d}"

        answer = _safe_eval(expr)
        return ReasoningTask(
            task_id=f"sym_{idx}",
            prompt=prompt,
            answer=_format_number(answer),
            metadata={"expression": expr, "family": "symbolic"},
        )

    def _logic_grid_task(self, idx: int) -> ReasoningTask:
        people = list(self._people)
        pets = list(self._pets)
        colors = list(self._colors)
        self._rng.shuffle(pets)
        self._rng.shuffle(colors)

        mapping = {
            person: {"pet": pet, "color": color}
            for person, pet, color in zip(people, pets, colors, strict=False)
        }

        target_person = self._rng.choice(people)
        target_pet = mapping[target_person]["pet"]
        target_color = mapping[target_person]["color"]

        remaining = [p for p in people if p != target_person]
        person_color_clue = self._rng.choice(remaining)
        person_neg = [p for p in remaining if p != person_color_clue][0]
        color_other = mapping[person_color_clue]["color"]

        clues = [
            f"{person_color_clue} wears {color_other}.",
            f"{person_neg} does not wear {target_color}.",
            f"The person wearing {target_color} has the {target_pet}.",
        ]
        clues_text = self._format_clues(clues)
        prompt = (
            "Three researchers (Ava, Ben, Cleo) each have a different pet "
            "(cat, dog, owl) and wear a different color (red, green, blue). "
            "Clues: "
            + clues_text
            + f" Question: Who has the {target_pet}? Provide the name only."
        )

        return ReasoningTask(
            task_id=f"logic_{idx}",
            prompt=prompt,
            answer=target_person,
            metadata={
                "solution": target_person,
                "options": people,
                "family": "logic_grid",
            },
        )

    # ========== NEW HARDER TASK FAMILIES ==========

    def _system_of_equations_task(self, idx: int, difficulty: float) -> ReasoningTask:
        """System of 2 linear equations with 2 unknowns."""
        max_val = int(5 + 15 * difficulty)

        # Generate solution first, then construct equations
        x_sol = self._rng.randint(1, max_val)
        y_sol = self._rng.randint(1, max_val)

        # Coefficients for equation 1: a1*x + b1*y = c1
        a1 = self._rng.randint(1, max_val)
        b1 = self._rng.randint(1, max_val)
        c1 = a1 * x_sol + b1 * y_sol

        # Coefficients for equation 2: a2*x + b2*y = c2
        # Ensure linearly independent (different ratio)
        a2 = self._rng.randint(1, max_val)
        b2 = self._rng.randint(1, max_val)
        while a1 * b2 == a2 * b1:  # Avoid parallel lines
            b2 = self._rng.randint(1, max_val)
        c2 = a2 * x_sol + b2 * y_sol

        prompt = (
            f"Solve the system of equations:\n"
            f"  {a1}x + {b1}y = {c1}\n"
            f"  {a2}x + {b2}y = {c2}\n"
            f"What is the value of x? Provide only the numeric answer."
        )

        return ReasoningTask(
            task_id=f"sys_{idx}",
            prompt=prompt,
            answer=str(x_sol),
            metadata={
                "expression": str(x_sol),  # Direct evaluable expression
                "family": "system_eq",
                "solution": str(x_sol),
                "x": str(x_sol),
                "y": str(y_sol),
            },
        )

    def _modular_arithmetic_task(self, idx: int, difficulty: float) -> ReasoningTask:
        """Modular arithmetic: remainders and modular operations."""
        max_val = int(20 + 80 * difficulty)
        modulus = self._rng.randint(3, 13)

        task_type = self._rng.choice(["remainder", "mod_add", "mod_mult", "mod_power"])

        if task_type == "remainder":
            n = self._rng.randint(max_val, max_val * 3)
            answer = n % modulus
            prompt = f"What is {n} mod {modulus}? Provide only the numeric answer."
            expr = f"{n} % {modulus}"
        elif task_type == "mod_add":
            a = self._rng.randint(1, max_val)
            b = self._rng.randint(1, max_val)
            answer = (a + b) % modulus
            prompt = f"Compute ({a} + {b}) mod {modulus}. Provide only the numeric answer."
            expr = f"({a} + {b}) % {modulus}"
        elif task_type == "mod_mult":
            a = self._rng.randint(1, max_val)
            b = self._rng.randint(1, max_val)
            answer = (a * b) % modulus
            prompt = f"Compute ({a} × {b}) mod {modulus}. Provide only the numeric answer."
            expr = f"({a} * {b}) % {modulus}"
        else:  # mod_power
            base = self._rng.randint(2, 7)
            exp = self._rng.randint(2, 5)
            answer = pow(base, exp, modulus)
            prompt = f"Compute {base}^{exp} mod {modulus}. Provide only the numeric answer."
            expr = f"pow({base}, {exp}, {modulus})"

        return ReasoningTask(
            task_id=f"mod_{idx}",
            prompt=prompt,
            answer=str(answer),
            metadata={"expression": expr, "family": "modular", "solution": str(answer)},
        )

    def _inequality_task(self, idx: int, difficulty: float) -> ReasoningTask:
        """Linear inequalities - find boundary or count integers."""
        max_val = int(5 + 30 * difficulty)

        task_type = self._rng.choice(["boundary", "count", "compound"])

        if task_type == "boundary":
            # ax + b < c, find largest integer x
            a = self._rng.randint(2, max_val)
            b = self._rng.randint(1, max_val)
            c = self._rng.randint(b + a * 2, b + a * max_val)
            # ax < c - b => x < (c-b)/a
            boundary = (c - b) / a
            answer = int(boundary) if boundary != int(boundary) else int(boundary) - 1
            prompt = (
                f"Find the largest integer x such that {a}x + {b} < {c}. "
                f"Provide only the numeric answer."
            )
            expr = f"floor(({c} - {b}) / {a} - epsilon)"
        elif task_type == "count":
            # Count integers in range a <= x <= b
            low = self._rng.randint(1, max_val)
            high = low + self._rng.randint(5, 20)
            answer = high - low + 1
            prompt = (
                f"How many integers x satisfy {low} ≤ x ≤ {high}? "
                f"Provide only the numeric answer."
            )
            expr = f"{high} - {low} + 1"
        else:  # compound
            # a < x + b < c, count integers
            b = self._rng.randint(1, max_val)
            a = self._rng.randint(1, 10)
            c = a + self._rng.randint(5, 15)
            # a < x + b < c => a - b < x < c - b
            low_x = a - b + 1
            high_x = c - b - 1
            answer = max(0, high_x - low_x + 1)
            prompt = (
                f"How many integers x satisfy {a} < x + {b} < {c}? "
                f"Provide only the numeric answer."
            )
            expr = f"count({a} < x + {b} < {c})"

        return ReasoningTask(
            task_id=f"ineq_{idx}",
            prompt=prompt,
            answer=str(answer),
            metadata={"expression": expr, "family": "inequality", "solution": str(answer)},
        )

    def _logic_grid_4x4_task(self, idx: int) -> ReasoningTask:
        """4x4 logic grid puzzle - harder than 3x3."""
        people = list(self._people_4)
        pets = list(self._pets_4)
        colors = list(self._colors_4)
        hobbies = list(self._hobbies)

        self._rng.shuffle(pets)
        self._rng.shuffle(colors)
        self._rng.shuffle(hobbies)

        mapping = {
            person: {"pet": pet, "color": color, "hobby": hobby}
            for person, pet, color, hobby in zip(people, pets, colors, hobbies, strict=False)
        }

        target_person = self._rng.choice(people)
        target_attr = self._rng.choice(["pet", "hobby"])
        target_value = mapping[target_person][target_attr]

        # Generate clues that allow deduction
        clues = []
        others = [p for p in people if p != target_person]

        # Clue 1: Direct assignment for one person
        p1 = self._rng.choice(others)
        clues.append(f"{p1} has the {mapping[p1]['pet']}.")

        # Clue 2: Color to hobby link
        p2 = self._rng.choice([p for p in others if p != p1])
        clues.append(f"The person wearing {mapping[p2]['color']} enjoys {mapping[p2]['hobby']}.")

        # Clue 3: Negative clue
        p3 = self._rng.choice([p for p in others if p not in [p1, p2]])
        wrong_pet = self._rng.choice([pet for pet in pets if pet != mapping[p3]["pet"]])
        clues.append(f"{p3} does not have the {wrong_pet}.")

        # Clue 4: Link target to identifiable attribute
        clues.append(
            f"The person wearing {mapping[target_person]['color']} has the {target_value}."
        )

        clues_text = self._format_clues(clues)
        prompt = (
            f"Four researchers ({', '.join(people)}) each have a different pet "
            f"({', '.join(self._pets_4)}), wear a different color "
            f"({', '.join(self._colors_4)}), and have a hobby ({', '.join(self._hobbies)}). "
            f"Clues: {clues_text} "
            f"Question: Who has the {target_value}? Provide the name only."
        )

        return ReasoningTask(
            task_id=f"logic4_{idx}",
            prompt=prompt,
            answer=target_person,
            metadata={
                "solution": target_person,
                "options": people,
                "family": "logic_grid_4x4",
            },
        )

    def _knights_knaves_task(self, idx: int) -> ReasoningTask:
        """Knights and Knaves puzzle - truth-tellers vs liars."""
        names = ["Alice", "Bob", "Carol"]
        n = self._rng.randint(2, 3)
        selected = names[:n]

        # Assign roles: True = Knight (truth-teller), False = Knave (liar)
        roles = {name: self._rng.choice([True, False]) for name in selected}

        # Generate statements
        statements = []
        for name in selected:
            is_knight = roles[name]
            target = self._rng.choice(selected)

            if target == name:
                # Self-reference
                stmt = f"{name} says: 'I am a knight.'"
            else:
                # About another
                target_is_knight = roles[target]
                if is_knight:
                    # Knight tells truth
                    claim = "knight" if target_is_knight else "knave"
                else:
                    # Knave lies
                    claim = "knave" if target_is_knight else "knight"
                stmt = f"{name} says: '{target} is a {claim}.'"
            statements.append(stmt)

        statements_text = " ".join(statements)
        target_person = self._rng.choice(selected)
        answer = "knight" if roles[target_person] else "knave"

        prompt = (
            f"On an island, knights always tell the truth and knaves always lie. "
            f"You meet {n} people: {', '.join(selected)}. "
            f"{statements_text} "
            f"Is {target_person} a knight or a knave? Answer with only 'knight' or 'knave'."
        )

        return ReasoningTask(
            task_id=f"kk_{idx}",
            prompt=prompt,
            answer=answer,
            metadata={
                "solution": answer,
                "options": ["knight", "knave"],
                "family": "knights_knaves",
                "roles": {k: "knight" if v else "knave" for k, v in roles.items()},
            },
        )

    def _code_verify_task(self, idx: int, difficulty: float) -> ReasoningTask:
        """Code verification task - trace through simple function."""
        task_type = self._rng.choice(["trace", "bug", "output"])
        max_val = int(5 + 20 * difficulty)

        if task_type == "trace":
            # Trace variable value through loop
            n = self._rng.randint(3, min(8, max_val))
            start = self._rng.randint(0, 5)
            step = self._rng.randint(1, 3)
            answer = start + step * n
            code = f"""x = {start}
for i in range({n}):
    x = x + {step}"""
            prompt = (
                f"What is the value of x after this code runs?\n```python\n{code}\n```\n"
                f"Provide only the numeric answer."
            )
            expr = f"{start} + {step} * {n}"
        elif task_type == "bug":
            # Find the bug - off-by-one errors
            n = self._rng.randint(3, 6)
            # Buggy sum: sum of 1 to n but with off-by-one
            buggy_sum = (n - 1) * n // 2  # Missing last element
            code = f"""def sum_to_n(n):
    total = 0
    for i in range(n):  # Bug: should be range(n+1) or range(1, n+1)
        total += i
    return total

result = sum_to_n({n})"""
            prompt = (
                f"What value does this buggy code return?\n```python\n{code}\n```\n"
                f"Provide only the numeric answer."
            )
            answer = buggy_sum
            expr = f"sum(range({n}))"
        else:  # output
            # Simple function output
            a = self._rng.randint(2, max_val)
            b = self._rng.randint(1, max_val)
            c = self._rng.randint(1, max_val)
            func_type = self._rng.choice(["linear", "quadratic", "factorial"])

            if func_type == "linear":
                answer = a * c + b
                code = f"""def f(x):
    return {a} * x + {b}

result = f({c})"""
                expr = f"{a} * {c} + {b}"
            elif func_type == "quadratic":
                answer = a * c * c + b * c
                code = f"""def f(x):
    return {a} * x * x + {b} * x

result = f({c})"""
                expr = f"{a} * {c}**2 + {b} * {c}"
            else:  # factorial
                n = self._rng.randint(3, 6)
                answer = 1
                for i in range(1, n + 1):
                    answer *= i
                code = f"""def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial({n})"""
                expr = f"{n}!"

            prompt = (
                f"What value does 'result' hold after this code runs?\n"
                f"```python\n{code}\n```\n"
                f"Provide only the numeric answer."
            )

        return ReasoningTask(
            task_id=f"code_{idx}",
            prompt=prompt,
            answer=str(answer),
            metadata={"expression": expr, "family": "code_verify", "solution": str(answer)},
        )


class BaseSolver:
    def solve(self, task: ReasoningTask, memory_hint: str) -> Solution:  # pragma: no cover
        raise NotImplementedError


class HeuristicSolver(BaseSolver):
    def __init__(
        self,
        name: str,
        role: str,
        *,
        noise_rate: float = 0.0,
        base_confidence: float = 0.7,
        seed: int | None = None,
    ) -> None:
        self.name = name
        self.role = role
        self.noise_rate = noise_rate
        self.base_confidence = base_confidence
        self._rng = random.Random(seed)

    def solve(self, task: ReasoningTask, memory_hint: str) -> Solution:
        expr: str = task.metadata.get("expression", "")
        solution = task.metadata.get("solution")
        options: list[str] = task.metadata.get("options", [])

        if solution is not None:
            answer_str = str(solution)
        elif expr:
            answer_str = _format_number(_safe_eval(expr))
        else:
            answer_str = _format_number(_fallback_parse(task.prompt))

        if self._rng.random() < self.noise_rate:
            answer_str = _introduce_noise(answer_str, options, self._rng)

        confidence = max(0.05, min(0.95, self.base_confidence))
        raw = f"Answer: {answer_str}\nConfidence: {confidence:.2f}"
        return Solution(
            solver_name=self.name,
            role=self.role,
            answer=answer_str,
            confidence=confidence,
            raw_text=raw,
            tokens=0,
        )


class AdversarialSolver(BaseSolver):
    """Adversarial solver with configurable attack strategies."""

    def __init__(
        self,
        name: str,
        strategy: str = AdversaryStrategy.NOISE,
        *,
        base_confidence: float = 0.85,
        seed: int | None = None,
        colluders: list["AdversarialSolver"] | None = None,
    ) -> None:
        self.name = name
        self.role = "adversary"
        self.strategy = strategy
        self.base_confidence = base_confidence
        self._rng = random.Random(seed)
        self.colluders = colluders or []
        self._last_answer: str | None = None  # For coordinated attacks
        self._peer_answers: list[str] = []  # Track peer answers for sycophant

    def set_peer_answers(self, answers: list[str]) -> None:
        """Set answers from peer solvers (for sycophant strategy)."""
        self._peer_answers = answers

    def solve(self, task: ReasoningTask, memory_hint: str) -> Solution:
        # First compute the correct answer
        expr: str = task.metadata.get("expression", "")
        solution = task.metadata.get("solution")
        options: list[str] = task.metadata.get("options", [])

        if solution is not None:
            correct_answer = str(solution)
        elif expr:
            correct_answer = _format_number(_safe_eval(expr))
        else:
            correct_answer = _format_number(_fallback_parse(task.prompt))

        # Apply adversarial strategy
        if self.strategy == AdversaryStrategy.CONFIDENT_WRONG:
            answer = self._confident_wrong(correct_answer, options)
            confidence = 0.95
        elif self.strategy == AdversaryStrategy.SYCOPHANT:
            answer = self._sycophant(correct_answer, options)
            confidence = self.base_confidence - 0.05  # Slightly less confident
        elif self.strategy == AdversaryStrategy.STRATEGIC:
            answer = _introduce_strategic_error(correct_answer, task, options, self._rng)
            confidence = self.base_confidence
        elif self.strategy == AdversaryStrategy.COORDINATED:
            answer = self._coordinated(correct_answer, options)
            confidence = self.base_confidence
        else:  # NOISE (default)
            answer = _introduce_noise(correct_answer, options, self._rng)
            confidence = self.base_confidence

        self._last_answer = answer
        raw = f"Answer: {answer}\nConfidence: {confidence:.2f}"
        return Solution(
            solver_name=self.name,
            role=self.role,
            answer=answer,
            confidence=confidence,
            raw_text=raw,
            tokens=0,
        )

    def _confident_wrong(self, correct: str, options: list[str]) -> str:
        """Always return a wrong answer with maximum confidence."""
        if options:
            wrong = [opt for opt in options if opt != correct]
            if wrong:
                return self._rng.choice(wrong)
        try:
            num = float(correct)
            # Make a significant error
            return _format_number(num + self._rng.choice([-10, -5, 5, 10]))
        except (TypeError, ValueError):
            return "WRONG"

    def _sycophant(self, correct: str, options: list[str]) -> str:
        """Agree with peers but subtly modify to be wrong."""
        if self._peer_answers:
            # Take a random peer answer and slightly modify it
            peer = self._rng.choice(self._peer_answers)
            try:
                num = float(peer)
                # Subtle modification that looks like agreement
                return _format_number(num + self._rng.choice([-0.1, 0.1, -1, 1]))
            except (TypeError, ValueError) as exc:
                logger.debug("Sycophant strategy failed to parse peer answer: %s", exc)
        # Fallback to strategic error
        return _introduce_noise(correct, options, self._rng)

    def _coordinated(self, correct: str, options: list[str]) -> str:
        """Coordinate with other adversaries to present unified wrong answer."""
        if self.colluders:
            # Check if any colluder has already answered
            for colluder in self.colluders:
                if colluder._last_answer is not None:
                    return colluder._last_answer
        # First adversary sets the coordinated wrong answer
        return self._confident_wrong(correct, options)


class LLMClient:
    def __init__(self, config: "LLMConfig") -> None:
        self.config = config

    def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        provider = self.config.provider
        if provider == LLMProvider.ANTHROPIC:
            return self._call_anthropic(system_prompt, user_prompt)
        if provider == LLMProvider.OPENAI:
            return self._call_openai(system_prompt, user_prompt)
        if provider == LLMProvider.OLLAMA:
            return self._call_ollama(system_prompt, user_prompt)
        raise ValueError(f"Unknown provider: {provider}")

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        try:
            import anthropic
        except ImportError as err:  # pragma: no cover - optional
            raise ImportError(
                "anthropic package not installed. Install swarm-safety[llm]."
            ) from err
        client = anthropic.Anthropic(api_key=self.config.api_key)
        message = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        block = message.content[0] if message.content else None
        text = block.text if block is not None and hasattr(block, "text") else ""
        usage = getattr(message, "usage", None)
        in_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        out_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        return text, in_tokens, out_tokens

    def _call_openai(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        try:
            import openai
        except ImportError as err:  # pragma: no cover - optional
            raise ImportError(
                "openai package not installed. Install swarm-safety[llm]."
            ) from err
        client = openai.OpenAI(api_key=self.config.api_key)
        response = client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text = response.choices[0].message.content if response.choices else ""
        usage = getattr(response, "usage", None)
        in_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        out_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        return text or "", in_tokens, out_tokens

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> tuple[str, int, int]:
        try:
            import httpx
        except ImportError as err:  # pragma: no cover - optional
            raise ImportError(
                "httpx package not installed. Install swarm-safety[llm]."
            ) from err
        base_url = self.config.base_url or "http://localhost:11434"
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
            "stream": False,
        }
        with httpx.Client(timeout=self.config.timeout) as client:
            resp = client.post(f"{base_url}/api/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()
        text = data.get("message", {}).get("content", "")
        return text, 0, 0


class LLMSolver(BaseSolver):
    def __init__(self, name: str, role: str, config: "LLMConfig") -> None:
        self.name = name
        self.role = role
        self.client = LLMClient(config)
        self.config = config

    def solve(self, task: ReasoningTask, memory_hint: str) -> Solution:
        system_prompt = f"You are a {self.role} math solver."
        user_prompt = (
            f"Problem: {task.prompt}\n\n"
            f"Helpful memory (if any):\n{memory_hint or 'None'}\n\n"
            "Return a JSON object with keys: answer (string), confidence (0-1), reasoning (string)."
        )
        text, in_tokens, out_tokens = self.client.complete(system_prompt, user_prompt)
        answer, confidence = _parse_llm_answer(text)
        return Solution(
            solver_name=self.name,
            role=self.role,
            answer=answer,
            confidence=confidence,
            raw_text=text,
            tokens=in_tokens + out_tokens,
        )


class SimpleCritic:
    def critique(self, solutions: list[Solution], task: ReasoningTask) -> str | None:
        if len(solutions) < 2:
            return None
        answers = [_normalize_answer(sol.answer) for sol in solutions]
        if any(not ans for ans in answers):
            return "missing answer"

        if task.metadata.get("family") == "logic_grid":
            options: list[str] = task.metadata.get("options", [])
            if options and any(ans not in {opt.lower() for opt in options} for ans in answers):
                return "invalid entity for logic grid"
            if len(set(answers)) > 1:
                return "logic grid inconsistency"
            return None

        numeric_answers = []
        for ans in answers:
            try:
                numeric_answers.append(float(ans))
            except (TypeError, ValueError):
                return "non-numeric answer in numeric task"

        family = task.metadata.get("family", "numeric")
        expr = task.metadata.get("expression", "")
        expected = _safe_eval(expr) if expr else None

        if expected is not None:
            diffs = [abs(val - expected) for val in numeric_answers]
            if all(diff > 1e-3 for diff in diffs):
                if family == "symbolic":
                    return "symbolic evaluation mismatch"
                if family == "algebra":
                    return "no solver matches derived solution"
                return "derived-solution mismatch"

            if expected.is_integer():
                if any(abs(val - round(val)) > 1e-6 for val in numeric_answers):
                    if family in ("arithmetic", "word"):
                        return "non-integer answer for integer arithmetic"
                    if family == "algebra":
                        return "non-integer solution for linear equation"

        if family == "word" and any(val < 0 for val in numeric_answers):
            return "negative answer in word problem"

        if family == "algebra" and expected is not None:
            if any(abs(val - expected) > 1e-3 for val in numeric_answers):
                return "algebra back-solve mismatch"
            lhs_expr = task.metadata.get("lhs", "")
            rhs_expr = task.metadata.get("rhs", "")
            if lhs_expr and rhs_expr:
                lhs_val = _safe_eval(lhs_expr.replace("x", str(expected)))
                rhs_val = _safe_eval(rhs_expr)
                if abs(lhs_val - rhs_val) > 1e-3:
                    return "algebra substitution mismatch"

        spread = max(numeric_answers) - min(numeric_answers)
        top_conf = sorted((sol.confidence for sol in solutions), reverse=True)
        if len(set(answers)) > 1 and top_conf[0] >= 0.7 and top_conf[1] >= 0.6:
            return "confident disagreement"
        if len(set(answers)) > 1 and spread >= 5:
            return "large numeric divergence"
        if len(set(answers)) > 1:
            return "moderate numeric divergence"
        return None


@dataclass
class ReputationStats:
    total: int = 0
    correct: int = 0
    ema_accuracy: float = 0.5
    deviation_count: int = 0


class ReputationTracker:
    def __init__(self, ema_alpha: float = 0.1) -> None:
        self.ema_alpha = ema_alpha
        self.solver_stats: dict[str, ReputationStats] = {}
        self.role_stats: dict[str, ReputationStats] = {}

    def _get(self, store: dict[str, ReputationStats], key: str) -> ReputationStats:
        if key not in store:
            store[key] = ReputationStats()
        return store[key]

    def update(self, solutions: list[Solution], ground_truth: str) -> None:
        majority = _majority_answer(solutions)
        for sol in solutions:
            correct = _answers_match(sol.answer, ground_truth)
            solver_stats = self._get(self.solver_stats, sol.solver_name)
            role_stats = self._get(self.role_stats, sol.role)
            self._update_stats(solver_stats, correct, sol, majority)
            self._update_stats(role_stats, correct, sol, majority)

    def _update_stats(
        self,
        stats: ReputationStats,
        correct: bool,
        sol: Solution,
        majority: str | None,
    ) -> None:
        stats.total += 1
        if correct:
            stats.correct += 1
        stats.ema_accuracy = (
            (1 - self.ema_alpha) * stats.ema_accuracy
            + self.ema_alpha * (1.0 if correct else 0.0)
        )
        if majority and _normalize_answer(sol.answer) != majority:
            stats.deviation_count += 1

    def score(self, solver_name: str, role: str) -> float:
        scores = []
        if solver_name in self.solver_stats:
            scores.append(self.solver_stats[solver_name].ema_accuracy)
        if role in self.role_stats:
            scores.append(self.role_stats[role].ema_accuracy)
        if not scores:
            return 0.5
        return sum(scores) / len(scores)

    def deviation_rate(self, solver_name: str, role: str) -> float:
        rates = []
        if solver_name in self.solver_stats and self.solver_stats[solver_name].total:
            stats = self.solver_stats[solver_name]
            rates.append(stats.deviation_count / stats.total)
        if role in self.role_stats and self.role_stats[role].total:
            stats = self.role_stats[role]
            rates.append(stats.deviation_count / stats.total)
        return max(rates) if rates else 0.0

    def snapshot(self) -> dict[str, dict[str, dict[str, float]]]:
        def _pack(stats: ReputationStats) -> dict[str, float]:
            rate = stats.correct / stats.total if stats.total else 0.0
            dev = stats.deviation_count / stats.total if stats.total else 0.0
            return {
                "total": float(stats.total),
                "accuracy": rate,
                "ema_accuracy": stats.ema_accuracy,
                "deviation_rate": dev,
            }

        return {
            "solvers": {name: _pack(stats) for name, stats in self.solver_stats.items()},
            "roles": {name: _pack(stats) for name, stats in self.role_stats.items()},
        }


class SimpleReconciler:
    def reconcile(self, solutions: list[Solution]) -> Solution:
        # pick highest confidence solution
        if not solutions:
            return Solution("reconciler", "reconciler", "", 0.0, "")
        return max(solutions, key=lambda sol: sol.confidence)


class TrackARunner:
    """Execute Track A experiments and produce a paper draft."""

    def __init__(self, config: TrackAConfig):
        self.config = config
        self.run_id = datetime.now(timezone.utc).strftime("track_a_%Y%m%d_%H%M%S")
        self.output_dir = Path(config.output_dir) / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.episodes_path = self.output_dir / "episodes.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self.memory_store = MemoryStore(self.output_dir / "memory.jsonl")
        self.audit = BasicAudit()
        self.agentrxiv = AgentRxivBridge(config.agentrxiv_url) if config.enable_agentrxiv else None
        if self.agentrxiv:
            self.agentrxiv.set_review_store(self.output_dir / "reviews.jsonl")
        self._agentrxiv_cache: list[MemoryArtifact] = []
        self._agentrxiv_loaded = False
        self.reputation = ReputationTracker()
        self._reputation_history: list[dict[str, object]] = []
        self._episode_index = 0

    def run(self) -> RunSummary:
        tasks = TrackATaskGenerator(self.config.seed).generate(
            self.config.n_tasks, self.config.difficulty
        )
        baseline_acc = None
        condition_metrics: list[ConditionMetrics] = []
        memory_artifacts: list[MemoryArtifact] = []
        family_metrics: dict[str, dict[str, dict[str, float]]] = {}
        all_logs: list[EpisodeLog] = []

        related_work: list[RelatedWorkItem] = []
        bib_entries = ""
        if self.agentrxiv and self.agentrxiv.available():
            papers = self.agentrxiv.related_work(self.config.query, limit=5)
            related_work, bib_entries = _build_related_work(papers)

        for condition in self.config.conditions:
            logs = self._run_condition(condition, tasks)
            all_logs.extend(logs)
            metrics = self._compute_metrics(condition, logs)
            condition_metrics.append(metrics)
            if condition.name == "single":
                baseline_acc = metrics.accuracy
            family_metrics[condition.name] = self._compute_family_metrics(logs)

            artifact = self._maybe_write_memory(condition, metrics, baseline_acc, len(tasks))
            if artifact is not None:
                memory_artifacts.append(artifact)

        summary = RunSummary(
            run_id=self.run_id,
            task_count=len(tasks),
            conditions=condition_metrics,
            memory_artifacts=memory_artifacts,
            related_work=related_work,
            bib_entries=bib_entries,
            family_metrics=family_metrics,
            critique_summary=self._summarize_critiques(all_logs),
            reputation_summary=self.reputation.snapshot(),
            reputation_drift=self._reputation_history,
        )
        self._write_summary(summary)
        self._write_paper(summary)
        return summary

    def _run_condition(self, condition: ConditionSpec, tasks: list[ReasoningTask]) -> list[EpisodeLog]:
        solver_pool = self._build_solvers(condition)
        critic = SimpleCritic() if condition.use_critic else None
        reconciler = SimpleReconciler()
        logs: list[EpisodeLog] = []

        for task in tasks:
            memory_items = self._retrieve_memory(task, condition)
            memory_hint = summarize_artifacts(memory_items)

            # Handle sycophant strategy - adversaries need to see peer answers
            if condition.adversary_strategy == AdversaryStrategy.SYCOPHANT:
                # First solve with honest solvers
                honest_solutions = []
                adversary_solvers = []
                for solver in solver_pool:
                    if isinstance(solver, AdversarialSolver):
                        adversary_solvers.append(solver)
                    else:
                        honest_solutions.append(solver.solve(task, memory_hint))

                # Set peer answers for adversaries
                peer_answers = [sol.answer for sol in honest_solutions]
                for adv in adversary_solvers:
                    adv.set_peer_answers(peer_answers)

                # Now solve with adversaries
                adversary_solutions = [adv.solve(task, memory_hint) for adv in adversary_solvers]
                solutions = honest_solutions + adversary_solutions
            else:
                solutions = [solver.solve(task, memory_hint) for solver in solver_pool]
            divergence = _divergence_score(solutions)
            critic_note = critic.critique(solutions, task) if critic else None
            if critic_note:
                for sol in solutions:
                    sol.confidence = max(0.01, sol.confidence - 0.1)

            reconciled = False
            final_solution = None
            if condition.use_reconcile and divergence >= condition.divergence_threshold:
                reconciled = True
                final_solution = reconciler.reconcile(solutions)
            else:
                final_solution = self._select_solution(condition, solutions)

            correct = _answers_match(final_solution.answer, task.answer)
            critic_flagged = bool(critic_note)
            adversary_flagged = _detect_adversary(
                solutions,
                task.answer,
                reputation=self.reputation,
                threshold=self.config.write_policy.adversary_conf_threshold,
            )

            # Track if adversary's answer was selected
            adversary_selected = final_solution.role == "adversary"
            # Track if adversary was blocked (flagged and not selected despite high confidence)
            adversary_blocked = False
            if adversary_flagged:
                adv_solutions = [s for s in solutions if s.role == "adversary"]
                if adv_solutions:
                    max_adv_conf = max(s.confidence for s in adv_solutions)
                    adversary_blocked = (
                        not adversary_selected and max_adv_conf >= final_solution.confidence
                    )

            tokens = sum(sol.tokens for sol in solutions)
            log = EpisodeLog(
                episode_id=uuid.uuid4().hex[:12],
                condition=condition.name,
                task_id=task.task_id,
                prompt=task.prompt,
                ground_truth=task.answer,
                final_answer=final_solution.answer,
                correct=correct,
                confidence=final_solution.confidence,
                solver_outputs=[
                    {
                        "solver": sol.solver_name,
                        "role": sol.role,
                        "answer": sol.answer,
                        "confidence": sol.confidence,
                        "tokens": sol.tokens,
                    }
                    for sol in solutions
                ],
                divergence_score=divergence,
                reconciled=reconciled,
                critic_flagged=critic_flagged,
                critic_note=critic_note or "",
                adversary_flagged=adversary_flagged,
                adversary_selected=adversary_selected,
                adversary_blocked=adversary_blocked,
                memory_used=[artifact.artifact_id for artifact in memory_items],
                tokens=tokens,
                timestamp=datetime.now(timezone.utc).isoformat(),
                family=task.metadata.get("family", "unknown"),
            )
            logs.append(log)
            self._append_episode(log)
            self.reputation.update(solutions, task.answer)
            self._record_reputation(condition.name)

        return logs

    def _build_solvers(self, condition: ConditionSpec) -> list[BaseSolver]:
        solvers: list[BaseSolver] = []
        if self.config.llm_enabled and self.config.llm_config is not None:
            base_cfg = self.config.llm_config
            precise_cfg = _adjust_llm_config(base_cfg, temperature=0.2)
            creative_cfg = _adjust_llm_config(base_cfg, temperature=0.8)
            solvers.append(LLMSolver("precise", "precise solver", precise_cfg))
            if condition.n_solvers > 1:
                solvers.append(LLMSolver("creative", "creative solver", creative_cfg))
            for idx in range(2, condition.n_solvers):
                cfg = _adjust_llm_config(base_cfg, temperature=0.6)
                solvers.append(LLMSolver(f"solver_{idx}", "solver", cfg))
        else:
            solvers.append(HeuristicSolver("precise", "precise solver", seed=self.config.seed))
            if condition.n_solvers > 1:
                solvers.append(
                    HeuristicSolver(
                        "creative",
                        "creative solver",
                        noise_rate=0.2,
                        base_confidence=0.6,
                        seed=self.config.seed + 1,
                    )
                )
            for idx in range(2, condition.n_solvers):
                solvers.append(
                    HeuristicSolver(
                        f"solver_{idx}",
                        "solver",
                        noise_rate=0.15,
                        base_confidence=0.55,
                        seed=self.config.seed + idx,
                    )
                )

        # Add adversarial solvers
        if condition.adversary:
            adversary_count = max(1, condition.adversary_count)
            strategy = condition.adversary_strategy

            if strategy == AdversaryStrategy.COORDINATED and adversary_count > 1:
                # Create coordinated adversaries that share state
                adversaries: list[AdversarialSolver] = []
                for idx in range(adversary_count):
                    adv = AdversarialSolver(
                        name=f"adversary_{idx}",
                        strategy=strategy,
                        base_confidence=0.85,
                        seed=self.config.seed + 100 + idx,
                    )
                    adversaries.append(adv)
                # Link colluders
                for adv in adversaries:
                    adv.colluders = [a for a in adversaries if a is not adv]
                solvers.extend(adversaries)
            else:
                # Independent adversaries
                for idx in range(adversary_count):
                    name = "adversary" if adversary_count == 1 else f"adversary_{idx}"
                    solvers.append(
                        AdversarialSolver(
                            name=name,
                            strategy=strategy,
                            base_confidence=0.85,
                            seed=self.config.seed + 100 + idx,
                        )
                    )

        return solvers

    def _select_solution(self, condition: ConditionSpec, solutions: list[Solution]) -> Solution:
        if not solutions:
            return Solution("none", "none", "", 0.0, "")
        if condition.vote and len(solutions) > 2:
            return _majority_vote(solutions)
        return max(solutions, key=lambda sol: sol.confidence)

    def _retrieve_memory(self, task: ReasoningTask, condition: ConditionSpec) -> list[MemoryArtifact]:
        if not condition.use_memory:
            return []
        query = f"{self.config.query} {task.prompt}"
        artifacts = self.memory_store.search(query, policy=self.config.retrieval_policy)
        if self.agentrxiv and self.agentrxiv.available():
            if not self._agentrxiv_loaded:
                self._agentrxiv_cache = self.agentrxiv.to_artifacts(
                    self.config.query,
                    limit=self.config.retrieval_policy.max_items,
                    min_score=self.config.retrieval_policy.min_score,
                )
                self._agentrxiv_loaded = True
            artifacts.extend(self._agentrxiv_cache)
        if len(artifacts) > self.config.retrieval_policy.max_items:
            artifacts = artifacts[: self.config.retrieval_policy.max_items]
        return artifacts

    def _compute_metrics(self, condition: ConditionSpec, logs: list[EpisodeLog]) -> ConditionMetrics:
        if not logs:
            return ConditionMetrics(condition.name, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        accuracy = sum(1 for log in logs if log.correct) / len(logs)
        avg_tokens = mean(log.tokens for log in logs)
        avg_conf = mean(log.confidence for log in logs)
        disagreement = sum(1 for log in logs if log.divergence_score >= condition.divergence_threshold)
        reconcile = sum(1 for log in logs if log.reconciled)
        disagreement_rate = disagreement / len(logs)
        reconcile_rate = reconcile / len(logs)
        critic_flag_rate = sum(1 for log in logs if log.critic_flagged) / len(logs)
        adversary_rate = sum(1 for log in logs if log.adversary_flagged) / len(logs)

        # New adversary-specific metrics
        adversary_selection_rate = sum(1 for log in logs if log.adversary_selected) / len(logs)
        adversary_block_rate = sum(1 for log in logs if log.adversary_blocked) / len(logs)
        # Adversary success = adversary selected AND answer was wrong
        adversary_success_rate = sum(
            1 for log in logs if log.adversary_selected and not log.correct
        ) / len(logs)

        return ConditionMetrics(
            name=condition.name,
            accuracy=accuracy,
            avg_tokens=avg_tokens,
            avg_confidence=avg_conf,
            disagreement_rate=disagreement_rate,
            reconcile_rate=reconcile_rate,
            critic_flag_rate=critic_flag_rate,
            adversary_rate=adversary_rate,
            adversary_selection_rate=adversary_selection_rate,
            adversary_block_rate=adversary_block_rate,
            adversary_success_rate=adversary_success_rate,
            note=condition.description,
        )

    def _maybe_write_memory(
        self,
        condition: ConditionSpec,
        metrics: ConditionMetrics,
        baseline_acc: float | None,
        n_tasks: int,
    ) -> MemoryArtifact | None:
        if condition.name == "single":
            return None
        delta = metrics.accuracy - baseline_acc if baseline_acc is not None else None
        artifact = MemoryArtifact(
            artifact_id=new_artifact_id(),
            title=f"Mechanism: {condition.name}",
            summary=(
                f"{condition.description}. Accuracy {metrics.accuracy:.2%}"
                + (f" (delta {delta:+.2%} vs baseline)." if delta is not None else ".")
            ),
            use_when="Verifiable reasoning tasks with deterministic checks",
            failure_modes=[],
            metrics={
                "accuracy": metrics.accuracy,
                "avg_tokens": metrics.avg_tokens,
                "disagreement_rate": metrics.disagreement_rate,
                "reconcile_rate": metrics.reconcile_rate,
            },
            source="local",
            source_id=self.run_id,
        )

        report = self.audit.evaluate(
            artifact=artifact,
            accuracy=metrics.accuracy,
            delta_vs_baseline=delta,
            n_tasks=n_tasks,
            critic_flag_rate=metrics.critic_flag_rate if condition.use_critic else None,
            adversary_rate=metrics.adversary_rate,
            policy=self.config.write_policy,
        )
        if self.config.write_policy.require_audit and not report.passed:
            return None

        self.memory_store.append(artifact)
        return artifact

    def _append_episode(self, log: EpisodeLog) -> None:
        with self.episodes_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(log.to_dict()) + "\n")

    def _write_summary(self, summary: RunSummary) -> None:
        payload = {
            "run_id": summary.run_id,
            "task_count": summary.task_count,
            "conditions": [
                {
                    "name": cond.name,
                    "accuracy": cond.accuracy,
                    "avg_tokens": cond.avg_tokens,
                    "avg_confidence": cond.avg_confidence,
                    "disagreement_rate": cond.disagreement_rate,
                    "reconcile_rate": cond.reconcile_rate,
                    "critic_flag_rate": cond.critic_flag_rate,
                    "adversary_rate": cond.adversary_rate,
                    "adversary_selection_rate": cond.adversary_selection_rate,
                    "adversary_block_rate": cond.adversary_block_rate,
                    "adversary_success_rate": cond.adversary_success_rate,
                    "note": cond.note,
                }
                for cond in summary.conditions
            ],
            "memory_artifacts": [artifact.to_dict() for artifact in summary.memory_artifacts],
            "related_work": [
                {
                    "title": item.title,
                    "cite_key": item.cite_key,
                    "paper_id": item.paper_id,
                }
                for item in summary.related_work
            ],
            "bib_entries": summary.bib_entries,
            "family_metrics": summary.family_metrics,
            "critique_summary": {
                "total_flags": summary.critique_summary.total_flags,
                "flag_rate": summary.critique_summary.flag_rate,
                "top_reasons": summary.critique_summary.top_reasons,
            },
            "reputation_summary": summary.reputation_summary,
            "reputation_drift": summary.reputation_drift,
            "peer_reviews": summary.peer_reviews,
            "review_gate_passed": summary.review_gate_passed,
        }
        self.summary_path.write_text(json.dumps(payload, indent=2))
        if summary.reputation_drift:
            self._write_reputation_csv(summary.reputation_drift)

    def _generate_peer_review(self, paper: Paper) -> PeerReview:
        """Lightweight self-review checking abstract, source length, completeness."""
        critiques: list[Critique] = []
        if not paper.abstract or len(paper.abstract.strip()) < 20:
            critiques.append(
                Critique(
                    severity="high",
                    category="completeness",
                    issue="Abstract is missing or too short",
                    suggestion="Add a substantive abstract",
                )
            )
        if not paper.source or len(paper.source) < 200:
            critiques.append(
                Critique(
                    severity="high",
                    category="completeness",
                    issue="Paper source is missing or too short",
                    suggestion="Ensure paper has substantive content",
                )
            )
        if paper.source and "\\section" not in paper.source:
            critiques.append(
                Critique(
                    severity="medium",
                    category="completeness",
                    issue="Paper has no sections",
                    suggestion="Add section structure",
                )
            )
        high = len([c for c in critiques if c.severity in ("high", "critical")])
        if high > 0:
            recommendation = "major_revision"
        elif critiques:
            recommendation = "minor_revision"
        else:
            recommendation = "accept"
        from swarm.research.agents import _RECOMMENDATION_RATINGS
        rating = _RECOMMENDATION_RATINGS.get(recommendation, 3)
        return PeerReview(
            review_id=uuid.uuid4().hex[:12],
            paper_id=paper.paper_id or paper.content_hash(),
            reviewer_id="track_a_self_review",
            critiques=critiques,
            recommendation=recommendation,
            summary=f"Self-review found {len(critiques)} issue(s). Recommendation: {recommendation}",
            rating=rating,
        )

    def _write_paper(self, summary: RunSummary) -> None:
        builder = PaperBuilder()
        figures, images = self._generate_plots(summary)
        conditions_payload = [
            {
                "name": cond.name,
                "accuracy": cond.accuracy,
                "avg_tokens": cond.avg_tokens,
                "note": cond.note,
            }
            for cond in summary.conditions
        ]
        context = PaperContext(
            title="SWARM Track A: Disagreement + Memory in Verifiable Reasoning",
            abstract=(
                "We benchmark SWARM coordination mechanisms on a verifiable reasoning "
                "track, comparing divergence, critique, reconciliation, and memory "
                "retrieval. We report accuracy, disagreement rates, and costs across "
                f"{summary.task_count} tasks."
            ),
            run_id=summary.run_id,
            task_count=summary.task_count,
            conditions=conditions_payload,
            related_work=summary.related_work,
            memory_items=summary.memory_artifacts,
            critique_summary=summary.critique_summary,
            family_metrics=summary.family_metrics,
            bib=summary.bib_entries,
            figures=figures,
            images=images,
        )
        paper = builder.build(context)
        tex_path = self.output_dir / "paper.tex"
        tex_path.write_text(paper.source)
        if summary.bib_entries:
            bib_path = self.output_dir / "references.bib"
            bib_path.write_text(summary.bib_entries)

        # Peer review gate
        if self.config.enable_peer_review and self.config.publish_to_agentrxiv:
            review = self._generate_peer_review(paper)
            summary.peer_reviews.append(review.to_dict())
            if self.agentrxiv:
                self.agentrxiv.submit_review(review)
            gate = QualityGates.review_gate()
            gate_result = gate.run(review.to_review())
            summary.review_gate_passed = gate_result.passed
            if not gate_result.passed:
                return

        if self.config.enable_pdf:
            try:
                pdf_path = paper_to_pdf(paper, output_path=self.output_dir / "paper.pdf")
            except PDFExportError:
                return
            if self.config.publish_to_agentrxiv and self.agentrxiv:
                result = self.agentrxiv.submit(paper, str(pdf_path))
                if result.success:
                    self.agentrxiv.trigger_update()

    def _generate_plots(self, summary: RunSummary) -> tuple[list[PaperFigure], dict[str, str]]:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            empty_figs: list[PaperFigure] = []
            empty_imgs: dict[str, str] = {}
            return empty_figs, empty_imgs

        figures: list[PaperFigure] = []
        images: dict[str, str] = {}

        names = [cond.name for cond in summary.conditions]
        accuracies = [cond.accuracy for cond in summary.conditions]
        tokens = [cond.avg_tokens for cond in summary.conditions]

        # Figure 1: Accuracy by condition
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        ax1.bar(names, accuracies, color="#4C72B0")
        ax1.set_ylim(0, 1.0)
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Track A Accuracy by Condition")
        ax1.grid(axis="y", linestyle="--", alpha=0.4)
        plt.xticks(rotation=45, ha="right")
        fig1.tight_layout()
        fig1_path = self.output_dir / "figure_accuracy.png"
        fig1.savefig(fig1_path, dpi=150)
        plt.close(fig1)

        figures.append(
            PaperFigure(
                filename=fig1_path.name,
                caption="Accuracy across coordination conditions.",
            )
        )
        images[fig1_path.name] = _encode_image(fig1_path)

        # Figure 2: Coordination rates
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        x = list(range(len(names)))
        width = 0.25
        disagreement = [cond.disagreement_rate for cond in summary.conditions]
        reconcile = [cond.reconcile_rate for cond in summary.conditions]
        adversary = [cond.adversary_rate for cond in summary.conditions]
        ax2.bar([i - width for i in x], disagreement, width=width, label="disagreement")
        ax2.bar(x, reconcile, width=width, label="reconcile")
        ax2.bar([i + width for i in x], adversary, width=width, label="adversary")
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha="right")
        ax2.set_ylim(0, 1.0)
        ax2.set_ylabel("Rate")
        ax2.set_title("Coordination & Adversary Flags")
        ax2.legend(fontsize=8, ncol=3, frameon=False)
        ax2.grid(axis="y", linestyle="--", alpha=0.4)
        fig2.tight_layout()
        fig2_path = self.output_dir / "figure_rates.png"
        fig2.savefig(fig2_path, dpi=150)
        plt.close(fig2)

        figures.append(
            PaperFigure(
                filename=fig2_path.name,
                caption="Disagreement, reconcile, and adversary-flag rates by condition.",
            )
        )
        images[fig2_path.name] = _encode_image(fig2_path)

        # Figure 3: Accuracy vs tokens (skip when no LLM solvers)
        has_token_data = any(t > 0 for t in tokens)
        if has_token_data:
            fig3, ax3 = plt.subplots(figsize=(10, 5))
            ax3.scatter(tokens, accuracies, color="#55A868", s=60)
            # Stagger annotations to reduce overlap
            offsets = [(6, 4), (-6, -12), (6, -12), (-6, 4), (10, 0), (-10, 0)]
            for i, (name, x_val, y_val) in enumerate(zip(names, tokens, accuracies, strict=False)):
                offset = offsets[i % len(offsets)]
                ax3.annotate(name, (x_val, y_val), textcoords="offset points", xytext=offset, fontsize=7)
            ax3.set_xlabel("Avg Tokens")
            ax3.set_ylabel("Accuracy")
            ax3.set_title("Efficiency Tradeoff")
            ax3.grid(True, linestyle="--", alpha=0.4)
            fig3.tight_layout()
            fig3_path = self.output_dir / "figure_efficiency.png"
            fig3.savefig(fig3_path, dpi=150)
            plt.close(fig3)

            figures.append(
                PaperFigure(
                    filename=fig3_path.name,
                    caption="Accuracy vs average token cost.",
                )
            )
            images[fig3_path.name] = _encode_image(fig3_path)

        if summary.family_metrics:
            fig4, ax4 = plt.subplots(figsize=(12, 8))
            families = _family_order(summary.family_metrics)
            conditions = [cond.name for cond in summary.conditions]
            matrix = []
            for cond in conditions:
                row = []
                for family in families:
                    # Access nested structure: {condition: {family: {metric: value}}}
                    family_data = summary.family_metrics.get(cond, {}).get(family, {})
                    if isinstance(family_data, dict):
                        row.append(family_data.get("accuracy", 0.0))
                    else:
                        # Backward compatibility with old float format
                        row.append(float(family_data) if family_data else 0.0)
                matrix.append(row)
            im = ax4.imshow(matrix, vmin=0.0, vmax=1.0, cmap="Blues")
            ax4.set_xticks(range(len(families)))
            ax4.set_xticklabels([_family_label(f) for f in families], rotation=45, ha="right", fontsize=9)
            ax4.set_yticks(range(len(conditions)))
            ax4.set_yticklabels(conditions, fontsize=9)
            ax4.set_title("Accuracy by Task Family and Condition")
            for i in range(len(conditions)):
                for j in range(len(families)):
                    ax4.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8)
            fig4.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)
            fig4.tight_layout()
            fig4_path = self.output_dir / "figure_family_heatmap.png"
            fig4.savefig(fig4_path, dpi=150)
            plt.close(fig4)

            figures.append(
                PaperFigure(
                    filename=fig4_path.name,
                    caption="Per-family accuracy (arithmetic, algebra, logic, symbolic, word).",
                )
            )
            images[fig4_path.name] = _encode_image(fig4_path)

            # Token efficiency heatmap - only generate if any tokens were used
            matrix_eff = []
            for cond in conditions:
                row = []
                for family in families:
                    family_data = summary.family_metrics.get(cond, {}).get(family, {})
                    if isinstance(family_data, dict):
                        row.append(family_data.get("token_eff", 0.0))
                    else:
                        row.append(0.0)
                matrix_eff.append(row)

            # Skip this figure if all token efficiency values are 0 (heuristic solvers)
            has_token_data = any(val > 0 for row in matrix_eff for val in row)
            if has_token_data:
                fig4b, ax4b = plt.subplots(figsize=(12, 8))
                im2 = ax4b.imshow(matrix_eff, vmin=0.0, cmap="Purples")
                ax4b.set_xticks(range(len(families)))
                ax4b.set_xticklabels([_family_label(f) for f in families], rotation=45, ha="right", fontsize=9)
                ax4b.set_yticks(range(len(conditions)))
                ax4b.set_yticklabels(conditions, fontsize=9)
                ax4b.set_title("Token Efficiency by Task Family and Condition")
                for i in range(len(conditions)):
                    for j in range(len(families)):
                        ax4b.text(j, i, f"{matrix_eff[i][j]:.1f}", ha="center", va="center", fontsize=8)
                fig4b.colorbar(im2, ax=ax4b, fraction=0.046, pad=0.04, label="Correct / 1k tokens")
                fig4b.tight_layout()
                fig4b_path = self.output_dir / "figure_family_token_eff.png"
                fig4b.savefig(fig4b_path, dpi=150)
                plt.close(fig4b)

                figures.append(
                    PaperFigure(
                        filename=fig4b_path.name,
                        caption="Per-family token efficiency (correct per 1k tokens).",
                    )
                )
                images[fig4b_path.name] = _encode_image(fig4b_path)

        if summary.reputation_drift:
            fig5, ax5 = plt.subplots(figsize=(10, 5))
            series = _build_reputation_series(summary.reputation_drift)
            for label, points in series.items():
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                ax5.plot(xs, ys, label=label)
            ax5.set_xlabel("Episode")
            ax5.set_ylabel("EMA Accuracy")
            ax5.set_title("Reputation Drift Over Episodes")
            ax5.set_ylim(0.0, 1.0)
            ax5.grid(True, linestyle="--", alpha=0.4)
            ax5.legend(fontsize=8, ncol=3, frameon=False, loc="lower right")
            fig5.tight_layout()
            fig5_path = self.output_dir / "figure_reputation_drift.png"
            fig5.savefig(fig5_path, dpi=150)
            plt.close(fig5)

            figures.append(
                PaperFigure(
                    filename=fig5_path.name,
                    caption="EMA reputation trajectories for key solvers/roles.",
                )
            )
            images[fig5_path.name] = _encode_image(fig5_path)

        return figures, images

    def _write_reputation_csv(self, drift: list[dict[str, Any]]) -> None:
        path = self.output_dir / "reputation_drift.csv"
        if not drift:
            return
        solver_keys: set[str] = set()
        role_keys: set[str] = set()
        for snap in drift:
            solvers = snap.get("solvers", {})
            roles = snap.get("roles", {})
            if isinstance(solvers, dict):
                solver_keys.update(solvers.keys())
            if isinstance(roles, dict):
                role_keys.update(roles.keys())
        solver_cols = sorted(solver_keys)
        role_cols = sorted(role_keys)
        headers = ["episode", "condition"]
        headers += [f"solver_{k}" for k in solver_cols]
        headers += [f"role_{k}" for k in role_cols]
        lines = [",".join(headers)]
        for snap in drift:
            row = [str(snap.get("episode", "")), str(snap.get("condition", ""))]
            solvers = snap.get("solvers", {})
            roles = snap.get("roles", {})
            for key in solver_cols:
                value = solvers.get(key, "")
                row.append(f"{float(value):.4f}" if value != "" else "")
            for key in role_cols:
                value = roles.get(key, "")
                row.append(f"{float(value):.4f}" if value != "" else "")
            lines.append(",".join(row))
        path.write_text("\n".join(lines))

    def _compute_family_metrics(self, logs: list[EpisodeLog]) -> dict[str, dict[str, float]]:
        """Compute per-family metrics including accuracy and token efficiency."""
        family_totals: dict[str, list] = {}  # [correct, total, tokens]
        for log in logs:
            family = log.family or "unknown"
            if family not in family_totals:
                family_totals[family] = [0, 0, 0]
            family_totals[family][1] += 1
            family_totals[family][2] += log.tokens
            if log.correct:
                family_totals[family][0] += 1

        result: dict[str, dict[str, float]] = {}
        for family, (correct, total, tokens) in family_totals.items():
            accuracy = correct / total if total else 0.0
            # Token efficiency: correct answers per 1000 tokens
            token_eff = (correct / tokens * 1000) if tokens > 0 else 0.0
            result[family] = {
                "accuracy": accuracy,
                "token_eff": token_eff,
            }
        return result

    def _summarize_critiques(self, logs: list[EpisodeLog]) -> CritiqueSummary:
        total = len(logs)
        flagged = [log for log in logs if log.critic_flagged]
        if total == 0:
            return CritiqueSummary(total_flags=0, flag_rate=0.0, top_reasons=[])
        reasons: dict[str, int] = {}
        for log in flagged:
            reason = log.critic_note.strip() or "unspecified"
            reasons[reason] = reasons.get(reason, 0) + 1
        top_reasons = [
            reason for reason, _ in sorted(reasons.items(), key=lambda item: item[1], reverse=True)[:3]
        ]
        return CritiqueSummary(
            total_flags=len(flagged),
            flag_rate=len(flagged) / total,
            top_reasons=top_reasons,
        )

    def _record_reputation(self, condition_name: str) -> None:
        solvers: dict[str, float] = {}
        roles: dict[str, float] = {}
        for key in ("precise", "creative", "adversary"):
            if key in self.reputation.solver_stats:
                solvers[key] = self.reputation.solver_stats[key].ema_accuracy
        for key in ("precise solver", "creative solver", "adversary"):
            if key in self.reputation.role_stats:
                roles[key] = self.reputation.role_stats[key].ema_accuracy
        snapshot: dict[str, Any] = {
            "episode": self._episode_index,
            "condition": condition_name,
            "solvers": solvers,
            "roles": roles,
        }
        self._reputation_history.append(snapshot)
        self._episode_index += 1


def _safe_eval(expr: str) -> float:
    if not re.fullmatch(r"[0-9+\-*/().\s]+", expr):
        return 0.0
    try:
        value = eval(expr, {"__builtins__": {}}, {})
    except (SyntaxError, NameError, TypeError, ZeroDivisionError):
        return 0.0
    return round(float(value), 4)


def _fallback_parse(prompt: str) -> float:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", prompt)
    return float(numbers[-1]) if numbers else 0.0


def _format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return str(round(float(value), 4))


def _introduce_noise(answer: str, options: list[str], rng: random.Random) -> str:
    if options:
        choices = [opt for opt in options if opt != answer]
        if choices:
            return rng.choice(choices)
    try:
        num = float(answer)
        num += rng.choice([-2, -1, 1, 2])
        return _format_number(num)
    except (TypeError, ValueError):
        return answer


def _introduce_strategic_error(
    answer: str,
    task: ReasoningTask,
    options: list[str],
    rng: random.Random,
) -> str:
    """Strategic errors that are harder to detect - plausible but wrong."""
    family = task.metadata.get("family", "")

    if family == "logic_grid" and options:
        # Pick a wrong but plausible option
        wrong = [opt for opt in options if opt != answer]
        return rng.choice(wrong) if wrong else answer

    if family in ("arithmetic", "algebra", "symbolic", "word"):
        try:
            num = float(answer)
            # Introduce subtle errors: off-by-one, sign flip, factor of 2
            error_type = rng.choice(["off_by_one", "sign", "factor", "digit_swap"])
            if error_type == "off_by_one":
                return _format_number(num + rng.choice([-1, 1]))
            elif error_type == "sign":
                return _format_number(-num) if num != 0 else "1"
            elif error_type == "factor":
                return _format_number(num * rng.choice([2, 0.5]))
            else:  # digit_swap
                s = str(int(num)) if num == int(num) else str(num)
                if len(s) >= 2:
                    i = rng.randint(0, len(s) - 2)
                    s = s[:i] + s[i + 1] + s[i] + s[i + 2:]
                return s
        except (TypeError, ValueError) as exc:
            logger.debug("Strategic error generation failed to parse numeric answer: %s", exc)

    return _introduce_noise(answer, options, rng)


def _encode_image(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def _parse_llm_answer(text: str) -> tuple[str, float]:
    try:
        data = json.loads(text)
        answer = str(data.get("answer", ""))
        confidence = float(data.get("confidence", 0.5))
        return answer, _clamp(confidence)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        logger.debug("Failed to parse structured LLM answer payload: %s", exc)
    answer_match = re.search(r"answer\s*[:=]\s*([\-0-9\.]+)", text, re.I)
    confidence_match = re.search(r"confidence\s*[:=]\s*([0-9\.]+)", text, re.I)
    answer = answer_match.group(1) if answer_match else text.strip().split("\n")[-1]
    confidence = float(confidence_match.group(1)) if confidence_match else 0.5
    return answer, _clamp(confidence)


def _normalize_answer(answer: str) -> str:
    answer = answer.strip()
    num_match = re.findall(r"-?\d+(?:\.\d+)?", answer)
    if num_match:
        try:
            value = float(num_match[-1])
            if value.is_integer():
                return str(int(value))
            return str(round(value, 4))
        except (TypeError, ValueError) as exc:
            logger.debug("Failed to normalize numeric answer %r: %s", num_match[-1], exc)
    return answer.lower()


def _answers_match(pred: str, gold: str) -> bool:
    pred_norm = _normalize_answer(pred)
    gold_norm = _normalize_answer(gold)
    try:
        return math.isclose(float(pred_norm), float(gold_norm), rel_tol=1e-3, abs_tol=1e-3)
    except ValueError:
        return pred_norm == gold_norm


def _divergence_score(solutions: list[Solution]) -> float:
    if len(solutions) < 2:
        return 0.0
    a = _normalize_answer(solutions[0].answer)
    b = _normalize_answer(solutions[1].answer)
    if a == b:
        return 0.0
    # simple normalized mismatch
    return 1.0


def _detect_adversary(
    solutions: list[Solution],
    ground_truth: str,
    *,
    reputation: "ReputationTracker",
    threshold: float = 0.8,
) -> bool:
    if not solutions:
        return False

    any_correct = any(_answers_match(sol.answer, ground_truth) for sol in solutions)
    majority = _majority_answer(solutions)
    for sol in solutions:
        if sol.confidence < threshold:
            continue
        if _answers_match(sol.answer, ground_truth):
            continue
        role = sol.role.lower()
        name = sol.solver_name.lower()
        rep = reputation.score(sol.solver_name, sol.role)
        dev = reputation.deviation_rate(sol.solver_name, sol.role)

        if "adversary" in role or "adversary" in name:
            return True
        if rep < 0.45 and dev > 0.5:
            return True
        if majority and _normalize_answer(sol.answer) != majority and dev > 0.6:
            return True

    top = max(solutions, key=lambda sol: sol.confidence)
    top_rep = reputation.score(top.solver_name, top.role)
    top_dev = reputation.deviation_rate(top.solver_name, top.role)
    if top.confidence >= threshold and not _answers_match(top.answer, ground_truth):
        if any_correct and (top_rep < 0.6 or top_dev > 0.5):
            return True
    return False


def _majority_vote(solutions: list[Solution]) -> Solution:
    counts: dict[str, int] = {}
    for sol in solutions:
        key = _normalize_answer(sol.answer)
        counts[key] = counts.get(key, 0) + 1
    winner = max(counts.items(), key=lambda item: item[1])[0]
    for sol in solutions:
        if _normalize_answer(sol.answer) == winner:
            return sol
    return solutions[0]


def _majority_answer(solutions: list[Solution]) -> str | None:
    if not solutions:
        return None
    counts: dict[str, int] = {}
    for sol in solutions:
        key = _normalize_answer(sol.answer)
        counts[key] = counts.get(key, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _family_order(family_metrics: dict[str, dict[str, dict[str, float]]]) -> list[str]:
    """Extract and order family names from nested metrics structure."""
    preferred = ["arithmetic", "algebra", "logic_grid", "symbolic", "word"]
    families = set()
    for condition_metrics in family_metrics.values():
        # condition_metrics is {family: {metric: value}}
        for family_name, family_data in condition_metrics.items():
            # Skip if this is actually a metric key (backward compat check)
            if isinstance(family_data, dict):
                families.add(family_name)
            elif family_name not in ("accuracy", "token_eff"):
                families.add(family_name)
    ordered = [fam for fam in preferred if fam in families]
    remainder = sorted(fam for fam in families if fam not in ordered)
    return ordered + remainder


def _family_label(family: str) -> str:
    return "logic" if family == "logic_grid" else family


def _build_reputation_series(
    history: list[dict[str, Any]],
) -> dict[str, list[tuple[int, float]]]:
    series: dict[str, list[tuple[int, float]]] = {}
    for snapshot in history:
        episode = int(snapshot.get("episode", 0))
        solvers = snapshot.get("solvers", {})
        roles = snapshot.get("roles", {})
        if isinstance(solvers, dict):
            for name, value in solvers.items():
                label = f"solver:{name}"
                series.setdefault(label, []).append((episode, float(value)))
        if isinstance(roles, dict):
            for name, value in roles.items():
                label = f"role:{name}"
                series.setdefault(label, []).append((episode, float(value)))
    return series


def _build_related_work(papers: list[Paper]) -> tuple[list[RelatedWorkItem], str]:
    used_keys: set[str] = set()
    items: list[RelatedWorkItem] = []
    entries: list[str] = []
    year = datetime.now(timezone.utc).year

    for paper in papers:
        title = paper.title or "Untitled"
        paper_id = paper.paper_id or ""
        key = _cite_key(title, paper_id, used_keys)
        authors = " and ".join(paper.authors) if paper.authors else "AgentRxiv Contributors"
        note = f"AgentRxiv ID: {paper_id}" if paper_id else "AgentRxiv"
        entries.append(
            "@misc{{{key},\n"
            "  title={{ {title} }},\n"
            "  author={{ {author} }},\n"
            "  howpublished={{AgentRxiv}},\n"
            "  year={{ {year} }},\n"
            "  note={{ {note} }}\n"
            "}}".format(
                key=key,
                title=_bib_escape(title),
                author=_bib_escape(authors),
                year=year,
                note=_bib_escape(note),
            )
        )
        items.append(RelatedWorkItem(title=title, cite_key=key, paper_id=paper_id))

    return items, "\n\n".join(entries)


def _cite_key(title: str, paper_id: str, used: set[str]) -> str:
    base = ""
    if paper_id:
        digits = re.sub(r"[^0-9]", "", paper_id)
        base = f"agentrxiv{digits}" if digits else ""
    if not base:
        slug = "_".join(re.findall(r"[a-z0-9]+", title.lower())[:4])
        base = f"agentrxiv_{slug}" if slug else "agentrxiv_paper"
    key = base
    counter = 1
    while key in used:
        counter += 1
        key = f"{base}_{counter}"
    used.add(key)
    return key


def _bib_escape(text: str) -> str:
    text = text.replace("\n", " ").replace("\r", " ")
    return text.replace("{", "\\{").replace("}", "\\}")


def _clamp(value: float) -> float:
    return max(0.01, min(0.99, value))


def _adjust_llm_config(config: "LLMConfig", temperature: float) -> "LLMConfig":
    return LLMConfig(
        provider=config.provider,
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=temperature,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
        max_retries=config.max_retries,
        retry_base_delay=config.retry_base_delay,
        persona=config.persona,
        system_prompt=config.system_prompt,
        cost_tracking=config.cost_tracking,
        prompt_audit_path=config.prompt_audit_path,
        prompt_audit_include_system_prompt=config.prompt_audit_include_system_prompt,
        prompt_audit_hash_system_prompt=config.prompt_audit_hash_system_prompt,
        prompt_audit_max_chars=config.prompt_audit_max_chars,
    )


