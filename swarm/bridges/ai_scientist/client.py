"""Parser for AI-Scientist output directories.

Reads ideas.json, final_info.json, review.txt, and log files
and emits typed AIScientistEvent objects.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

from swarm.bridges.ai_scientist.config import AIScientistClientConfig
from swarm.bridges.ai_scientist.events import (
    AIScientistEvent,
    AIScientistEventType,
    ExperimentRunEvent,
    IdeaEvent,
    ReviewEvent,
    WriteupEvent,
)


class AIScientistClient:
    """Parses AI-Scientist output directories into structured events."""

    def __init__(self, config: AIScientistClientConfig | None = None) -> None:
        self._config = config or AIScientistClientConfig()

    @staticmethod
    def _strip_timestamp_prefix(name: str) -> str:
        """Strip timestamp prefix from directory name if present.

        Converts YYYYMMDD_HHMMSS_idea_name to idea_name.
        If no timestamp prefix is found, returns the original name.

        Args:
            name: Directory name that may have timestamp prefix.

        Returns:
            Name with timestamp prefix stripped, or original name if no prefix.
        """
        parts = name.split("_", 2)
        if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            return parts[2]
        return name

    def parse_idea(
        self,
        ideas_data: List[Dict[str, Any]],
        idea_name: str,
    ) -> List[AIScientistEvent]:
        """Parse idea entries from ideas.json data for a specific idea."""
        events: List[AIScientistEvent] = []
        for entry in ideas_data:
            if entry.get("Name", "") != idea_name:
                continue

            idea_evt = IdeaEvent(
                idea_name=idea_name,
                interestingness=float(entry.get("Interestingness", 0)),
                feasibility=float(entry.get("Feasibility", 0)),
                novelty_score=float(entry.get("Novelty", 0)),
                novel=bool(entry.get("novel", True)),
            )

            events.append(
                AIScientistEvent(
                    event_type=AIScientistEventType.IDEA_GENERATED,
                    idea_name=idea_name,
                    phase="idea",
                    payload=idea_evt.to_dict(),
                )
            )

            # Novelty check event
            if idea_evt.novel:
                events.append(
                    AIScientistEvent(
                        event_type=AIScientistEventType.NOVELTY_CHECK_PASSED,
                        idea_name=idea_name,
                        phase="idea",
                        payload=idea_evt.to_dict(),
                    )
                )
            else:
                events.append(
                    AIScientistEvent(
                        event_type=AIScientistEventType.NOVELTY_CHECK_FAILED,
                        idea_name=idea_name,
                        phase="idea",
                        payload=idea_evt.to_dict(),
                    )
                )

        return events

    def parse_experiment_runs(
        self,
        idea_dir: str,
    ) -> List[AIScientistEvent]:
        """Parse experiment run directories (run_0/, run_1/, etc.)."""
        events: List[AIScientistEvent] = []
        idea_name = os.path.basename(idea_dir)

        # Strip timestamp prefix if present (format: YYYYMMDD_HHMMSS_idea_name)
        idea_name = self._strip_timestamp_prefix(idea_name)

        events.append(
            AIScientistEvent(
                event_type=AIScientistEventType.EXPERIMENT_STARTED,
                idea_name=idea_name,
                phase="experiment",
                payload={},
            )
        )

        run_dirs = sorted(
            [
                d
                for d in os.listdir(idea_dir)
                if d.startswith("run_") and os.path.isdir(os.path.join(idea_dir, d))
            ]
        )

        consecutive_failures = 0
        for i, run_dir in enumerate(run_dirs):
            final_info_path = os.path.join(idea_dir, run_dir, "final_info.json")
            run_evt = ExperimentRunEvent(run_index=i, retry_count=consecutive_failures)

            if os.path.isfile(final_info_path):
                try:
                    with open(final_info_path) as f:
                        info = json.load(f)
                    run_evt.success = True
                    run_evt.metrics = {
                        k: float(v) for k, v in info.items() if isinstance(v, (int, float))
                    }
                    consecutive_failures = 0
                except (json.JSONDecodeError, OSError):
                    run_evt.success = False
                    run_evt.execution_error = "Failed to parse final_info.json"
                    consecutive_failures += 1
            else:
                run_evt.success = False
                run_evt.execution_error = "No final_info.json found"
                consecutive_failures += 1

            evt_type = (
                AIScientistEventType.EXPERIMENT_RUN_COMPLETED
                if run_evt.success
                else AIScientistEventType.EXPERIMENT_RUN_FAILED
            )
            events.append(
                AIScientistEvent(
                    event_type=evt_type,
                    idea_name=idea_name,
                    phase="experiment",
                    step=i,
                    payload=run_evt.to_dict(),
                )
            )

        # Mark experiment completed if any runs succeeded
        any_success = any(
            e.event_type == AIScientistEventType.EXPERIMENT_RUN_COMPLETED for e in events
        )
        if any_success:
            events.append(
                AIScientistEvent(
                    event_type=AIScientistEventType.EXPERIMENT_COMPLETED,
                    idea_name=idea_name,
                    phase="experiment",
                    payload={"total_runs": len(run_dirs)},
                )
            )

        return events

    def parse_writeup(self, idea_dir: str) -> List[AIScientistEvent]:
        """Parse writeup artifacts from an idea directory."""
        events: List[AIScientistEvent] = []
        idea_name = os.path.basename(idea_dir)

        idea_name = self._strip_timestamp_prefix(idea_name)

        latex_dir = os.path.join(idea_dir, "latex")
        if os.path.isdir(latex_dir):
            tex_file = os.path.join(latex_dir, "template.tex")
            if os.path.isfile(tex_file):
                try:
                    with open(tex_file) as f:
                        content = f.read()
                    # Count citations
                    citations = len(re.findall(r"\\cite\{[^}]+\}", content))
                    # Detect sections
                    sections = re.findall(r"\\section\{([^}]+)\}", content)

                    for section in sections:
                        writeup = WriteupEvent(
                            section=section,
                            citation_count=citations,
                        )
                        events.append(
                            AIScientistEvent(
                                event_type=AIScientistEventType.WRITEUP_SECTION,
                                idea_name=idea_name,
                                phase="writeup",
                                payload=writeup.to_dict(),
                            )
                        )

                    if citations > 0:
                        events.append(
                            AIScientistEvent(
                                event_type=AIScientistEventType.CITATION_ADDED,
                                idea_name=idea_name,
                                phase="writeup",
                                payload={"citation_count": citations},
                            )
                        )
                except OSError:
                    pass

        # Check for compiled PDF
        pdf_files = [f for f in os.listdir(idea_dir) if f.endswith(".pdf")]
        if pdf_files:
            writeup = WriteupEvent(compiled=True, page_count=0)
            events.append(
                AIScientistEvent(
                    event_type=AIScientistEventType.WRITEUP_COMPILED,
                    idea_name=idea_name,
                    phase="writeup",
                    payload=writeup.to_dict(),
                )
            )
        else:
            # Check if latex dir exists but no PDF = compilation failure
            if os.path.isdir(latex_dir):
                writeup = WriteupEvent(
                    compiled=False,
                    compilation_error="No PDF output found",
                )
                events.append(
                    AIScientistEvent(
                        event_type=AIScientistEventType.WRITEUP_FAILED,
                        idea_name=idea_name,
                        phase="writeup",
                        payload=writeup.to_dict(),
                    )
                )

        return events

    def parse_review(self, review_path: str, idea_name: str = "") -> List[AIScientistEvent]:
        """Parse a review.txt file (JSON format with review scores)."""
        events: List[AIScientistEvent] = []

        try:
            with open(review_path) as f:
                content = f.read()

            review_data = json.loads(content)

            review_evt = ReviewEvent(
                overall_score=float(review_data.get("Overall", 0)),
                decision=review_data.get("Decision", ""),
                confidence=float(review_data.get("Confidence", 0)),
                soundness=float(review_data.get("Soundness", 0)),
                presentation=float(review_data.get("Presentation", 0)),
                contribution=float(review_data.get("Contribution", 0)),
                strengths=review_data.get("Strengths", []),
                weaknesses=review_data.get("Weaknesses", []),
            )

            events.append(
                AIScientistEvent(
                    event_type=AIScientistEventType.REVIEW_SUBMITTED,
                    idea_name=idea_name,
                    phase="review",
                    payload=review_evt.to_dict(),
                )
            )
        except (json.JSONDecodeError, OSError):
            events.append(
                AIScientistEvent(
                    event_type=AIScientistEventType.ERROR,
                    idea_name=idea_name,
                    phase="review",
                    payload={"error": f"Failed to parse review: {review_path}"},
                )
            )

        return events

    def parse_idea_directory(self, idea_dir: str) -> List[AIScientistEvent]:
        """Parse all stages from an idea output directory."""
        events: List[AIScientistEvent] = []

        # Experiment runs
        events.extend(self.parse_experiment_runs(idea_dir))

        # Writeup
        events.extend(self.parse_writeup(idea_dir))

        # Review
        review_path = os.path.join(idea_dir, "review.txt")
        idea_name = os.path.basename(idea_dir)
        idea_name = self._strip_timestamp_prefix(idea_name)

        if os.path.isfile(review_path):
            events.extend(self.parse_review(review_path, idea_name))

        return events

    def parse_results_directory(
        self,
        results_dir: str,
        ideas_data: Optional[List[Dict[str, Any]]] = None,
    ) -> List[AIScientistEvent]:
        """Parse an entire AI-Scientist results directory."""
        events: List[AIScientistEvent] = []

        if not os.path.isdir(results_dir):
            return events

        # Parse ideas.json if present
        ideas_path = os.path.join(results_dir, "ideas.json")
        if ideas_data is None and os.path.isfile(ideas_path):
            try:
                with open(ideas_path) as f:
                    ideas_data = json.load(f)
            except (json.JSONDecodeError, OSError):
                ideas_data = []

        # Parse each idea directory
        for entry in sorted(os.listdir(results_dir)):
            entry_path = os.path.join(results_dir, entry)
            if not os.path.isdir(entry_path) or entry.startswith("."):
                continue
            # Skip non-idea directories
            if entry in ("latex", "logs"):
                continue

            # Extract idea name
            idea_name = self._strip_timestamp_prefix(entry)

            # Add idea events from ideas.json
            if ideas_data:
                events.extend(self.parse_idea(ideas_data, idea_name))

            # Parse the idea directory
            events.extend(self.parse_idea_directory(entry_path))

        return events
