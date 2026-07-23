"""Task synthesis from execution traces (SciForge-style dependency graphs).

This module enables automatic generation of CompositeTask structures from
agent execution traces. It extracts subtask boundaries, infers dependencies,
and creates verifiable task graphs for replay testing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from swarm.bridges.awm.mcp_client import AWMEpisodeTrace, ToolCallRecord
from swarm.env.composite_tasks import (
    CapabilityType,
    CompositeTask,
    Subtask,
)

logger = logging.getLogger(__name__)


@dataclass
class TraceSegment:
    """A logical segment of an execution trace representing a subtask.

    Attributes:
        segment_id: Unique identifier for this segment
        start_idx: Starting index in the tool call sequence
        end_idx: Ending index (exclusive) in the tool call sequence
        tool_calls: Tool calls within this segment
        description: Generated description of this segment
        primary_tools: Most frequently used tools in this segment
    """

    segment_id: str
    start_idx: int
    end_idx: int
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    description: str = ""
    primary_tools: Set[str] = field(default_factory=set)

    @property
    def call_count(self) -> int:
        """Number of tool calls in this segment."""
        return len(self.tool_calls)

    @property
    def error_rate(self) -> float:
        """Fraction of calls that resulted in errors."""
        if not self.tool_calls:
            return 0.0
        errors = sum(1 for tc in self.tool_calls if tc.is_error_response)
        return errors / len(self.tool_calls)


class TraceSegmenter:
    """Partitions execution traces into logical subtask boundaries.

    Uses heuristics to identify natural breakpoints in tool usage patterns:
    - Tool clustering (similar tools used together)
    - Phase transitions (shift in tool types)
    - Execution pauses (gaps in call timing if available)
    """

    def __init__(
        self,
        min_calls_per_segment: int = 2,
        max_calls_per_segment: int = 10,
    ):
        """Initialize segmenter.

        Args:
            min_calls_per_segment: Minimum tool calls per segment
            max_calls_per_segment: Maximum tool calls per segment
        """
        self.min_calls_per_segment = min_calls_per_segment
        self.max_calls_per_segment = max_calls_per_segment

    def segment(self, trace: AWMEpisodeTrace) -> List[TraceSegment]:
        """Segment an episode trace into logical subtasks.

        Args:
            trace: The episode trace to segment

        Returns:
            List of trace segments representing subtasks
        """
        if not trace.tool_calls:
            return []

        segments: List[TraceSegment] = []
        current_start = 0
        current_tools: Set[str] = set()

        for idx, call in enumerate(trace.tool_calls):
            current_tools.add(call.tool_name)

            # Check if we should create a new segment
            should_split = False
            segment_size = idx - current_start + 1

            # Split if we've reached max size
            if segment_size >= self.max_calls_per_segment:
                should_split = True
            # Split if tool pattern changes significantly
            elif segment_size >= self.min_calls_per_segment:
                if self._is_phase_transition(trace.tool_calls, current_start, idx):
                    should_split = True

            if should_split or idx == len(trace.tool_calls) - 1:
                # Create segment
                end_idx = idx + 1

                segment = TraceSegment(
                    segment_id=f"{trace.episode_id}_seg_{len(segments)}",
                    start_idx=current_start,
                    end_idx=end_idx,
                    tool_calls=trace.tool_calls[current_start:end_idx],
                    primary_tools=current_tools.copy(),
                )
                segment.description = self._generate_description(segment)
                segments.append(segment)

                # Reset for next segment
                current_start = idx + 1
                current_tools = set()

        # Ensure we have at least one segment
        if not segments and trace.tool_calls:
            segment = TraceSegment(
                segment_id=f"{trace.episode_id}_seg_0",
                start_idx=0,
                end_idx=len(trace.tool_calls),
                tool_calls=trace.tool_calls,
                primary_tools={tc.tool_name for tc in trace.tool_calls},
            )
            segment.description = self._generate_description(segment)
            segments.append(segment)

        return segments

    def _is_phase_transition(
        self,
        calls: List[ToolCallRecord],
        start: int,
        current: int,
    ) -> bool:
        """Detect if there's a phase transition in tool usage.

        A phase transition occurs when the tool usage pattern changes
        significantly (e.g., from read operations to write operations).
        """
        if current - start < self.min_calls_per_segment:
            return False

        # Get tool names for previous window and current position
        window_size = min(3, current - start)
        prev_tools = {calls[i].tool_name for i in range(current - window_size, current)}
        current_tool = calls[current].tool_name

        # Transition if current tool is completely different from recent tools
        return current_tool not in prev_tools

    def _generate_description(self, segment: TraceSegment) -> str:
        """Generate a human-readable description for a segment."""
        if not segment.tool_calls:
            return "Empty segment"

        tool_names = list(segment.primary_tools)
        if len(tool_names) == 1:
            return f"Execute {tool_names[0]} operations"
        elif len(tool_names) <= 3:
            return f"Execute {', '.join(tool_names)} operations"
        else:
            return f"Execute multi-tool operations ({len(tool_names)} tools)"


class DependencyInferencer:
    """Infers task dependencies from execution patterns.

    Analyzes trace segments to determine:
    - Precedence (execution order)
    - Data flow (outputs consumed as inputs)
    - Resource dependencies (shared state access)
    """

    def infer_dependencies(
        self,
        segments: List[TraceSegment],
    ) -> Dict[str, Set[str]]:
        """Infer dependencies between trace segments.

        Args:
            segments: List of trace segments to analyze

        Returns:
            Dict mapping segment_id -> set of prerequisite segment_ids
        """
        dependencies: Dict[str, Set[str]] = {seg.segment_id: set() for seg in segments}

        # For now, use simple precedence: each segment depends on previous
        # This creates a linear chain, which is safe but conservative
        for i in range(1, len(segments)):
            dependencies[segments[i].segment_id].add(segments[i - 1].segment_id)

        return dependencies


class TaskSynthesizer:
    """Synthesizes CompositeTask structures from trace segments.

    Combines segmentation and dependency inference to create verifiable
    task graphs from agent execution traces.
    """

    def __init__(
        self,
        segmenter: Optional[TraceSegmenter] = None,
        inferencer: Optional[DependencyInferencer] = None,
    ):
        """Initialize synthesizer.

        Args:
            segmenter: Trace segmenter (creates default if None)
            inferencer: Dependency inferencer (creates default if None)
        """
        self.segmenter = segmenter or TraceSegmenter()
        self.inferencer = inferencer or DependencyInferencer()

    def synthesize(
        self,
        trace: AWMEpisodeTrace,
        task_name: Optional[str] = None,
        bounty: float = 20.0,
    ) -> CompositeTask:
        """Synthesize a CompositeTask from an execution trace.

        Args:
            trace: The episode trace to synthesize from
            task_name: Optional name for the task
            bounty: Total bounty for the task

        Returns:
            A CompositeTask with subtasks and dependencies
        """
        # Step 1: Segment the trace
        segments = self.segmenter.segment(trace)

        if not segments:
            # Return minimal task if no segments
            return CompositeTask(
                name=task_name or f"Synthesized: {trace.task_description[:50]}",
                description=trace.task_description,
                goal="Complete synthesized task from trace",
                min_agents=1,
                max_agents=1,
                total_bounty=bounty,
            )

        # Step 2: Infer dependencies
        dependencies = self.inferencer.infer_dependencies(segments)

        # Step 3: Create CompositeTask
        task = CompositeTask(
            name=task_name or f"Synthesized: {trace.task_description[:50]}",
            description=trace.task_description,
            goal=f"Reproduce task from episode {trace.episode_id[:8]}",
            min_agents=1,
            max_agents=min(len(segments), 5),
            total_bounty=bounty,
            completion_bonus=bounty * 0.2,
        )

        # Step 4: Create Subtask for each segment
        segment_id_to_subtask_id: Dict[str, str] = {}
        bounty_per_segment = 1.0 / len(segments) if segments else 1.0

        for segment in segments:
            # Map segment tools to capabilities
            capabilities = self._infer_capabilities(segment)

            subtask = Subtask(
                name=segment.description,
                description=f"Segment {segment.segment_id} with tools: {', '.join(segment.primary_tools)}",
                required_capabilities=capabilities,
                bounty_share=bounty_per_segment,
                estimated_steps=segment.call_count,
            )

            task.add_subtask(subtask)
            segment_id_to_subtask_id[segment.segment_id] = subtask.subtask_id

        # Step 5: Wire up dependencies
        for i, segment in enumerate(segments):
            subtask = task.subtasks[i]
            seg_deps = dependencies.get(segment.segment_id, set())

            for dep_seg_id in seg_deps:
                if dep_seg_id in segment_id_to_subtask_id:
                    subtask.dependencies.add(segment_id_to_subtask_id[dep_seg_id])

        # Step 6: Infer required capabilities for the composite task
        all_capabilities: Set[CapabilityType] = set()
        for subtask in task.subtasks:
            all_capabilities.update(subtask.required_capabilities)
        task.required_capabilities = all_capabilities

        logger.info(
            f"Synthesized task '{task.name}' with {len(segments)} subtasks "
            f"from trace {trace.episode_id}"
        )

        return task

    def _infer_capabilities(self, segment: TraceSegment) -> Set[CapabilityType]:
        """Infer required capabilities from a trace segment.

        Maps tool usage patterns to capability types.
        """
        capabilities: Set[CapabilityType] = set()

        # Default to execution capability
        capabilities.add(CapabilityType.EXECUTION)

        # Add analysis if multiple diverse tools used
        if len(segment.primary_tools) >= 3:
            capabilities.add(CapabilityType.ANALYSIS)

        # Add verification if error rate is low (suggests careful execution)
        if segment.error_rate < 0.1 and segment.call_count >= 3:
            capabilities.add(CapabilityType.VERIFICATION)

        return capabilities


@dataclass
class SynthesisMetrics:
    """Metrics for task synthesis from traces.

    Tracks quality and characteristics of synthesized tasks.
    """

    total_traces_processed: int = 0
    total_tasks_synthesized: int = 0
    total_segments_extracted: int = 0
    avg_segments_per_task: float = 0.0
    avg_dependencies_per_segment: float = 0.0
    synthesis_failures: int = 0

    def update(
        self,
        segments_count: int,
        avg_deps: float,
        success: bool = True,
    ) -> None:
        """Update metrics after processing a trace."""
        self.total_traces_processed += 1
        if success:
            self.total_tasks_synthesized += 1
            self.total_segments_extracted += segments_count

            # Running average
            n = self.total_tasks_synthesized
            self.avg_segments_per_task = (
                (self.avg_segments_per_task * (n - 1) + segments_count) / n
            )
            self.avg_dependencies_per_segment = (
                (self.avg_dependencies_per_segment * (n - 1) + avg_deps) / n
            )
        else:
            self.synthesis_failures += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_traces_processed": self.total_traces_processed,
            "total_tasks_synthesized": self.total_tasks_synthesized,
            "total_segments_extracted": self.total_segments_extracted,
            "avg_segments_per_task": self.avg_segments_per_task,
            "avg_dependencies_per_segment": self.avg_dependencies_per_segment,
            "synthesis_failures": self.synthesis_failures,
        }
