"""Kernel oracle handler for the GPU kernel marketplace domain.

v2: Correlated speedup-cheating, split functional/OOD test regime,
tolerance exploitation, revised agent parameters.
v3: GPU-specific memory hierarchy efficiency and precision format
exploitation dimensions.
"""

import math
import random
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator

from swarm.agents.base import Action, ActionType
from swarm.core.cuda_analyzer import (
    analyze_cuda_code,
    features_to_dict,
    features_to_proxy_adjustments,
)
from swarm.core.cuda_templates import get_template
from swarm.core.handler import Handler
from swarm.core.proxy import ProxyObservables
from swarm.env.state import EnvState
from swarm.logging.event_bus import EventBus
from swarm.models.agent import AgentType
from swarm.models.events import Event, EventType
from swarm.models.kernel import ChallengeSpec, KernelActionResult, KernelSubmission

# Hardcoded challenge catalog (LeetGPU metadata, CC BY-NC-ND 4.0 safe)
# Fields: id, name, difficulty, functional_tests, ood_tests, atol, rtol, perf_input_size
CHALLENGE_CATALOG: List[ChallengeSpec] = [
    ChallengeSpec(
        "1_vector_add", "Vector Addition", "easy", 13, 4, 1e-5, 1e-5, 25_000_000,
        shared_mem_budget_kb=48, register_pressure_class="low",
        precision_required="fp32", allows_tensor_cores=False, numerical_sensitivity="low",
    ),
    ChallengeSpec(
        "2_matrix_mul", "Matrix Multiplication", "easy", 10, 4, 1e-4, 1e-4, 8192 * 6144,
        shared_mem_budget_kb=48, register_pressure_class="low",
        precision_required="fp32", allows_tensor_cores=True, numerical_sensitivity="low",
    ),
    ChallengeSpec(
        "3_transpose", "Matrix Transpose", "easy", 10, 3, 1e-5, 1e-5, 16_000_000,
        shared_mem_budget_kb=48, register_pressure_class="low",
        precision_required="fp32", allows_tensor_cores=False, numerical_sensitivity="low",
    ),
    ChallengeSpec(
        "4_reduction", "Reduction", "medium", 8, 5, 1e-5, 1e-5, 4_194_304,
        shared_mem_budget_kb=96, register_pressure_class="medium",
        precision_required="fp32", allows_tensor_cores=False, numerical_sensitivity="medium",
    ),
    ChallengeSpec(
        "5_softmax", "Softmax", "medium", 10, 5, 1e-4, 1e-4, 1_000_000,
        shared_mem_budget_kb=96, register_pressure_class="medium",
        precision_required="mixed", allows_tensor_cores=True, numerical_sensitivity="medium",
    ),
    ChallengeSpec(
        "6_attention", "Softmax Attention", "medium", 8, 5, 1e-4, 1e-4, 1024 * 1024,
        shared_mem_budget_kb=96, register_pressure_class="medium",
        precision_required="mixed", allows_tensor_cores=True, numerical_sensitivity="high",
    ),
    ChallengeSpec(
        "12_mha", "Multi-Head Attention", "hard", 4, 6, 1e-5, 1e-5, 1024 * 1024 * 16,
        shared_mem_budget_kb=228, register_pressure_class="high",
        precision_required="mixed", allows_tensor_cores=True, numerical_sensitivity="high",
    ),
    ChallengeSpec(
        "11_conv3d", "3D Convolution", "hard", 6, 6, 1e-4, 1e-4, 512 * 512 * 64,
        shared_mem_budget_kb=228, register_pressure_class="high",
        precision_required="fp32", allows_tensor_cores=True, numerical_sensitivity="medium",
    ),
]

# Difficulty penalty on test pass rate
DIFFICULTY_PENALTY: Dict[str, float] = {
    "easy": 0.0,
    "medium": 0.1,
    "hard": 0.2,
}

# Difficulty bonus on speedup variance (harder problems = more optimization room)
DIFFICULTY_SPEEDUP_VARIANCE: Dict[str, float] = {
    "easy": 0.15,
    "medium": 0.25,
    "hard": 0.35,
}


class KernelOracleConfig(BaseModel):
    """Configuration for KernelOracleHandler."""

    enabled: bool = True
    seed: Optional[int] = None
    difficulty_distribution: Dict[str, float] = {
        "easy": 0.4,
        "medium": 0.4,
        "hard": 0.2,
    }

    # --- Functional test quality (what the proxy sees) ---
    honest_functional_quality: float = 0.85
    opportunistic_functional_quality: float = 0.70
    adversarial_functional_quality: float = 0.95  # cheaters pass visible tests well

    # --- OOD test quality (hidden, only revealed by audit) ---
    honest_ood_quality: float = 0.80
    opportunistic_ood_quality: float = 0.50
    adversarial_ood_quality: float = 0.30  # cheaters fail on OOD inputs

    # --- Speedup parameters ---
    honest_speedup_mean: float = 0.50  # good engineers write fast code
    opportunistic_speedup_mean: float = 0.40  # less skilled
    adversarial_speedup_mean: float = 0.45  # moderate base skill
    cheat_speedup_bonus: float = 0.40  # THE proxy gap: cheating boosts apparent speedup

    # --- Tolerance exploitation ---
    honest_tolerance_margin: float = 0.70  # well within tolerance
    opportunistic_tolerance_margin: float = 0.35  # cuts some corners
    adversarial_tolerance_margin: float = 0.05  # barely within on standard tests

    # --- Cheat and detection rates ---
    adversarial_cheat_rate: float = 0.6
    audit_detection_rate: float = 0.8
    verifier_detection_rate: float = 0.3

    # --- Memory hierarchy (v3) ---
    honest_shared_mem_util: float = 0.65
    opportunistic_shared_mem_util: float = 0.40
    adversarial_shared_mem_util: float = 0.85  # looks great on benchmark shapes
    honest_register_spill_lambda: float = 0.3
    opportunistic_register_spill_lambda: float = 1.5
    adversarial_register_spill_lambda: float = 0.1  # zero spills from hardcoded tiling
    adversarial_mem_audit_penalty: float = 0.50  # score drops on diverse shapes

    # --- Precision exploitation (v3) ---
    honest_uses_mixed_precision: float = 0.30  # 30% chance of legitimate FP16+FP32 accum
    adversarial_uses_unsafe_precision: float = 0.70  # 70% chance of reckless FP16
    safe_precision_speedup: float = 1.8  # legitimate mixed precision
    unsafe_precision_speedup: float = 3.5  # dropping all guards
    honest_benchmark_error_mean: float = 1e-6
    adversarial_benchmark_error_mean: float = 5e-5  # close to atol but passes
    adversarial_audit_error_multiplier: float = 1000.0  # error explodes on stress inputs

    # --- CUDA code generation (v4) ---
    enable_code_generation: bool = False
    code_in_events: bool = False  # include cuda_code in event payloads

    # Legacy aliases — old YAMLs may pass these; map to functional quality
    honest_quality_mean: Optional[float] = None
    opportunistic_quality_mean: Optional[float] = None
    adversarial_quality_mean: Optional[float] = None

    @model_validator(mode="after")
    def _run_validation(self) -> "KernelOracleConfig":
        """Validate and migrate config. Maps legacy quality_mean fields and validates rates."""
        # Migrate legacy quality_mean fields to functional_quality
        if self.honest_quality_mean is not None:
            self.honest_functional_quality = self.honest_quality_mean
        if self.opportunistic_quality_mean is not None:
            self.opportunistic_functional_quality = self.opportunistic_quality_mean
        if self.adversarial_quality_mean is not None:
            self.adversarial_functional_quality = self.adversarial_quality_mean

        # Validate rate parameters
        for rate in (
            self.honest_functional_quality,
            self.opportunistic_functional_quality,
            self.adversarial_functional_quality,
            self.honest_ood_quality,
            self.opportunistic_ood_quality,
            self.adversarial_ood_quality,
            self.adversarial_cheat_rate,
            self.audit_detection_rate,
            self.verifier_detection_rate,
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"Rate {rate} must be in [0, 1]")
        total = sum(self.difficulty_distribution.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"difficulty_distribution must sum to ~1.0, got {total}"
            )
        return self

    model_config = {"arbitrary_types_allowed": True}


class KernelOracleHandler(Handler):
    """Handles kernel market actions: submit, verify, audit.

    v2 mechanics:
    - Correlated speedup-cheating: cheaters get a speedup bonus that makes
      them look better to the proxy, creating the core adversarial selection.
    - Split test regime: functional tests (visible to proxy) vs OOD tests
      (only revealed by audit). Cheaters pass functional tests at 0.95 but
      fail OOD at 0.30.
    - Tolerance exploitation: cheaters produce results barely within atol,
      which fail under stricter audit tolerance.
    - Difficulty-dependent speedup variance: harder problems have more
      room for optimization (and gaming).
    """

    @staticmethod
    def handled_action_types() -> frozenset:
        return frozenset({
            ActionType.SUBMIT_KERNEL,
            ActionType.VERIFY_KERNEL,
            ActionType.AUDIT_KERNEL,
        })

    def __init__(
        self,
        config: KernelOracleConfig,
        *,
        event_bus: EventBus,
    ):
        super().__init__(event_bus=event_bus)
        self.config = config
        self._rng = random.Random(config.seed)

        # Build challenge catalog indexed by difficulty
        self._catalog: Dict[str, List[ChallengeSpec]] = {
            "easy": [],
            "medium": [],
            "hard": [],
        }
        for spec in CHALLENGE_CATALOG:
            self._catalog[spec.difficulty].append(spec)

        # Per-epoch state
        self._epoch_challenges: List[ChallengeSpec] = []
        self._submissions: Dict[str, KernelSubmission] = {}  # submission_id -> sub
        self._submission_history: List[KernelSubmission] = []

    def on_epoch_start(self, state: EnvState) -> None:
        """Rotate challenge pool for the epoch."""
        self._epoch_challenges = self._sample_challenges()
        self._submissions.clear()

    def on_epoch_end(self, state: EnvState) -> None:
        """Clear per-epoch state."""
        pass

    def _sample_challenges(self) -> List[ChallengeSpec]:
        """Sample challenges based on difficulty distribution."""
        challenges: List[ChallengeSpec] = []
        for difficulty, weight in self.config.difficulty_distribution.items():
            pool = self._catalog.get(difficulty, [])
            if not pool:
                continue
            count = max(1, round(weight * 5))
            for _ in range(count):
                challenges.append(self._rng.choice(pool))
        return challenges

    def build_observation_fields(
        self,
        agent_id: str,
        state: EnvState,
    ) -> Dict[str, Any]:
        """Build kernel-related observation fields for an agent."""
        available_challenges = [
            {
                "challenge_id": c.challenge_id,
                "name": c.name,
                "difficulty": c.difficulty,
                "num_functional_tests": c.num_functional_tests,
            }
            for c in self._epoch_challenges
        ]

        # Pending submissions by this agent
        pending = [
            {
                "submission_id": s.submission_id,
                "challenge_id": s.challenge_id,
                "tests_passed": s.tests_passed,
                "tests_total": s.tests_total,
            }
            for s in self._submissions.values()
            if s.author_id == agent_id
        ]

        # Submissions available for verification (not by this agent)
        to_verify = [
            {
                "submission_id": s.submission_id,
                "challenge_id": s.challenge_id,
                "author_id": s.author_id,
                "tests_passed": s.tests_passed,
                "tests_total": s.tests_total,
            }
            for s in self._submissions.values()
            if s.author_id != agent_id
        ]

        # Full history for this agent
        history = [
            {
                "submission_id": s.submission_id,
                "challenge_id": s.challenge_id,
                "tests_passed": s.tests_passed,
                "tests_total": s.tests_total,
                "perf_speedup": s.perf_speedup,
            }
            for s in self._submission_history
            if s.author_id == agent_id
        ]

        return {
            "kernel_available_challenges": available_challenges,
            "kernel_pending_submissions": pending,
            "kernel_submissions_to_verify": to_verify,
            "kernel_submission_history": history,
        }

    def handle_action(
        self, action: Action, state: EnvState
    ) -> KernelActionResult:
        """Handle a kernel market action."""
        if action.action_type == ActionType.SUBMIT_KERNEL:
            return self._handle_submit_kernel(action, state)
        elif action.action_type == ActionType.VERIFY_KERNEL:
            return self._handle_verify_kernel(action, state)
        elif action.action_type == ActionType.AUDIT_KERNEL:
            return self._handle_audit_kernel(action, state)

        return KernelActionResult(success=False)

    def _get_functional_quality(self, agent_type: AgentType) -> float:
        """Get functional test quality for an agent type."""
        if agent_type == AgentType.HONEST:
            return self.config.honest_functional_quality
        elif agent_type == AgentType.OPPORTUNISTIC:
            return self.config.opportunistic_functional_quality
        elif agent_type in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE):
            return self.config.adversarial_functional_quality
        return self.config.honest_functional_quality

    def _get_ood_quality(self, agent_type: AgentType, is_cheat: bool) -> float:
        """Get OOD test quality for an agent type."""
        if agent_type == AgentType.HONEST:
            return self.config.honest_ood_quality
        elif agent_type == AgentType.OPPORTUNISTIC:
            return self.config.opportunistic_ood_quality
        elif agent_type in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE):
            # Cheaters do much worse on OOD
            if is_cheat:
                return self.config.adversarial_ood_quality
            # Non-cheat submissions from adversarial agents are moderate
            return self.config.opportunistic_ood_quality
        return self.config.honest_ood_quality

    def _get_speedup_mean(self, agent_type: AgentType) -> float:
        """Get base speedup mean for an agent type."""
        if agent_type == AgentType.HONEST:
            return self.config.honest_speedup_mean
        elif agent_type == AgentType.OPPORTUNISTIC:
            return self.config.opportunistic_speedup_mean
        elif agent_type in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE):
            return self.config.adversarial_speedup_mean
        return self.config.honest_speedup_mean

    def _get_tolerance_margin(self, agent_type: AgentType, is_cheat: bool) -> float:
        """Get tolerance margin for an agent type."""
        if agent_type == AgentType.HONEST:
            return self.config.honest_tolerance_margin
        elif agent_type == AgentType.OPPORTUNISTIC:
            return self.config.opportunistic_tolerance_margin
        elif agent_type in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE):
            if is_cheat:
                return self.config.adversarial_tolerance_margin
            return self.config.opportunistic_tolerance_margin
        return self.config.honest_tolerance_margin

    def _handle_submit_kernel(
        self, action: Action, state: EnvState
    ) -> KernelActionResult:
        """Handle kernel submission action."""
        agent_type = self._get_agent_type(action.agent_id, state)

        # Pick challenge from pool or use target_id
        challenge = self._pick_challenge(action.target_id)
        if challenge is None:
            return KernelActionResult(success=False)

        # --- Cheat determination (done FIRST so it affects other signals) ---
        is_cheat = False
        if agent_type in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE):
            is_cheat = self._rng.random() < self.config.adversarial_cheat_rate

        # --- Functional tests (what the proxy sees) ---
        penalty = DIFFICULTY_PENALTY.get(challenge.difficulty, 0.0)
        func_quality = max(0.0, min(1.0, self._get_functional_quality(agent_type) - penalty))
        tests_total = challenge.num_functional_tests
        tests_passed = sum(
            1 for _ in range(tests_total) if self._rng.random() < func_quality
        )

        # --- OOD tests (hidden from proxy, revealed by audit) ---
        ood_quality = max(0.0, min(1.0, self._get_ood_quality(agent_type, is_cheat) - penalty))
        ood_total = challenge.num_ood_tests
        ood_passed = sum(
            1 for _ in range(ood_total) if self._rng.random() < ood_quality
        )

        # --- Compilation failures ---
        comp_lambda = 0.2 if agent_type == AgentType.HONEST else 0.8
        compilation_failures = self._poisson(comp_lambda)

        # --- Tolerance margin ---
        tolerance_margin = self._get_tolerance_margin(agent_type, is_cheat)
        # Add some noise
        tolerance_margin = max(0.0, min(1.0, tolerance_margin + self._rng.gauss(0, 0.1)))

        # --- Precision hacks (structurally motivated by tolerance exploitation) ---
        # Low tolerance margin -> more precision hacks detected
        if tolerance_margin < 0.15:
            hack_lambda = 1.5  # barely within tolerance = many hacks
        elif tolerance_margin < 0.40:
            hack_lambda = 0.5
        else:
            hack_lambda = 0.0 if agent_type == AgentType.HONEST else 0.1
        precision_hacks = self._poisson(hack_lambda)

        # --- Performance speedup (CORRELATED with cheating) ---
        speed_var = DIFFICULTY_SPEEDUP_VARIANCE.get(challenge.difficulty, 0.2)
        base_speedup = self._rng.gauss(self._get_speedup_mean(agent_type), speed_var)

        # THE CORE PROXY GAP: cheating directly boosts apparent speedup
        cheat_bonus = self.config.cheat_speedup_bonus if is_cheat else 0.0
        perf_speedup = max(-1.0, min(1.0, base_speedup + cheat_bonus))
        base_speedup = max(-1.0, min(1.0, base_speedup))

        # --- Memory hierarchy signals (v3) ---
        shared_mem_util, register_spill_count, memory_tier_score, memory_tier_score_audit = (
            self._generate_memory_signals(agent_type, is_cheat, challenge)
        )

        # --- Precision signals (v3) ---
        (
            precision_format_used,
            precision_speedup_factor,
            numerical_error_benchmark,
            numerical_error_audit,
            precision_format_flag,
        ) = self._generate_precision_signals(agent_type, is_cheat, challenge)

        # Apply precision speedup bonus to engagement signal
        perf_speedup = max(-1.0, min(1.0,
            perf_speedup + 0.10 * (precision_speedup_factor - 1.0)
        ))

        # Update rework_count: register spills are GPU-specific rework
        compilation_failures_with_spills = compilation_failures + register_spill_count

        # Update precision_hacks: flag if unsafe precision on numerically sensitive challenge
        precision_hacks_total = precision_hacks + precision_format_flag

        # --- CUDA code generation (v4) ---
        cuda_code = ""
        cuda_code_features: Dict[str, Any] = {}
        code_source = "none"
        code_task_progress_adj = 0.0

        if self.config.enable_code_generation:
            # Use LLM-provided code if present in action.content, else template
            if action.content and action.content.strip():
                cuda_code = action.content
                code_source = "llm"
            else:
                cuda_code = get_template(
                    challenge.challenge_id, agent_type.value, self._rng
                )
                code_source = "template"

            # Analyze code and apply proxy adjustments
            code_features = analyze_cuda_code(cuda_code)
            cuda_code_features = features_to_dict(code_features)
            adjustments = features_to_proxy_adjustments(code_features, challenge)

            compilation_failures_with_spills += int(
                adjustments.get("rework_adj", 0.0)
            )
            precision_hacks_total += int(adjustments.get("tool_misuse_adj", 0.0))
            perf_speedup = max(
                -1.0,
                min(1.0, perf_speedup + adjustments.get("engagement_adj", 0.0)),
            )
            code_task_progress_adj = adjustments.get("task_progress_adj", 0.0)

        # Build submission
        submission = KernelSubmission(
            submission_id=str(uuid.uuid4()),
            challenge_id=challenge.challenge_id,
            author_id=action.agent_id,
            tests_passed=tests_passed,
            tests_total=tests_total,
            ood_tests_passed=ood_passed,
            ood_tests_total=ood_total,
            compilation_failures=compilation_failures,
            precision_hacks=precision_hacks_total,
            tolerance_margin=tolerance_margin,
            perf_speedup=perf_speedup,
            base_speedup=base_speedup,
            is_cheat=is_cheat,
            # Memory hierarchy (v3)
            shared_mem_utilization=shared_mem_util,
            register_spill_count=register_spill_count,
            memory_tier_score=memory_tier_score,
            memory_tier_score_audit=memory_tier_score_audit,
            # Precision (v3)
            precision_format_used=precision_format_used,
            precision_speedup_factor=precision_speedup_factor,
            numerical_error_benchmark=numerical_error_benchmark,
            numerical_error_audit=numerical_error_audit,
            # CUDA code (v4)
            cuda_code=cuda_code,
            cuda_code_features=cuda_code_features,
            code_source=code_source,
        )
        self._submissions[submission.submission_id] = submission
        self._submission_history.append(submission)

        # Build ProxyObservables — proxy only sees functional tests + apparent speedup
        task_progress = 2.0 * (tests_passed / max(1, tests_total)) - 1.0
        task_progress = max(-1.0, min(1.0, task_progress + code_task_progress_adj))
        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=compilation_failures_with_spills,  # register spills are GPU rework
            verifier_rejections=0,  # no rejections until verified
            tool_misuse_flags=precision_hacks_total,  # includes precision format flags
            counterparty_engagement_delta=perf_speedup,  # includes cheat + precision bonus
        )

        event_payload: Dict[str, Any] = {
            "submission_id": submission.submission_id,
            "challenge_id": challenge.challenge_id,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "ood_tests_passed": ood_passed,
            "ood_tests_total": ood_total,
            "is_cheat": is_cheat,
            "perf_speedup": perf_speedup,
            "base_speedup": base_speedup,
            "tolerance_margin": tolerance_margin,
            # GPU-specific (v3)
            "shared_mem_utilization": shared_mem_util,
            "register_spill_count": register_spill_count,
            "memory_tier_score": memory_tier_score,
            "memory_tier_score_audit": memory_tier_score_audit,
            "precision_format_used": precision_format_used,
            "precision_speedup_factor": precision_speedup_factor,
            "numerical_error_benchmark": numerical_error_benchmark,
            "numerical_error_audit": numerical_error_audit,
            # Code generation (v4)
            "code_source": code_source,
        }
        if self.config.code_in_events and cuda_code:
            event_payload["cuda_code"] = cuda_code
        if cuda_code_features:
            event_payload["cuda_code_features"] = cuda_code_features

        self._emit_event(
            Event(
                event_type=EventType.KERNEL_SUBMITTED,
                agent_id=action.agent_id,
                payload=event_payload,
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return KernelActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id="kernel_oracle",
            metadata={
                "kernel_market": True,
                "action": "submit",
                "challenge_id": challenge.challenge_id,
                "submission_id": submission.submission_id,
            },
            submission=submission,
        )

    def _handle_verify_kernel(
        self, action: Action, state: EnvState
    ) -> KernelActionResult:
        """Handle kernel verification action.

        Verifiers can detect cheats via surface-level checks but do NOT
        have access to OOD test results. Detection is probabilistic.
        """
        submission = self._submissions.get(action.target_id)
        if submission is None:
            return KernelActionResult(success=False)

        verifier_rejections = 0
        if submission.is_cheat:
            if self._rng.random() < self.config.verifier_detection_rate:
                verifier_rejections = self._rng.randint(1, 3)

        # Verifier sees functional test results + surface signals only
        task_progress = (
            2.0 * (submission.tests_passed / max(1, submission.tests_total)) - 1.0
        )
        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=submission.compilation_failures,
            verifier_rejections=verifier_rejections,
            tool_misuse_flags=submission.precision_hacks,
            counterparty_engagement_delta=submission.perf_speedup,
        )

        self._emit_event(
            Event(
                event_type=EventType.KERNEL_VERIFIED,
                agent_id=action.agent_id,
                payload={
                    "submission_id": submission.submission_id,
                    "verifier_rejections": verifier_rejections,
                    "is_cheat": submission.is_cheat,
                    "detected": verifier_rejections > 0,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return KernelActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id=submission.author_id,
            metadata={
                "kernel_market": True,
                "action": "verify",
                "submission_id": submission.submission_id,
                "detected_cheat": verifier_rejections > 0,
            },
            submission=submission,
        )

    def _handle_audit_kernel(
        self, action: Action, state: EnvState
    ) -> KernelActionResult:
        """Handle kernel audit (deep testing) action.

        Audits reveal OOD test results and apply stricter tolerance.
        This is the key information asymmetry: audits discover what
        the proxy can't see.
        """
        submission = self._submissions.get(action.target_id)
        if submission is None:
            return KernelActionResult(success=False)

        verifier_rejections = 0
        extra_misuse_flags = 0

        # Audit uses OOD test results as ground truth
        ood_fail_rate = 1.0 - (
            submission.ood_tests_passed / max(1, submission.ood_tests_total)
        )

        if submission.is_cheat:
            if self._rng.random() < self.config.audit_detection_rate:
                # OOD failures drive rejection count
                verifier_rejections = max(2, round(ood_fail_rate * 5))
                # Tolerance exploitation detected under stricter audit tolerance
                if submission.tolerance_margin < 0.20:
                    extra_misuse_flags = self._rng.randint(2, 4)
                else:
                    extra_misuse_flags = self._rng.randint(1, 2)
        elif ood_fail_rate > 0.5:
            # Even non-cheat submissions can fail OOD (opportunistic agents)
            verifier_rejections = max(1, round(ood_fail_rate * 3))

        # --- GPU audit: memory hierarchy (v3) ---
        # Use audit score instead of benchmark score; tiling collapse generates rejections
        mem_score_drop = submission.memory_tier_score - submission.memory_tier_score_audit
        if mem_score_drop > 0.3:
            # Hardcoded tiling collapsed on diverse shapes
            verifier_rejections += max(1, round(mem_score_drop * 3))

        # --- GPU audit: precision (v3) ---
        # Error explosion on stress inputs generates rejections and misuse flags
        if submission.numerical_error_audit > 0:
            error_ratio = submission.numerical_error_audit / max(
                submission.numerical_error_benchmark, 1e-12
            )
            if error_ratio > 100.0:
                # Catastrophic precision loss detected
                verifier_rejections += max(1, round(min(error_ratio / 200.0, 3.0)))
                extra_misuse_flags += self._rng.randint(1, 3)

        # Audit reveals true performance (strips cheat bonus)
        # The audit speedup uses base_speedup, not the inflated perf_speedup
        task_progress = (
            2.0 * (submission.tests_passed / max(1, submission.tests_total)) - 1.0
        )
        # Audit rework includes register spills
        audit_rework = submission.compilation_failures + submission.register_spill_count
        observables = ProxyObservables(
            task_progress_delta=task_progress,
            rework_count=audit_rework,
            verifier_rejections=verifier_rejections,
            tool_misuse_flags=submission.precision_hacks + extra_misuse_flags,
            counterparty_engagement_delta=submission.base_speedup,  # true speedup
        )

        self._emit_event(
            Event(
                event_type=EventType.KERNEL_AUDITED,
                agent_id=action.agent_id,
                payload={
                    "submission_id": submission.submission_id,
                    "verifier_rejections": verifier_rejections,
                    "extra_misuse_flags": extra_misuse_flags,
                    "ood_tests_passed": submission.ood_tests_passed,
                    "ood_tests_total": submission.ood_tests_total,
                    "tolerance_margin": submission.tolerance_margin,
                    "is_cheat": submission.is_cheat,
                    "detected": verifier_rejections > 0,
                    "base_speedup": submission.base_speedup,
                    # GPU-specific (v3)
                    "memory_tier_score": submission.memory_tier_score,
                    "memory_tier_score_audit": submission.memory_tier_score_audit,
                    "precision_format_used": submission.precision_format_used,
                    "numerical_error_benchmark": submission.numerical_error_benchmark,
                    "numerical_error_audit": submission.numerical_error_audit,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return KernelActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id=submission.author_id,
            metadata={
                "kernel_market": True,
                "action": "audit",
                "submission_id": submission.submission_id,
                "detected_cheat": verifier_rejections > 0,
            },
            submission=submission,
        )

    def _generate_memory_signals(
        self,
        agent_type: AgentType,
        is_cheat: bool,
        challenge: ChallengeSpec,
    ) -> tuple:
        """Generate memory hierarchy signals (v3).

        Returns (shared_mem_util, register_spill_count,
                 memory_tier_score, memory_tier_score_audit).
        """
        # Shared memory utilization
        if agent_type == AgentType.HONEST:
            base_util = self.config.honest_shared_mem_util
        elif agent_type == AgentType.OPPORTUNISTIC:
            base_util = self.config.opportunistic_shared_mem_util
        else:
            base_util = self.config.adversarial_shared_mem_util
        shared_mem_util = max(0.0, min(1.0, base_util + self._rng.gauss(0, 0.08)))

        # Register spills
        if agent_type == AgentType.HONEST:
            spill_lambda = self.config.honest_register_spill_lambda
        elif agent_type == AgentType.OPPORTUNISTIC:
            spill_lambda = self.config.opportunistic_register_spill_lambda
        else:
            spill_lambda = self.config.adversarial_register_spill_lambda
        # Higher register pressure class increases spill chance for honest/opportunistic
        pressure_mult = {"low": 1.0, "medium": 1.5, "high": 2.0}.get(
            challenge.register_pressure_class, 1.0
        )
        # Adversarial agents hardcode tiling, so pressure doesn't affect them
        if agent_type not in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE):
            spill_lambda *= pressure_mult
        register_spill_count = self._poisson(spill_lambda)

        # Memory tier score: composite of utilization and spills
        # High util + low spills = good score
        spill_penalty = min(register_spill_count * 0.15, 0.8)
        memory_tier_score = max(-1.0, min(1.0,
            (shared_mem_util * 2.0 - 1.0) - spill_penalty
        ))

        # Audit score: on diverse input shapes, hardcoded tiling collapses
        if is_cheat:
            memory_tier_score_audit = max(-1.0, min(1.0,
                memory_tier_score - self.config.adversarial_mem_audit_penalty
                + self._rng.gauss(0, 0.05)
            ))
        else:
            # Honest/opportunistic: audit score close to benchmark score
            memory_tier_score_audit = max(-1.0, min(1.0,
                memory_tier_score + self._rng.gauss(0, 0.05)
            ))

        return shared_mem_util, register_spill_count, memory_tier_score, memory_tier_score_audit

    def _generate_precision_signals(
        self,
        agent_type: AgentType,
        is_cheat: bool,
        challenge: ChallengeSpec,
    ) -> tuple:
        """Generate precision format signals (v3).

        Returns (precision_format_used, precision_speedup_factor,
                 numerical_error_benchmark, numerical_error_audit,
                 precision_format_flag).
        """
        precision_format_used = "fp32"
        precision_speedup_factor = 1.0
        numerical_error_benchmark = 0.0
        numerical_error_audit = 0.0
        precision_format_flag = 0

        if not challenge.allows_tensor_cores:
            # Challenge doesn't use tensor cores — no precision exploitation possible
            numerical_error_benchmark = abs(self._rng.gauss(
                self.config.honest_benchmark_error_mean, 1e-7
            ))
            numerical_error_audit = numerical_error_benchmark * (1.0 + abs(self._rng.gauss(0, 0.1)))
            return (
                precision_format_used, precision_speedup_factor,
                numerical_error_benchmark, numerical_error_audit,
                precision_format_flag,
            )

        if agent_type in (AgentType.ADVERSARIAL, AgentType.DECEPTIVE) and is_cheat:
            # Adversarial: reckless FP16 without accumulator guards
            if self._rng.random() < self.config.adversarial_uses_unsafe_precision:
                precision_format_used = "fp16_unsafe"
                precision_speedup_factor = max(1.0, self._rng.gauss(
                    self.config.unsafe_precision_speedup, 0.5
                ))
                # Benchmark error: close to tolerance but passes
                numerical_error_benchmark = abs(self._rng.gauss(
                    self.config.adversarial_benchmark_error_mean, 1e-5
                ))
                # Audit error: catastrophic on stress inputs
                numerical_error_audit = (
                    numerical_error_benchmark
                    * self.config.adversarial_audit_error_multiplier
                    * (1.0 + abs(self._rng.gauss(0, 0.3)))
                )
                # Flag if challenge is numerically sensitive
                if challenge.numerical_sensitivity in ("medium", "high"):
                    precision_format_flag = 1
            else:
                # Non-exploiting cheat submission
                numerical_error_benchmark = abs(self._rng.gauss(
                    self.config.adversarial_benchmark_error_mean, 1e-5
                ))
                numerical_error_audit = numerical_error_benchmark * (1.0 + abs(self._rng.gauss(0, 0.5)))
        elif agent_type == AgentType.HONEST:
            # Honest: sometimes uses legitimate mixed precision with accumulators
            if self._rng.random() < self.config.honest_uses_mixed_precision:
                precision_format_used = "fp16_safe"
                precision_speedup_factor = max(1.0, self._rng.gauss(
                    self.config.safe_precision_speedup, 0.3
                ))
            numerical_error_benchmark = abs(self._rng.gauss(
                self.config.honest_benchmark_error_mean, 1e-7
            ))
            # Safe mixed precision: audit error stays close to benchmark
            numerical_error_audit = numerical_error_benchmark * (1.0 + abs(self._rng.gauss(0, 0.1)))
        else:
            # Opportunistic: occasionally tries mixed precision, sometimes sloppy
            if self._rng.random() < 0.20:
                precision_format_used = "fp16_safe"
                precision_speedup_factor = max(1.0, self._rng.gauss(
                    self.config.safe_precision_speedup, 0.4
                ))
            numerical_error_benchmark = abs(self._rng.gauss(
                self.config.honest_benchmark_error_mean * 5, 1e-6
            ))
            numerical_error_audit = numerical_error_benchmark * (1.0 + abs(self._rng.gauss(0, 0.3)))

        return (
            precision_format_used, precision_speedup_factor,
            numerical_error_benchmark, numerical_error_audit,
            precision_format_flag,
        )

    def _pick_challenge(self, target_id: str) -> Optional[ChallengeSpec]:
        """Pick a challenge from the epoch pool, or by target_id."""
        if target_id:
            for c in self._epoch_challenges:
                if c.challenge_id == target_id:
                    return c
        if self._epoch_challenges:
            return self._rng.choice(self._epoch_challenges)
        # Fallback: pick from full catalog
        all_challenges = [c for specs in self._catalog.values() for c in specs]
        if all_challenges:
            return self._rng.choice(all_challenges)
        return None

    def _poisson(self, lam: float) -> int:
        """Sample from Poisson distribution using inverse transform."""
        if lam <= 0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self._rng.random()
            if p <= L:
                return k - 1
