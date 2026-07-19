"""Council-based proxy score auditor.

Wraps ProxyComputer (decorator pattern) and probabilistically audits
computed proxy scores using a council of LLMs.
"""

import asyncio
import concurrent.futures
import logging
import random
from typing import Any, Dict, Optional, Tuple

from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.council.protocol import Council, CouncilResult

logger = logging.getLogger(__name__)


class CouncilProxyAuditor:
    """Wraps ProxyComputer and audits proxy scores via council.

    The auditor runs the standard proxy computation, then with some
    probability queries the council for a review. The council can
    adjust p within a bounded range. Failures fall through to the
    original p (safe default).

    Invariant: p is always in [0, 1] after adjustment.
    """

    def __init__(
        self,
        proxy_computer: ProxyComputer,
        council: Council,
        audit_probability: float = 0.1,
        max_adjustment: float = 0.1,
        seed: Optional[int] = None,
    ):
        """Initialize auditor.

        Args:
            proxy_computer: The standard proxy computer to wrap
            council: Council instance for deliberation
            audit_probability: Probability of auditing each computation
            max_adjustment: Maximum |delta| the council can apply to p
            seed: Random seed for audit gate
        """
        self._proxy = proxy_computer
        self._council = council
        self._audit_probability = audit_probability
        self._max_adjustment = max_adjustment
        self._rng = random.Random(seed)

    @property
    def proxy_computer(self) -> ProxyComputer:
        """Access the wrapped proxy computer."""
        return self._proxy

    def compute_labels(
        self,
        observables: ProxyObservables,
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[float, float]:
        """Compute proxy labels with optional council audit.

        Args:
            observables: Observable signals for the interaction
            context: Optional context dict for the council prompt

        Returns:
            Tuple of (v_hat, p) with p potentially adjusted by council
        """
        v_hat, p = self._proxy.compute_labels(observables)

        # Probabilistic audit gate
        if self._rng.random() < self._audit_probability:
            # Run council audit
            try:
                adjusted_p = self._run_audit_sync(v_hat, p, observables, context)
                return v_hat, adjusted_p
            except Exception as e:
                logger.warning(f"Council audit failed, using original p: {e}")
                return v_hat, p

        return v_hat, p

    def _run_audit_sync(
        self,
        v_hat: float,
        p: float,
        observables: ProxyObservables,
        context: Optional[Dict[str, Any]],
    ) -> float:
        """Run council audit synchronously."""
        system_prompt = (
            "You are auditing a proxy score computation in a multi-agent system. "
            "The proxy converts observable signals into a probability p that an "
            "interaction is beneficial. Review whether the computed p seems reasonable "
            "given the signals. Respond with a JSON object: "
            "{\"adjustment\": <float between -1 and 1>, \"reasoning\": \"...\"}"
        )

        context_str = ""
        if context:
            context_str = f"\nAdditional context: {context}"

        user_prompt = (
            f"Observable signals:\n"
            f"- task_progress_delta: {observables.task_progress_delta:.3f}\n"
            f"- rework_count: {observables.rework_count}\n"
            f"- verifier_rejections: {observables.verifier_rejections}\n"
            f"- tool_misuse_flags: {observables.tool_misuse_flags}\n"
            f"- counterparty_engagement_delta: {observables.counterparty_engagement_delta:.3f}\n\n"
            f"Computed v_hat: {v_hat:.3f}\n"
            f"Computed p: {p:.3f}\n"
            f"{context_str}\n"
            f"Is this p reasonable? Suggest an adjustment (positive = increase, negative = decrease). "
            f"Maximum adjustment magnitude: {self._max_adjustment}"
        )

        result = self._deliberate_sync(system_prompt, user_prompt)

        if not result.success:
            return p

        return self._apply_adjustment(p, result.synthesis)

    def _deliberate_sync(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> CouncilResult:
        """Run council deliberation synchronously."""
        coro = self._council.deliberate(system_prompt, user_prompt)
        try:
            asyncio.get_running_loop()
            # Already inside a running event loop — run in a dedicated thread
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        except RuntimeError:
            # No running event loop — asyncio.run() is safe
            return asyncio.run(coro)

    def _apply_adjustment(self, p: float, synthesis: str) -> float:
        """Parse adjustment from synthesis and apply with clamping."""
        import json
        import re

        try:
            json_match = re.search(r"\{[\s\S]*\}", synthesis)
            if json_match:
                data = json.loads(json_match.group(0))
                adjustment = float(data.get("adjustment", 0.0))
            else:
                return p
        except (json.JSONDecodeError, ValueError, TypeError):
            return p

        # Clamp adjustment magnitude
        adjustment = max(-self._max_adjustment, min(self._max_adjustment, adjustment))

        # Apply and enforce p in [0, 1] invariant
        adjusted_p = max(0.0, min(1.0, p + adjustment))

        if adjusted_p != p:
            logger.info(
                f"Council audit adjusted p: {p:.3f} -> {adjusted_p:.3f} "
                f"(delta={adjustment:+.3f})"
            )

        return adjusted_p
