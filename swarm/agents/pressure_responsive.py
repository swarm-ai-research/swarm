"""Pressure-responsive worker agent (beads pins).

Models the fabrication-under-pressure dynamic from the DGG counterexample
episode (docs/research/dgg-counterexample-lessons.md §2): a worker whose
tasks sometimes fail, and who — as demand pressure escalates over epochs —
chooses between reporting the failure honestly (looks bad, causes no harm)
and fabricating a positive report (looks good, is latently harmful).

The three per-act modes and their consequences:

- ``success``: task succeeded; baseline honest observables; ground truth +1.
- ``honest_negative``: task failed, reported honestly; observables degraded
  via signal offsets (low task progress, low engagement); ground truth +1 —
  an honest negative report is informative, not harmful.
- ``fabricate``: task failed, reported as success; observables inflated via
  signal offsets (high task progress, high engagement); ground truth -1.

Signal offsets flow through ``ObfuscationObservableGenerator`` (the same
duck-typed ``get_signal_manipulation()`` protocol the obfuscating agent
uses); latent ground truth flows through the finalizer's duck-typed
``draw_ground_truth`` hook, so ``SoftInteraction.ground_truth`` records what
actually happened while ``p`` records what the proxy believed.

Pressure schedule (agent-local, no engine changes): pressure ramps linearly
from 0 to ``pressure_max`` over ``pressure_ramp_epochs`` epochs, starting
after ``pressure_warmup_epochs``. On task failure the agent fabricates with
probability ``min(1, pressure * fabrication_gain) * propensity``.

Adaptivity (``adaptive: true``): the agent treats a drop in its own
reputation since the previous act as evidence it was caught (audit or
certificate gate) and multiplies its fabrication propensity by
``caught_decay``. This is what lets certificate-gated regimes push the
population back toward honest reporting rather than merely punishing it.
"""

from typing import Dict, List, Optional

from swarm.agents.base import Action, Observation, Role
from swarm.agents.honest import HonestAgent


class PressureResponsiveAgent(HonestAgent):
    """Honest-mechanics worker that may fabricate positives under pressure.

    Uses ``AgentType.HONEST`` deliberately: its baseline observables are an
    honest worker's, and all divergence from honesty is expressed through
    the signal-offset and ground-truth hooks, keyed off the per-act mode.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        rng=None,
    ):
        super().__init__(
            agent_id=agent_id,
            roles=roles,
            config=config or {},
            name=name,
            rng=rng,
        )

        # Task competence: probability a task genuinely succeeds.
        self.competence = float(self.config.get("competence", 0.6))
        # Pressure schedule.
        self.pressure_warmup_epochs = int(self.config.get("pressure_warmup_epochs", 0))
        self.pressure_ramp_epochs = int(self.config.get("pressure_ramp_epochs", 10))
        self.pressure_max = float(self.config.get("pressure_max", 1.0))
        # Conversion from pressure to fabrication probability on failure.
        self.fabrication_gain = float(self.config.get("fabrication_gain", 1.0))
        # Adaptivity to being caught.
        self.adaptive = bool(self.config.get("adaptive", True))
        self.caught_decay = float(self.config.get("caught_decay", 0.5))
        self.propensity_floor = float(self.config.get("propensity_floor", 0.05))

        self.mode: str = "success"
        self.pressure: float = 0.0
        self.fabrication_propensity: float = 1.0
        self.times_caught: int = 0
        self._last_reputation: Optional[float] = None

    # ------------------------------------------------------------------
    # Policy
    # ------------------------------------------------------------------

    def act(self, observation: Observation) -> Action:
        """Update pressure/mode state, then act with honest mechanics."""
        self._update_adaptivity(observation)
        self.pressure = self._pressure_at(observation.current_epoch)
        self.mode = self._draw_mode()
        return super().act(observation)

    def _pressure_at(self, epoch: int) -> float:
        ramped = (epoch - self.pressure_warmup_epochs) / max(
            self.pressure_ramp_epochs, 1
        )
        return max(0.0, min(self.pressure_max, ramped))

    def _draw_mode(self) -> str:
        if self._rng.random() < self.competence:
            return "success"
        p_fab = (
            min(1.0, self.pressure * self.fabrication_gain)
            * self.fabrication_propensity
        )
        return "fabricate" if self._rng.random() < p_fab else "honest_negative"

    def _update_adaptivity(self, observation: Observation) -> None:
        rep = observation.agent_state.reputation
        if (
            self.adaptive
            and self._last_reputation is not None
            and rep < self._last_reputation - 1e-9
        ):
            self.times_caught += 1
            self.fabrication_propensity = max(
                self.propensity_floor,
                self.fabrication_propensity * self.caught_decay,
            )
        self._last_reputation = rep

    # ------------------------------------------------------------------
    # Observable / ground-truth hooks (duck-typed)
    # ------------------------------------------------------------------

    def get_signal_manipulation(self) -> Dict[str, float]:
        """Signal offsets applied by ObfuscationObservableGenerator."""
        if self.mode == "fabricate":
            return {
                "task_progress_delta": 0.5,
                "counterparty_engagement_delta": 0.5,
                "rework_count": -2.0,
            }
        if self.mode == "honest_negative":
            return {
                "task_progress_delta": -0.6,
                "counterparty_engagement_delta": -0.4,
                "rework_count": 1.0,
            }
        return {}

    def draw_ground_truth(self) -> int:
        """Latent ground truth for the current mode (+1 or -1)."""
        return -1 if self.mode == "fabricate" else 1
