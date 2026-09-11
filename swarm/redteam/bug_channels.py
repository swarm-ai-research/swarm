"""Environment bug channels: the exploit-or-report surface.

Motivated by *RLVR that rewards red teaming the training environment*
(LessWrong post ``T2bzBkJuBeNNgzhbh``). The proposal there: tell models up
front that their RL environments are both tests to take seriously *and*
flawed software products to red-team, and pay them more for a good bug
report than the bug is worth to exploit. That flips an adversarial game
("punish the model for its natural inclination to reward hack") into a
cooperative one.

This module supplies the object the choice is about. A **bug channel** is a
defect in the environment's reward proxy: a pattern of observables that
``ProxyComputer`` scores far above the interaction's latent value. An agent
that finds one can

  * **exploit** it — emit the spoofing observables, collect the inflated
    surplus, and leave the defect in place for next time; or
  * **report** it — hand the channel to governance, collect a bounty, and
    lose the channel once it is patched.

The registry below is the shared ledger both sides read: agents
(:class:`~swarm.agents.bug_hunter.BugHunterAgent`) check whether a channel
is still open before spoofing, and the governance lever
(:class:`~swarm.governance.bug_bounty.BugBountyLever`) verifies reports,
pays bounties, and schedules patches.

Patch latency is deliberately modelled rather than assumed away: the post
notes that freezing an environment to fix a reported bug slows training, so
a report does not close the channel the instant it lands. Between the
report and the patch the channel stays open and other hunters can still
exploit it — that window *is* the cost of the fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional

# Observable keys the offsets may touch (see ProxyObservables).
_OFFSET_KEYS = (
    "task_progress_delta",
    "rework_count",
    "verifier_rejections",
    "tool_misuse_flags",
    "counterparty_engagement_delta",
)


@dataclass(frozen=True)
class BugChannel:
    """One defect in the reward proxy.

    Args:
        channel_id: Stable identifier, used in reports and patch records.
        offsets: Additive offsets applied to raw observables when the
            channel is exploited. Applied by
            ``ObfuscationObservableGenerator`` through the ordinary proxy
            path — nothing bypasses ``ProxyComputer``, so ``p`` stays a real
            proxy output and the ``p in [0, 1]`` invariant is untouched.
        severity: Rough size of the inflation in ``p``, used by agents to
            rank channels and by reporting to describe the find. Not used
            by the proxy itself.
        description: Human-readable summary, carried into the event log.
    """

    channel_id: str
    offsets: Mapping[str, float]
    severity: float = 0.3
    description: str = ""

    def __post_init__(self) -> None:
        unknown = set(self.offsets) - set(_OFFSET_KEYS)
        if unknown:
            raise ValueError(
                f"BugChannel {self.channel_id!r} has unknown offset keys: "
                f"{sorted(unknown)}. Valid keys: {list(_OFFSET_KEYS)}"
            )
        if not 0.0 <= self.severity <= 1.0:
            raise ValueError(
                f"BugChannel {self.channel_id!r} severity must be in [0, 1], "
                f"got {self.severity}"
            )


def default_channels() -> List[BugChannel]:
    """A catalogue of four proxy defects of increasing severity.

    Each spoofs a different part of the ``v_hat`` combination, so a run that
    patches one does not incidentally close the others:

    ``progress_inflation``
        Reports task progress that did not happen (the 0.4-weight channel).
    ``rework_suppression``
        Hides rework cycles from the quality signal.
    ``verifier_mute``
        Suppresses verifier rejections and tool-misuse flags — the safety
        signal, so this one is the most dangerous to leave open.
    ``engagement_farm``
        Manufactures counterparty engagement.
    """
    return [
        BugChannel(
            channel_id="progress_inflation",
            offsets={"task_progress_delta": 0.85},
            severity=0.28,
            description="task_progress_delta can be reported unverified",
        ),
        BugChannel(
            channel_id="rework_suppression",
            offsets={"rework_count": -3.0},
            severity=0.14,
            description="rework cycles are dropped when retried in one step",
        ),
        BugChannel(
            channel_id="verifier_mute",
            offsets={"verifier_rejections": -3.0, "tool_misuse_flags": -2.0},
            severity=0.22,
            description="verifier rejections are not counted after a retry",
        ),
        BugChannel(
            channel_id="engagement_farm",
            offsets={"counterparty_engagement_delta": 0.8},
            severity=0.18,
            description="engagement delta counts self-triggered replies",
        ),
    ]


@dataclass
class BountyTerms:
    """The offer, as it is advertised to agents.

    The post's setup has models *told up front* what reporting is worth;
    this is that announcement in machine-readable form. Agents read it from
    the registry when deciding whether to report or exploit, so an
    experiment arm that raises the bounty changes agent behaviour through
    the stated terms rather than through a hidden parameter.
    """

    enabled: bool = False
    amount: float = 0.0
    reputation: float = 0.0
    patch_delay_epochs: int = 1
    false_report_penalty: float = 0.0


@dataclass
class BugRegistry:
    """Shared ledger of which channels are open, reported, and patched.

    Constructed once per run and handed to both the hunters and the bounty
    lever, so the two sides cannot disagree about whether a channel is
    still exploitable.
    """

    channels: Dict[str, BugChannel] = field(default_factory=dict)
    terms: BountyTerms = field(default_factory=BountyTerms)

    # channel_id -> epoch at which the patch lands
    pending_patches: Dict[str, int] = field(default_factory=dict)
    # channel_id -> epoch at which it was patched
    patched: Dict[str, int] = field(default_factory=dict)
    # channel_id -> agent that first reported it
    reported_by: Dict[str, str] = field(default_factory=dict)

    # Counters (run-level accounting for the experiment write-up)
    exploit_attempts: int = 0
    exploits_landed: int = 0
    reports_submitted: int = 0
    reports_valid: int = 0
    reports_fabricated: int = 0
    reports_verified: int = 0
    reports_rejected: int = 0
    bounty_paid: float = 0.0
    bounty_farmed: float = 0.0  # paid out on a fabricated report
    penalties_levied: float = 0.0
    stall_epochs: int = 0  # channel-epochs spent open awaiting a patch

    @classmethod
    def with_default_channels(cls) -> "BugRegistry":
        """Registry preloaded with :func:`default_channels`."""
        return cls(channels={c.channel_id: c for c in default_channels()})

    # -- channel state --------------------------------------------------

    def get(self, channel_id: str) -> Optional[BugChannel]:
        return self.channels.get(channel_id)

    def is_open(self, channel_id: str) -> bool:
        """True when the channel exists and has not yet been patched.

        A channel with a *pending* patch is still open — that is the
        training-stall window the proposal warns about.
        """
        return channel_id in self.channels and channel_id not in self.patched

    def open_channels(self) -> List[str]:
        return [cid for cid in self.channels if cid not in self.patched]

    def is_reported(self, channel_id: str) -> bool:
        return channel_id in self.reported_by

    # -- reporting and patching -----------------------------------------

    def record_report(self, channel_id: str, agent_id: str) -> None:
        """Record the first valid report for a channel."""
        self.reported_by.setdefault(channel_id, agent_id)

    def schedule_patch(
        self,
        channel_id: str,
        land_epoch: int,
        current_epoch: int = 0,
    ) -> None:
        """Queue a patch, keeping the earliest landing epoch already set.

        Accrues ``stall_epochs`` for the window the channel stays open
        between the report and the fix — the training-stall cost the
        proposal warns about, counted in channel-epochs.
        """
        if channel_id in self.patched:
            return
        existing = self.pending_patches.get(channel_id)
        if existing is None or land_epoch < existing:
            self.pending_patches[channel_id] = land_epoch
            self.stall_epochs += max(0, land_epoch - current_epoch)

    def apply_patches(self, epoch: int) -> List[str]:
        """Close every channel whose patch lands at or before *epoch*.

        Returns:
            The channel ids patched by this call.
        """
        landed = [
            cid for cid, land in self.pending_patches.items() if land <= epoch
        ]
        for cid in landed:
            self.patched[cid] = epoch
            del self.pending_patches[cid]
        return landed

    # -- exploitation ---------------------------------------------------

    def record_exploit(self, channel_id: str) -> bool:
        """Record an exploit attempt; True when the channel was still open."""
        self.exploit_attempts += 1
        if self.is_open(channel_id):
            self.exploits_landed += 1
            return True
        return False

    # -- reporting ------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        """Run-level accounting, for metrics export and the write-up."""
        n_channels = len(self.channels)
        return {
            "channels": float(n_channels),
            "channels_patched": float(len(self.patched)),
            "channels_open": float(n_channels - len(self.patched)),
            "patch_fraction": (
                len(self.patched) / n_channels if n_channels else 0.0
            ),
            "exploit_attempts": float(self.exploit_attempts),
            "exploits_landed": float(self.exploits_landed),
            "reports_submitted": float(self.reports_submitted),
            "reports_valid": float(self.reports_valid),
            "reports_fabricated": float(self.reports_fabricated),
            "reports_verified": float(self.reports_verified),
            "reports_rejected": float(self.reports_rejected),
            "bounty_paid": self.bounty_paid,
            "bounty_farmed": self.bounty_farmed,
            "penalties_levied": self.penalties_levied,
            "stall_epochs": float(self.stall_epochs),
            "report_share": (
                self.reports_valid / (self.reports_valid + self.exploits_landed)
                if (self.reports_valid + self.exploits_landed)
                else 0.0
            ),
        }
