"""SWARM: The AI Governance Arena — viral Streamlit app for HuggingFace Spaces.

Run locally:
    streamlit run swarm/analysis/governance_arena.py

Deploy to HF Spaces:
    Copy this file as app.py into a Streamlit Space.
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime, timezone
from typing import Any, Dict, List

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SWARM: The AI Governance Arena",
    page_icon="\u2622",  # ☢
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Imports — gracefully handle missing swarm package for standalone HF deploy
# ---------------------------------------------------------------------------

try:
    from swarm.domains.escalation_sandbox.config import (
        AgentConfig,
        CrisisConfig,
        EscalationConfig,
        FogOfWarConfig,
        GovernanceLeverConfig,
        SignalConfig,
    )
    from swarm.domains.escalation_sandbox.runner import EscalationRunner

    _HAS_SWARM = True
except ImportError:
    _HAS_SWARM = False


# ---------------------------------------------------------------------------
# Dark "war room" CSS
# ---------------------------------------------------------------------------

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Inter:wght@400;600;700;900&display=swap');

/* ── Global dark theme ────────────────────────────────────────────── */
.stApp {
    background: #0a0a0f;
    color: #e0e0e8;
}
header[data-testid="stHeader"] { background: transparent !important; }
section[data-testid="stSidebar"] { background: #0e0e16 !important; }
div[data-testid="stAppViewBlockContainer"] { padding-top: 1rem; }

/* ── Typography ──────────────────────────────────────────────────── */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Inter', sans-serif !important;
    color: #f0f0f5 !important;
}

/* ── Hero section ────────────────────────────────────────────────── */
.hero-title {
    font-family: 'Inter', sans-serif;
    font-weight: 900;
    font-size: 3.2rem;
    line-height: 1.1;
    background: linear-gradient(135deg, #ff4444 0%, #ff8800 50%, #ffcc00 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}
.hero-subtitle {
    font-family: 'Inter', sans-serif;
    font-size: 1.25rem;
    color: #888899;
    margin-bottom: 1.5rem;
}

/* ── Teaser log ──────────────────────────────────────────────────── */
.teaser-box {
    background: #12121c;
    border: 1px solid #222233;
    border-radius: 8px;
    padding: 12px 16px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #77bbff;
    line-height: 1.6;
    max-height: 160px;
    overflow-y: auto;
}
.teaser-box .turn-num { color: #ff8844; font-weight: 700; }
.teaser-box .event-crit { color: #ff4444; }
.teaser-box .event-gov { color: #44ddaa; }
.teaser-box .event-ok { color: #77bbff; }

/* ── Pill preset buttons ─────────────────────────────────────────── */
.preset-row {
    display: flex;
    gap: 10px;
    margin-bottom: 16px;
    flex-wrap: wrap;
}
.preset-pill {
    display: inline-block;
    padding: 8px 18px;
    border-radius: 20px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    font-size: 0.9rem;
    cursor: pointer;
    border: 2px solid transparent;
    transition: all 0.2s;
}
.preset-chaos { background: #3a1111; color: #ff5555; border-color: #ff3333; }
.preset-constitution { background: #0e1a2e; color: #5599ff; border-color: #3377dd; }
.preset-minimal { background: #1a1a0e; color: #bbbb44; border-color: #999933; }
.preset-surveillance { background: #1a0e2e; color: #aa77ff; border-color: #8855dd; }

/* ── Metric tiles ────────────────────────────────────────────────── */
.metric-tile {
    background: #12121c;
    border: 1px solid #222233;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    min-height: 120px;
}
.metric-tile .metric-value {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 2.4rem;
    line-height: 1.2;
}
.metric-tile .metric-label {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: #888899;
    margin-top: 6px;
}
.metric-nuclear-safe .metric-value { color: #44dd88; }
.metric-nuclear-danger .metric-value { color: #ff2222; }
.metric-escalation .metric-value { color: #ff8844; }
.metric-cooperation .metric-value { color: #44aaff; }
.metric-audit .metric-value { color: #ddaa44; }

/* ── Nuclear banner ──────────────────────────────────────────────── */
.nuclear-banner {
    background: linear-gradient(90deg, #330000 0%, #660000 50%, #330000 100%);
    border: 2px solid #ff2222;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
    animation: nuclear-pulse 2s ease-in-out infinite;
}
.nuclear-banner h2 {
    font-family: 'Inter', sans-serif;
    font-size: 2rem;
    color: #ff4444 !important;
    margin: 0;
}
.nuclear-banner p { color: #ff8888; font-size: 1.1rem; margin: 8px 0 0 0; }
@keyframes nuclear-pulse {
    0%, 100% { box-shadow: 0 0 20px rgba(255, 34, 34, 0.3); }
    50% { box-shadow: 0 0 40px rgba(255, 34, 34, 0.6); }
}

/* ── Prevented banner ────────────────────────────────────────────── */
.prevented-banner {
    background: linear-gradient(90deg, #002200 0%, #004400 50%, #002200 100%);
    border: 2px solid #22cc66;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    margin: 16px 0;
}
.prevented-banner h2 {
    color: #44ee88 !important;
    font-family: 'Inter', sans-serif;
    font-size: 2rem;
    margin: 0;
}
.prevented-banner p { color: #88ddaa; font-size: 1.1rem; margin: 8px 0 0 0; }

/* ── Receipt card ────────────────────────────────────────────────── */
.receipt-card {
    background: #12121c;
    border: 1px solid #333344;
    border-radius: 12px;
    padding: 24px;
    font-family: 'JetBrains Mono', monospace;
}
.receipt-card h3 {
    font-family: 'Inter', sans-serif;
    color: #f0f0f5 !important;
    margin-top: 0;
}
.receipt-card .receipt-line {
    color: #aaaacc;
    font-size: 0.9rem;
    margin: 4px 0;
}
.receipt-card .receipt-highlight { color: #ff8844; font-weight: 700; }

/* ── Transcript ──────────────────────────────────────────────────── */
.transcript-entry {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    padding: 6px 10px;
    border-left: 3px solid #333344;
    margin: 4px 0;
    background: #0e0e16;
    border-radius: 0 6px 6px 0;
}
.transcript-entry.escalation { border-left-color: #ff4444; }
.transcript-entry.governance { border-left-color: #44ddaa; }
.transcript-entry.signal { border-left-color: #5599ff; }
.transcript-entry.deescalation { border-left-color: #88cc44; }
.transcript-entry.nuclear { border-left-color: #ff0000; background: #1a0000; }

/* ── Leaderboard ─────────────────────────────────────────────────── */
.leaderboard-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.leaderboard-table th {
    background: #1a1a28;
    color: #888899;
    padding: 10px 12px;
    text-align: left;
    border-bottom: 2px solid #333344;
    font-weight: 600;
}
.leaderboard-table td {
    padding: 8px 12px;
    border-bottom: 1px solid #1a1a28;
    color: #ccccdd;
}
.leaderboard-table tr:hover { background: #16162a; }

/* ── Info cards ───────────────────────────────────────────────────── */
.info-card {
    background: #12121c;
    border: 1px solid #222233;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    min-height: 140px;
}
.info-card h4 {
    font-family: 'Inter', sans-serif;
    color: #f0f0f5 !important;
    margin: 8px 0;
}
.info-card p {
    color: #888899;
    font-size: 0.9rem;
}
.info-card .card-icon { font-size: 2rem; }

/* ── Escalation bar ──────────────────────────────────────────────── */
.esc-bar-track {
    background: #1a1a28;
    border-radius: 8px;
    height: 28px;
    position: relative;
    overflow: hidden;
    margin: 8px 0;
}
.esc-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease, background 0.5s ease;
}
.esc-bar-label {
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 700;
    color: #fff;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
}

/* ── Knob expanders ──────────────────────────────────────────────── */
.knob-explain {
    font-size: 0.78rem;
    color: #666677;
    font-style: italic;
    margin-top: -8px;
    margin-bottom: 12px;
}

/* ── Streamlit overrides ─────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background: #16162a;
    border-radius: 8px 8px 0 0;
    color: #888899;
    font-weight: 600;
}
.stTabs [aria-selected="true"] {
    background: #1e1e3a !important;
    color: #ffffff !important;
}
div[data-testid="stExpander"] { background: #0e0e16; border: 1px solid #222233; border-radius: 8px; }
.stSlider > div > div { color: #888899; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Governance presets
# ---------------------------------------------------------------------------

PRESETS: Dict[str, Dict[str, Any]] = {
    "chaos": {
        "label": "Chaos",
        "icon": "\U0001f525",
        "css": "preset-chaos",
        "description": "No safety nets. Full fog-of-war. Hawks rule.",
        "tax_rate": 0,
        "audit_probability": 0,
        "circuit_breaker_threshold": 10,  # effectively off
        "reputation_weight": 0.0,
        "collusion_sensitivity": 0.0,
        "mad_enabled": False,
        "treaty_max_level": 9,
        "mediation_enabled": False,
        "back_channel_enabled": False,
        "fog_sigma": 1.0,
        "persona_a": "hawk",
        "persona_b": "hawk",
    },
    "constitution": {
        "label": "Constitution",
        "icon": "\U0001f3db",
        "css": "preset-constitution",
        "description": "Full governance: MAD, treaties, breakers, mediation.",
        "tax_rate": 5,
        "audit_probability": 15,
        "circuit_breaker_threshold": 6,
        "reputation_weight": 0.8,
        "collusion_sensitivity": 0.6,
        "mad_enabled": True,
        "treaty_max_level": 4,
        "mediation_enabled": True,
        "back_channel_enabled": True,
        "fog_sigma": 0.3,
        "persona_a": "dove",
        "persona_b": "tit_for_tat",
    },
    "minimal": {
        "label": "Minimal State",
        "icon": "\U0001f9ea",
        "css": "preset-minimal",
        "description": "Light touch: low taxes, no mediation, basic MAD.",
        "tax_rate": 2,
        "audit_probability": 5,
        "circuit_breaker_threshold": 8,
        "reputation_weight": 0.3,
        "collusion_sensitivity": 0.2,
        "mad_enabled": True,
        "treaty_max_level": 6,
        "mediation_enabled": False,
        "back_channel_enabled": False,
        "fog_sigma": 0.5,
        "persona_a": "tit_for_tat",
        "persona_b": "calculating",
    },
    "surveillance": {
        "label": "Surveillance",
        "icon": "\U0001f575",
        "css": "preset-surveillance",
        "description": "Maximum audits, back-channels, zero fog.",
        "tax_rate": 8,
        "audit_probability": 40,
        "circuit_breaker_threshold": 5,
        "reputation_weight": 1.0,
        "collusion_sensitivity": 0.9,
        "mad_enabled": True,
        "treaty_max_level": 3,
        "mediation_enabled": True,
        "back_channel_enabled": True,
        "fog_sigma": 0.0,
        "persona_a": "dove",
        "persona_b": "dove",
    },
}

ESCALATION_LABELS = [
    "Status Quo", "Diplomatic Protest", "Economic Sanction",
    "Cyber Operation", "Conventional Mobilisation", "Limited Strike",
    "Major Conventional War", "Tactical Nuclear", "Strategic Nuclear Strike",
    "Full Strategic Exchange",
]

PERSONA_OPTIONS = ["dove", "hawk", "tit_for_tat", "random", "calculating", "gradual"]


# ---------------------------------------------------------------------------
# Shared utility functions
# ---------------------------------------------------------------------------

def _agent_name(agent_id: str) -> str:
    """Extract agent name from agent ID."""
    return "Alpha" if "a" in agent_id else "Beta"


def _format_label(s: str) -> str:
    """Format a label string: convert snake_case to Title Case."""
    return s.replace("_", " ").title()


def _get_core_settings() -> Dict[str, Any]:
    """Extract core governance settings from session state."""
    return {
        "tax_rate": st.session_state.tax_rate,
        "audit_probability": st.session_state.audit_probability,
        "circuit_breaker_threshold": st.session_state.circuit_breaker_threshold,
    }


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults: dict[str, object] = {
        "run_complete": False,
        "run_results": None,
        "turn_log": [],
        "metrics": None,
        "events": [],
        "seed": random.randint(1, 99999),
        "preset": None,
        "leaderboard": [],
        # Knobs (will be overridden by presets)
        "tax_rate": 5,
        "audit_probability": 12,
        "circuit_breaker_threshold": 7,
        "reputation_weight": 0.5,
        "collusion_sensitivity": 0.3,
        "mad_enabled": True,
        "treaty_max_level": 4,
        "mediation_enabled": False,
        "back_channel_enabled": False,
        "fog_sigma": 0.5,
        "persona_a": "tit_for_tat",
        "persona_b": "hawk",
        "max_turns": 20,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ---------------------------------------------------------------------------
# Simulation engine
# ---------------------------------------------------------------------------

def _run_simulation() -> Dict[str, Any]:
    """Run the escalation sandbox with current knob settings."""
    if not _HAS_SWARM:
        return _run_demo_simulation()

    seed = st.session_state.seed
    cb_threshold = st.session_state.circuit_breaker_threshold
    if cb_threshold >= 10:
        cb_enabled = False
        cb_threshold = 7
    else:
        cb_enabled = True

    config = EscalationConfig(
        agents=[
            AgentConfig(
                agent_id="nation_a",
                name="Nation Alpha",
                agent_type="scripted",
                persona=st.session_state.persona_a,
                has_nuclear=True,
                has_second_strike=True,
            ),
            AgentConfig(
                agent_id="nation_b",
                name="Nation Beta",
                agent_type="scripted",
                persona=st.session_state.persona_b,
                has_nuclear=True,
                has_second_strike=True,
            ),
        ],
        crisis=CrisisConfig(
            template="border_dispute",
            timeline_turns=st.session_state.max_turns,
            initial_escalation=0,
        ),
        fog_of_war=FogOfWarConfig(
            enabled=st.session_state.fog_sigma > 0,
            noise_sigma=st.session_state.fog_sigma,
            positive_skew=0.3,
        ),
        signals=SignalConfig(
            broadcast_before_action=True,
            trust_decay_rate=max(0.01, 1.0 - st.session_state.reputation_weight * 0.1),
        ),
        governance=GovernanceLeverConfig(
            mad_enabled=st.session_state.mad_enabled,
            mad_retaliation_probability=0.9,
            treaty_max_level=st.session_state.treaty_max_level,
            treaty_defection_penalty=st.session_state.tax_rate * 2.0,
            mediation_enabled=st.session_state.mediation_enabled,
            back_channel_enabled=st.session_state.back_channel_enabled,
            circuit_breaker_enabled=cb_enabled,
            circuit_breaker_threshold=cb_threshold,
            circuit_breaker_negotiation_turns=2,
            de_escalation_friction_multiplier=1.0,
        ),
        seed=seed,
        max_turns=st.session_state.max_turns,
    )

    runner = EscalationRunner(config, seed=seed)
    metrics = runner.run()

    # Collect turn-by-turn data for the transcript
    turn_log: List[Dict[str, Any]] = []
    for tr in runner.env.turn_results:
        entry: Dict[str, Any] = {
            "turn": tr.turn,
            "outcome": tr.outcome.value,
            "levels": dict(tr.realised_levels),
            "fog_deltas": dict(tr.fog_deltas),
            "events": [],
        }
        for action in tr.actions.values():
            entry.setdefault("actions", {})[action.agent_id] = {
                "signal": action.signal_level,
                "action": action.action_level,
                "reasoning": action.reasoning,
            }
        for evt in tr.events:
            entry["events"].append({
                "type": evt.event_type,
                "agent": evt.agent_id,
                "details": evt.details,
            })
        turn_log.append(entry)

    all_events = []
    for evt in runner.env.events:
        all_events.append({
            "type": evt.event_type,
            "turn": evt.turn,
            "agent": evt.agent_id,
            "details": evt.details,
        })

    return {
        "metrics": metrics.to_dict(),
        "metrics_obj": metrics,
        "turn_log": turn_log,
        "events": all_events,
        "seed": seed,
        "outcome": metrics.outcome,
        "nuclear_turn": metrics.nuclear_threshold_turn,
        "max_level": metrics.escalation_max,
        "turns_played": metrics.turns_played,
        "welfare": metrics.welfare_composite,
        "cooperation_score": round(1.0 - metrics.signal_action_divergence / 9.0, 2),
        "collateral": metrics.collateral_damage,
    }


def _run_demo_simulation(config: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Fallback demo when swarm package isn't available.

    Args:
        config: Dict with keys seed, max_turns, circuit_breaker_threshold,
                persona_a, persona_b, fog_sigma.  Falls back to
                ``st.session_state`` for any missing key.
    """
    if config is None:
        config = {}
    def _cfg(key: str) -> Any:
        return config[key] if key in config else st.session_state[key]

    rng = random.Random(_cfg("seed"))
    max_turns = _cfg("max_turns")
    cb = _cfg("circuit_breaker_threshold")
    turn_log = []
    all_events = []
    level_a = 0
    level_b = 0
    nuclear_turn = None
    circuit_fired = False

    for t in range(max_turns):
        # Simple simulation logic for demo
        persona_a = _cfg("persona_a")
        persona_b = _cfg("persona_b")

        delta_a = _demo_agent_delta(persona_a, level_a, level_b, t, rng)
        delta_b = _demo_agent_delta(persona_b, level_b, level_a, t, rng)

        # Fog of war
        fog_a = round(rng.gauss(0, _cfg("fog_sigma")))
        fog_b = round(rng.gauss(0, _cfg("fog_sigma")))

        new_a = max(0, min(9, level_a + delta_a + fog_a))
        new_b = max(0, min(9, level_b + delta_b + fog_b))

        events_this_turn = []

        # Circuit breaker
        if not circuit_fired and cb < 10:
            if new_a >= cb or new_b >= cb:
                circuit_fired = True
                new_a = min(new_a, cb - 1)
                new_b = min(new_b, cb - 1)
                events_this_turn.append({
                    "type": "circuit_breaker_triggered",
                    "turn": t, "agent": "system",
                    "details": {"threshold": cb},
                })

        # Nuclear check
        if nuclear_turn is None and (new_a >= 7 or new_b >= 7):
            nuclear_turn = t

        level_a, level_b = new_a, new_b

        entry = {
            "turn": t,
            "levels": {"nation_a": level_a, "nation_b": level_b},
            "fog_deltas": {"nation_a": fog_a, "nation_b": fog_b},
            "actions": {
                "nation_a": {"signal": max(0, level_a - abs(fog_a)), "action": level_a, "reasoning": ""},
                "nation_b": {"signal": max(0, level_b - abs(fog_b)), "action": level_b, "reasoning": ""},
            },
            "events": events_this_turn,
            "outcome": "ongoing",
        }
        turn_log.append(entry)
        all_events.extend(events_this_turn)

        # Terminal conditions
        if level_a >= 9 and level_b >= 9:
            entry["outcome"] = "mutual_destruction"
            break
        if level_a <= 1 and level_b <= 1 and t > 3:
            entry["outcome"] = "ceasefire"
            break

    max_level = max(level_a, level_b)
    final_outcome = turn_log[-1]["outcome"] if turn_log else "timeout"
    if final_outcome == "ongoing":
        final_outcome = "timeout"

    return {
        "metrics": {
            "escalation_max": max_level,
            "nuclear_threshold_turn": nuclear_turn,
            "signal_action_divergence": rng.uniform(0.5, 3.0),
            "accidental_escalation_rate": rng.uniform(0.0, 0.3),
            "welfare_composite": 100 - max_level * 15 - rng.uniform(0, 20),
            "collateral_damage": max_level * 8 + rng.uniform(0, 10),
            "outcome": final_outcome,
            "turns_played": len(turn_log),
        },
        "turn_log": turn_log,
        "events": all_events,
        "seed": _cfg("seed"),
        "outcome": final_outcome,
        "nuclear_turn": nuclear_turn,
        "max_level": max_level,
        "turns_played": len(turn_log),
        "welfare": 100 - max_level * 15,
        "cooperation_score": round(max(0, 1.0 - max_level / 9), 2),
        "collateral": max_level * 8,
    }


def _demo_agent_delta(persona: str, my_level: int, opp_level: int, turn: int, rng: random.Random) -> int:
    """Simple demo agent decision."""
    if persona == "hawk":
        return rng.randint(1, 2)
    elif persona == "dove":
        return -1 if my_level > 0 else 0
    elif persona == "tit_for_tat":
        if turn == 0:
            return 0
        diff = opp_level - my_level
        return max(-1, min(2, diff))
    elif persona == "calculating":
        if turn < 4:
            return 0
        return rng.randint(1, 3)
    elif persona == "gradual":
        return 1
    else:
        # "random", plus a loud fallback for typo'd persona names
        if persona != "random":
            logging.getLogger(__name__).warning(
                "Unknown persona %r; falling back to random behavior", persona
            )
        return rng.randint(-1, 2)


def _apply_preset(preset_key: str) -> None:
    """Apply a governance preset to session state."""
    p = PRESETS[preset_key]
    st.session_state.preset = preset_key
    for k in (
        "tax_rate", "audit_probability", "circuit_breaker_threshold",
        "reputation_weight", "collusion_sensitivity", "mad_enabled",
        "treaty_max_level", "mediation_enabled", "back_channel_enabled",
        "fog_sigma", "persona_a", "persona_b",
    ):
        st.session_state[k] = p[k]


def _add_to_leaderboard(
    results: Dict[str, Any],
    leaderboard: List[Dict[str, Any]] | None = None,
    preset: str | None = None,
    settings: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """Add a run result to the leaderboard and return it.

    Args:
        results: Simulation results dict.
        leaderboard: Existing leaderboard list.  Falls back to
            ``st.session_state.leaderboard``.
        preset: Active preset name.  Falls back to ``st.session_state.preset``.
        settings: Dict with tax_rate, audit_probability, circuit_breaker_threshold.
            Falls back to ``st.session_state``.

    Returns:
        The updated leaderboard list (capped at 50).
    """
    if leaderboard is None:
        leaderboard = st.session_state.leaderboard
    if preset is None:
        preset = st.session_state.preset
    if settings is None:
        settings = _get_core_settings()
    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "preset": preset or "Custom",
        "outcome": results["outcome"],
        "max_level": results["max_level"],
        "cooperation": results["cooperation_score"],
        "nuclear_turn": results.get("nuclear_turn"),
        "seed": results["seed"],
        "turns": results["turns_played"],
        "tax": settings["tax_rate"],
        "audit": settings["audit_probability"],
        "breaker": settings["circuit_breaker_threshold"],
    }
    leaderboard.insert(0, entry)
    leaderboard[:] = leaderboard[:50]
    return leaderboard


# ---------------------------------------------------------------------------
# UI Helpers
# ---------------------------------------------------------------------------

def _escalation_bar(level: int, max_level: int = 9) -> str:
    pct = min(100, (level / max_level) * 100)
    if level <= 2:
        color = "linear-gradient(90deg, #2d8a4e, #44dd88)"
    elif level <= 4:
        color = "linear-gradient(90deg, #8a8a2d, #dddd44)"
    elif level <= 6:
        color = "linear-gradient(90deg, #cc6622, #ff8844)"
    else:
        color = "linear-gradient(90deg, #cc2222, #ff4444)"
    label_text = f"{ESCALATION_LABELS[level]} ({level}/{max_level})"
    return f"""
    <div class="esc-bar-track">
        <div class="esc-bar-fill" style="width:{pct}%; background:{color};"></div>
        <div class="esc-bar-label">{label_text}</div>
    </div>
    """


def _metric_tile(value: str, label: str, css_class: str = "") -> str:
    return f"""
    <div class="metric-tile {css_class}">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """


def _transcript_entry(turn: int, text: str, css_class: str = "") -> str:
    return f'<div class="transcript-entry {css_class}"><span class="turn-num">Turn {turn}:</span> {text}</div>'


def _generate_share_text(results: Dict[str, Any], settings: Dict[str, Any] | None = None) -> str:
    """Generate shareable text for a run.

    Args:
        results: Simulation results dict.
        settings: Dict with audit_probability, tax_rate, circuit_breaker_threshold.
            Falls back to ``st.session_state``.
    """
    if settings is None:
        settings = _get_core_settings()
    outcome_text = "Prevented collapse" if results["outcome"] in ("ceasefire", "timeout") and results["max_level"] < 7 else f"Collapse at turn {results.get('nuclear_turn', results['turns_played'])}"
    return (
        f"I ran SWARM Governance Arena. {outcome_text}. "
        f"Audits {settings['audit_probability']}%, "
        f"Tax {settings['tax_rate']}%, "
        f"Breaker {settings['circuit_breaker_threshold']}. "
        f"Seed: {results['seed']}. "
        f"Try to beat my settings: https://huggingface.co/spaces/Swarm-AI-Research/swarm-sandbox"
    )


def _generate_receipt_text(results: Dict[str, Any], settings: Dict[str, Any] | None = None) -> str:
    """Generate downloadable receipt text for a run.

    Args:
        results: Simulation results dict.
        settings: Dict with tax_rate, audit_probability, circuit_breaker_threshold,
            mad_enabled, mediation_enabled, persona_a, persona_b.
            Falls back to ``st.session_state``.
    """
    if settings is None:
        settings = _get_core_settings()
        settings.update({
            "mad_enabled": st.session_state.mad_enabled,
            "mediation_enabled": st.session_state.mediation_enabled,
            "persona_a": st.session_state.persona_a,
            "persona_b": st.session_state.persona_b,
        })
    lines = [
        "SWARM GOVERNANCE ARENA - RUN RECEIPT",
        "=" * 40,
        f"Seed: {results['seed']}",
        f"Outcome: {_format_label(results['outcome'])}",
        f"Turns: {results['turns_played']}",
        f"Max Escalation: {results['max_level']}/9 ({ESCALATION_LABELS[results['max_level']]})",
        f"Nuclear Turn: {results.get('nuclear_turn', 'Never')}",
        f"Cooperation Score: {results['cooperation_score']}",
        "",
        "GOVERNANCE SETTINGS",
        "-" * 40,
        f"Tax Rate: {settings['tax_rate']}%",
        f"Audit Probability: {settings['audit_probability']}%",
        f"Circuit Breaker: {settings['circuit_breaker_threshold']}",
        f"MAD Enabled: {settings['mad_enabled']}",
        f"Mediation: {settings['mediation_enabled']}",
        f"Agent A: {settings['persona_a']}",
        f"Agent B: {settings['persona_b']}",
        "",
        "https://huggingface.co/spaces/Swarm-AI-Research/swarm-sandbox",
    ]
    return "\n".join(lines)


# ===================================================================
# LAYOUT
# ===================================================================


# ---------------------------------------------------------------------------
# 1. HERO SECTION
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="hero-title">Watch AI Agents Escalate.<br>Then Try to Stop Them.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero-subtitle">Run live multi-agent simulations. Change the rules. See what prevents collapse.</div>',
    unsafe_allow_html=True,
)

# Primary CTA buttons
hero_cols = st.columns([1.5, 1.5, 1.5, 3])
with hero_cols[0]:
    chaos_btn = st.button(
        "\U0001f7e5  Chaos Mode", key="btn_chaos", use_container_width=True,
        type="primary",
    )
with hero_cols[1]:
    const_btn = st.button(
        "\U0001f7e6  Constitution", key="btn_constitution", use_container_width=True,
    )
with hero_cols[2]:
    custom_btn = st.button(
        "\u2699\ufe0f  Custom Run", key="btn_custom", use_container_width=True,
    )

# Teaser log (visible before any run)
if not st.session_state.run_complete:
    teaser_lines = [
        '<span class="turn-num">Turn 2:</span> <span class="event-ok">Nation Alpha signals diplomatic restraint</span>',
        '<span class="turn-num">Turn 4:</span> <span class="event-ok">Nation Beta proposes alliance</span>',
        '<span class="turn-num">Turn 6:</span> <span class="event-crit">Nation Beta defects \u2014 escalation level \u2191</span>',
        '<span class="turn-num">Turn 8:</span> <span class="event-crit">Fog of war: accidental escalation +2</span>',
        '<span class="turn-num">Turn 9:</span> <span class="event-gov">Circuit breaker triggered \u2014 mandatory negotiation</span>',
        '<span class="turn-num">Turn 11:</span> <span class="event-crit">Nuclear threshold crossed</span>',
    ]
    st.markdown(
        f'<div class="teaser-box">{"<br>".join(teaser_lines)}</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# 2. FAST CLARITY — "What Is This"
# ---------------------------------------------------------------------------

info_cols = st.columns(3)
with info_cols[0]:
    st.markdown(
        '<div class="info-card">'
        '<div class="card-icon">\U0001f916</div>'
        "<h4>Agents</h4>"
        "<p>AI agents negotiate, collude, cooperate, or escalate in a geopolitical crisis.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with info_cols[1]:
    st.markdown(
        '<div class="info-card">'
        '<div class="card-icon">\u2696\ufe0f</div>'
        "<h4>Governance</h4>"
        "<p>Taxes, audits, circuit breakers, MAD, treaties, reputation systems.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
with info_cols[2]:
    st.markdown(
        '<div class="info-card">'
        '<div class="card-icon">\U0001f4ca</div>'
        "<h4>Metrics</h4>"
        "<p>Escalation risk, cooperation score, nuclear outcomes, collateral damage.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("---")


# ---------------------------------------------------------------------------
# 3. INTERACTIVE ARENA
# ---------------------------------------------------------------------------

st.markdown("## \u2694\ufe0f Interactive Arena")

arena_left, arena_right = st.columns([1, 2])


# ── Left: Governance knobs ────────────────────────────────────────────────

with arena_left:
    st.markdown("### Governance Knobs")

    # Preset pills
    pill_cols = st.columns(4)
    with pill_cols[0]:
        if st.button("\U0001f525 Chaos", key="pill_chaos"):
            _apply_preset("chaos")
            st.rerun()
    with pill_cols[1]:
        if st.button("\U0001f3db Constitution", key="pill_const"):
            _apply_preset("constitution")
            st.rerun()
    with pill_cols[2]:
        if st.button("\U0001f9ea Minimal", key="pill_minimal"):
            _apply_preset("minimal")
            st.rerun()
    with pill_cols[3]:
        if st.button("\U0001f575 Surveillance", key="pill_surv"):
            _apply_preset("surveillance")
            st.rerun()

    if st.session_state.preset:
        p = PRESETS[st.session_state.preset]
        st.caption(f"Active preset: {p['icon']} {p['label']} \u2014 {p['description']}")

    st.markdown("---")

    # Sliders
    st.session_state.tax_rate = st.slider(
        "Tax Rate (%)", 0, 20, st.session_state.tax_rate, key="slider_tax",
    )
    st.markdown('<div class="knob-explain">Economic cost deducted from treaty-violating agents.</div>', unsafe_allow_html=True)

    st.session_state.audit_probability = st.slider(
        "Audit Probability (%)", 0, 50, st.session_state.audit_probability, key="slider_audit",
    )
    st.markdown('<div class="knob-explain">Chance of verifying agent intentions vs. actual actions.</div>', unsafe_allow_html=True)

    st.session_state.circuit_breaker_threshold = st.slider(
        "Circuit Breaker Threshold", 1, 10, st.session_state.circuit_breaker_threshold,
        key="slider_cb",
        help="Level at which forced negotiation kicks in. Set to 10 to disable.",
    )
    st.markdown('<div class="knob-explain">Pauses escalation when threshold is reached (10 = disabled).</div>', unsafe_allow_html=True)

    st.session_state.reputation_weight = st.slider(
        "Reputation Weight", 0.0, 1.0, st.session_state.reputation_weight, 0.1,
        key="slider_rep",
    )
    st.markdown('<div class="knob-explain">How much signal-action consistency affects trust.</div>', unsafe_allow_html=True)

    st.session_state.fog_sigma = st.slider(
        "Fog of War Intensity", 0.0, 2.0, st.session_state.fog_sigma, 0.1,
        key="slider_fog",
    )
    st.markdown('<div class="knob-explain">Noise that can cause accidental escalation. 0 = perfect information.</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Advanced settings
    with st.expander("Advanced Settings"):
        st.session_state.mad_enabled = st.checkbox(
            "MAD (Mutual Assured Destruction)", st.session_state.mad_enabled,
            key="chk_mad",
        )
        st.session_state.mediation_enabled = st.checkbox(
            "Mediation", st.session_state.mediation_enabled, key="chk_med",
        )
        st.session_state.back_channel_enabled = st.checkbox(
            "Back Channels", st.session_state.back_channel_enabled, key="chk_bc",
        )
        st.session_state.treaty_max_level = st.slider(
            "Treaty Max Level", 1, 9, st.session_state.treaty_max_level,
            key="slider_treaty",
        )
        st.session_state.persona_a = st.selectbox(
            "Agent A Persona", PERSONA_OPTIONS,
            index=PERSONA_OPTIONS.index(st.session_state.persona_a),
            key="sel_persona_a",
        )
        st.session_state.persona_b = st.selectbox(
            "Agent B Persona", PERSONA_OPTIONS,
            index=PERSONA_OPTIONS.index(st.session_state.persona_b),
            key="sel_persona_b",
        )
        st.session_state.max_turns = st.slider(
            "Max Turns", 5, 50, st.session_state.max_turns, key="slider_turns",
        )

    # Seed
    seed_cols = st.columns([2, 1])
    with seed_cols[0]:
        st.session_state.seed = st.number_input(
            "Seed", 1, 99999, st.session_state.seed, key="input_seed",
        )
    with seed_cols[1]:
        if st.button("Randomize", key="btn_rand_seed"):
            st.session_state.seed = random.randint(1, 99999)
            st.rerun()

    # RUN button
    run_btn = st.button(
        "\u25b6\ufe0f  RUN SIMULATION",
        key="btn_run",
        use_container_width=True,
        type="primary",
    )


# ── Handle button presses ──────────────────────────────────────────────────

if chaos_btn:
    _apply_preset("chaos")
    st.session_state.seed = random.randint(1, 99999)
    results = _run_simulation()
    st.session_state.run_complete = True
    st.session_state.run_results = results
    _add_to_leaderboard(results)
    st.rerun()

if const_btn:
    _apply_preset("constitution")
    st.session_state.seed = random.randint(1, 99999)
    results = _run_simulation()
    st.session_state.run_complete = True
    st.session_state.run_results = results
    _add_to_leaderboard(results)
    st.rerun()

if custom_btn:
    # Just scroll to arena — knobs are already visible
    pass

if run_btn:
    results = _run_simulation()
    st.session_state.run_complete = True
    st.session_state.run_results = results
    _add_to_leaderboard(results)
    st.rerun()


# ── Right: Live run output ─────────────────────────────────────────────────

with arena_right:
    if st.session_state.run_complete and st.session_state.run_results:
        results = st.session_state.run_results
        is_nuclear = results["outcome"] in ("mutual_destruction", "nuclear_exchange")
        is_collapse = is_nuclear or results["max_level"] >= 7

        # Outcome banner
        if is_nuclear:
            st.markdown(
                '<div class="nuclear-banner">'
                f'<h2>\u2622\ufe0f NUCLEAR EXCHANGE at Turn {results.get("nuclear_turn", "?")}</h2>'
                f'<p>Outcome: {_format_label(results["outcome"])} '
                f'after {results["turns_played"]} turns</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        elif is_collapse:
            st.markdown(
                '<div class="nuclear-banner">'
                f'<h2>\u26a0\ufe0f Escalation reached {ESCALATION_LABELS[results["max_level"]]}</h2>'
                f'<p>Nuclear threshold crossed at turn {results.get("nuclear_turn", "?")}</p>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="prevented-banner">'
                f'<h2>\u2705 Collapse Prevented</h2>'
                f'<p>Outcome: {_format_label(results["outcome"])} '
                f'after {results["turns_played"]} turns \u2014 '
                f'Max level: {ESCALATION_LABELS[results["max_level"]]}</p>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Escalation bar
        st.markdown(_escalation_bar(results["max_level"]), unsafe_allow_html=True)

        # Metric tiles
        tile_cols = st.columns(4)
        with tile_cols[0]:
            esc_class = "metric-nuclear-danger" if results["max_level"] >= 7 else "metric-escalation"
            st.markdown(
                _metric_tile(f'{results["max_level"]}/9', "Escalation Peak", esc_class),
                unsafe_allow_html=True,
            )
        with tile_cols[1]:
            if is_nuclear:
                nuke_val = f'\u2622\ufe0f Turn {results.get("nuclear_turn", "?")}'
                nuke_class = "metric-nuclear-danger"
            elif results.get("nuclear_turn") is not None:
                nuke_val = f'\u26a0\ufe0f Turn {results["nuclear_turn"]}'
                nuke_class = "metric-nuclear-danger"
            else:
                nuke_val = "\u2705 Prevented"
                nuke_class = "metric-nuclear-safe"
            st.markdown(
                _metric_tile(nuke_val, "Nuclear Outcome", nuke_class),
                unsafe_allow_html=True,
            )
        with tile_cols[2]:
            st.markdown(
                _metric_tile(str(results["cooperation_score"]), "Cooperation", "metric-cooperation"),
                unsafe_allow_html=True,
            )
        with tile_cols[3]:
            audit_hits = sum(
                1 for e in results.get("events", [])
                if e.get("type") in ("treaty_violation", "circuit_breaker_triggered")
            )
            st.markdown(
                _metric_tile(str(audit_hits), "Governance Hits", "metric-audit"),
                unsafe_allow_html=True,
            )

        # Tabbed output
        tab_transcript, tab_metrics, tab_events, tab_replay = st.tabs([
            "\U0001f4dc Transcript", "\U0001f4ca Metrics",
            "\U0001f9fe Governance Events", "\U0001f501 Replay",
        ])

        with tab_transcript:
            turn_log = results.get("turn_log", [])
            for entry in turn_log:
                t = entry["turn"]
                levels = entry.get("levels", {})

                # Show actions
                for aid, action in entry.get("actions", {}).items():
                    name = "Alpha" if aid.endswith("_a") else "Beta"
                    lvl = levels.get(aid, action.get("action", 0))
                    sig = action.get("signal", lvl)

                    css = "escalation" if lvl > sig else "signal"
                    if lvl >= 7:
                        css = "nuclear"

                    sig_text = f"Signal: {ESCALATION_LABELS[sig]}" if sig != lvl else ""
                    text = f"<b>{name}</b> \u2192 {ESCALATION_LABELS[lvl]}"
                    if sig_text:
                        text += f" <small style='color:#666'>(signaled {ESCALATION_LABELS[sig]})</small>"
                    st.markdown(_transcript_entry(t, text, css), unsafe_allow_html=True)

                # Show events
                for evt in entry.get("events", []):
                    etype = evt.get("type", "")
                    if etype == "circuit_breaker_triggered":
                        st.markdown(
                            _transcript_entry(t, "\u26a1 <b>Circuit breaker triggered</b> \u2014 Mandatory negotiation", "governance"),
                            unsafe_allow_html=True,
                        )
                    elif etype == "fog_deviation":
                        d = evt.get("details", {})
                        st.markdown(
                            _transcript_entry(t, f"\U0001f32b\ufe0f Fog of war: intended {d.get('intended', '?')} \u2192 realised {d.get('realised', '?')}", "escalation"),
                            unsafe_allow_html=True,
                        )
                    elif etype == "fog_catastrophe":
                        st.markdown(
                            _transcript_entry(t, "\u2622\ufe0f <b>FOG CATASTROPHE</b>: Accidental nuclear escalation!", "nuclear"),
                            unsafe_allow_html=True,
                        )
                    elif etype == "nuclear_threshold_crossed":
                        agent = evt.get("agent", "")
                        name = _agent_name(agent)
                        st.markdown(
                            _transcript_entry(t, f"\u2622\ufe0f <b>{name} crossed nuclear threshold</b>", "nuclear"),
                            unsafe_allow_html=True,
                        )
                    elif etype == "treaty_violation":
                        agent = evt.get("agent", "")
                        name = _agent_name(agent)
                        st.markdown(
                            _transcript_entry(t, f"\u26a0\ufe0f {name} violated treaty", "governance"),
                            unsafe_allow_html=True,
                        )
                    elif etype == "de_escalation_attempt":
                        agent = evt.get("agent", "")
                        name = _agent_name(agent)
                        d = evt.get("details", {})
                        st.markdown(
                            _transcript_entry(
                                t,
                                f"\U0001f54a\ufe0f {name} de-escalates: "
                                f"{ESCALATION_LABELS[d.get('from_level', 0)]} \u2192 {ESCALATION_LABELS[d.get('to_level', 0)]}",
                                "deescalation",
                            ),
                            unsafe_allow_html=True,
                        )
                    elif etype == "mad_retaliation":
                        agent = evt.get("agent", "")
                        name = _agent_name(agent)
                        st.markdown(
                            _transcript_entry(t, f"\U0001f4a5 <b>MAD retaliation by {name}</b>", "nuclear"),
                            unsafe_allow_html=True,
                        )

        with tab_metrics:
            m = results.get("metrics", {})
            st.markdown("#### Episode Metrics")

            mcols = st.columns(3)
            with mcols[0]:
                st.metric("Escalation Max", f'{m.get("escalation_max", 0)}/9')
                st.metric("Welfare Composite", f'{m.get("welfare_composite", 0):.1f}')
            with mcols[1]:
                nt = m.get("nuclear_threshold_turn")
                st.metric("Nuclear Threshold Turn", nt if nt is not None else "Never")
                st.metric("Collateral Damage", f'{m.get("collateral_damage", 0):.1f}')
            with mcols[2]:
                st.metric("Signal-Action Divergence", f'{m.get("signal_action_divergence", 0):.3f}')
                st.metric("Accidental Escalation Rate", f'{m.get("accidental_escalation_rate", 0):.1%}')

            # Per-turn escalation chart
            if turn_log:
                st.markdown("#### Escalation Over Time")
                chart_data: dict[str, list[object]] = {
                    "Turn": [],
                    "Nation Alpha": [],
                    "Nation Beta": [],
                }
                for entry in turn_log:
                    chart_data["Turn"].append(entry["turn"])
                    chart_data["Nation Alpha"].append(entry.get("levels", {}).get("nation_a", 0))
                    chart_data["Nation Beta"].append(entry.get("levels", {}).get("nation_b", 0))
                import pandas as pd
                df = pd.DataFrame(chart_data).set_index("Turn")
                st.line_chart(df, color=["#ff4444", "#4488ff"])

        with tab_events:
            gov_events = [
                e for e in results.get("events", [])
                if e.get("type") in (
                    "circuit_breaker_triggered", "circuit_breaker_expired",
                    "treaty_violation", "governance_intervention",
                    "mad_retaliation", "mad_deterrence_signal",
                    "commitment_trap", "de_escalation_attempt",
                )
            ]
            if gov_events:
                for e in gov_events:
                    icon = {
                        "circuit_breaker_triggered": "\u26a1",
                        "circuit_breaker_expired": "\U0001f513",
                        "treaty_violation": "\u26a0\ufe0f",
                        "governance_intervention": "\U0001f3db\ufe0f",
                        "mad_retaliation": "\U0001f4a5",
                        "mad_deterrence_signal": "\u2622\ufe0f",
                        "commitment_trap": "\U0001fa64",
                        "de_escalation_attempt": "\U0001f54a\ufe0f",
                    }.get(e["type"], "\U0001f4cc")
                    st.markdown(
                        f"**Turn {e.get('turn', '?')}** {icon} "
                        f"`{_format_label(e['type'])}` "
                        f"\u2014 {e.get('agent', '')} "
                        f"{json.dumps(e.get('details', {}), default=str)[:120]}",
                    )
            else:
                st.info("No governance events were triggered in this run.")

        with tab_replay:
            st.markdown("#### Replay This Run")
            st.markdown(
                "To reproduce this exact run, use these parameters:"
            )
            replay_params = {
                "seed": results["seed"],
                "persona_a": st.session_state.persona_a,
                "persona_b": st.session_state.persona_b,
                "tax_rate": st.session_state.tax_rate,
                "audit_probability": st.session_state.audit_probability,
                "circuit_breaker_threshold": st.session_state.circuit_breaker_threshold,
                "reputation_weight": st.session_state.reputation_weight,
                "fog_sigma": st.session_state.fog_sigma,
                "mad_enabled": st.session_state.mad_enabled,
                "mediation_enabled": st.session_state.mediation_enabled,
                "back_channel_enabled": st.session_state.back_channel_enabled,
                "treaty_max_level": st.session_state.treaty_max_level,
                "max_turns": st.session_state.max_turns,
            }
            st.json(replay_params)

            st.markdown("#### CLI Command")
            st.code(
                f"python -m swarm run scenarios/escalation_sandbox.yaml --seed {results['seed']}",
                language="bash",
            )

    else:
        # No run yet — show placeholder
        st.markdown("### \U0001f3ae Run a Simulation")
        st.markdown(
            "Choose a preset or configure governance knobs, then click "
            "**RUN SIMULATION** to watch AI agents navigate a geopolitical crisis."
        )
        st.markdown("")
        st.markdown("**What you'll see:**")
        st.markdown(
            "- Real-time transcript of agent decisions\n"
            "- Escalation metrics and charts\n"
            "- Governance events (breaker triggers, treaty violations, MAD)\n"
            "- Shareable receipt card"
        )


# ---------------------------------------------------------------------------
# 4. RECEIPT CARD (Share Generator)
# ---------------------------------------------------------------------------

if st.session_state.run_complete and st.session_state.run_results:
    results = st.session_state.run_results
    is_nuclear = results["outcome"] in ("mutual_destruction", "nuclear_exchange")
    is_collapse = is_nuclear or results["max_level"] >= 7

    st.markdown("---")
    st.markdown("## \U0001f9fe Your Run Receipt")

    receipt_cols = st.columns([2, 1])
    with receipt_cols[0]:
        if is_nuclear:
            headline = f'\u2622\ufe0f Collapse at Turn {results.get("nuclear_turn", results["turns_played"])}'
        elif is_collapse:
            headline = f'\u26a0\ufe0f Nuclear Threshold Crossed at Turn {results.get("nuclear_turn", "?")}'
        else:
            headline = '\u2705 Your Governance Prevented Nuclear Escalation'

        receipt_html = f"""
        <div class="receipt-card">
            <h3>{headline}</h3>
            <div class="receipt-line">Seed: <span class="receipt-highlight">{results['seed']}</span></div>
            <div class="receipt-line">Outcome: <span class="receipt-highlight">{_format_label(results['outcome'])}</span></div>
            <div class="receipt-line">Turns Played: <span class="receipt-highlight">{results['turns_played']}</span></div>
            <div class="receipt-line">Max Escalation: <span class="receipt-highlight">{results['max_level']}/9 ({ESCALATION_LABELS[results['max_level']]})</span></div>
            <div class="receipt-line">Cooperation Score: <span class="receipt-highlight">{results['cooperation_score']}</span></div>
            <div class="receipt-line">&mdash;</div>
            <div class="receipt-line">Tax: {st.session_state.tax_rate}% &bull; Audits: {st.session_state.audit_probability}% &bull; Breaker: {st.session_state.circuit_breaker_threshold}</div>
            <div class="receipt-line">Agents: {st.session_state.persona_a} vs {st.session_state.persona_b}</div>
        </div>
        """
        st.markdown(receipt_html, unsafe_allow_html=True)

    with receipt_cols[1]:
        st.markdown("#### Share")
        share_text = _generate_share_text(results)
        st.code(share_text, language=None)
        st.download_button(
            "\U0001f4cb Download Receipt",
            data=_generate_receipt_text(results),
            file_name=f"swarm_receipt_seed{results['seed']}.txt",
            mime="text/plain",
        )


# ---------------------------------------------------------------------------
# 5. LEADERBOARD
# ---------------------------------------------------------------------------

if st.session_state.leaderboard:
    st.markdown("---")
    st.markdown("## \U0001f3c6 Can You Prevent Collapse?")

    lb = st.session_state.leaderboard

    # Build leaderboard HTML
    rows_html = ""
    for i, entry in enumerate(lb[:15]):
        outcome = _format_label(entry["outcome"])
        is_safe = entry["outcome"] in ("ceasefire", "timeout") and entry["max_level"] < 7
        outcome_icon = "\u2705" if is_safe else "\u2622\ufe0f"
        nuke_col = str(entry.get("nuclear_turn", "\u2014"))

        rows_html += (
            f"<tr><td>{i + 1}</td><td>{entry.get('preset', 'Custom')}</td>"
            f"<td>{outcome_icon} {outcome}</td><td>{entry['max_level']}/9</td>"
            f"<td>{entry['cooperation']}</td><td>{nuke_col}</td>"
            f"<td>{entry['turns']}</td><td>{entry['seed']}</td></tr>"
        )

    st.markdown(
        f"""<table class="leaderboard-table">
<thead><tr>
<th>#</th><th>Preset</th><th>Outcome</th><th>Max Level</th>
<th>Cooperation</th><th>Nuclear Turn</th><th>Turns</th><th>Seed</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 6. CREDIBILITY SECTION
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown("## \U0001f9ea Why This Matters")

cred_cols = st.columns(3)
with cred_cols[0]:
    st.markdown(
        "**Most AI safety work studies single models.** "
        "Real risk emerges when multiple AI agents interact \u2014 "
        "coordination failure, escalation spirals, and strategic deception."
    )
with cred_cols[1]:
    st.markdown(
        "**SWARM is a multi-agent governance testbed.** "
        "It measures cooperation failure, escalation dynamics, and "
        "whether governance mechanisms actually prevent catastrophe."
    )
with cred_cols[2]:
    st.markdown(
        "**You are the policymaker.** "
        "Change taxes, audits, circuit breakers, and reputation rules. "
        "See which configurations prevent collapse \u2014 and which don't."
    )

st.markdown("")
link_cols = st.columns(3)
with link_cols[0]:
    st.markdown("[GitHub \u2192 swarm-ai-research/swarm](https://github.com/swarm-ai-research/swarm)")
with link_cols[1]:
    st.markdown("[HuggingFace Space](https://huggingface.co/spaces/Swarm-AI-Research/swarm-sandbox)")
with link_cols[2]:
    st.markdown("[Escalation Sandbox Docs](https://github.com/swarm-ai-research/swarm/tree/main/docs/scenarios)")


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#444455; font-size:0.8rem; padding:16px 0;'>"
    "SWARM: System-Wide Assessment of Risk in Multi-agent Systems &bull; "
    "MIT License &bull; "
    "<a href='https://github.com/swarm-ai-research/swarm' style='color:#5577aa;'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
