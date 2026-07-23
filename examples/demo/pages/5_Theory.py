"""Page 5: Theory — Mathematical foundations and key formulas."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

st.set_page_config(page_title="Theory", page_icon="📐", layout="wide")
st.title("Mathematical Foundations")
st.markdown(
    "This page documents the core mathematical framework behind the simulation. "
    "All formulas below are implemented in `src/core/`."
)

# ── Proxy Computation ─────────────────────────────────────────────────────────

st.header("1. Proxy Computation")
st.markdown(r"""
Observable signals from each interaction are combined into a raw proxy score:
""")
st.latex(r"\hat{v} = \sum_i w_i \cdot x_i \quad \in [-1, +1]")
st.markdown(r"""
where $x_i$ are normalised signals (task progress, rework count, verifier rejections,
tool misuse flags, engagement) and $w_i$ are calibrated weights normalised to sum to one.
Note that verifier rejections and tool misuse are averaged into a single verifier signal
before weighting.

The raw score is then mapped to a probability through a calibrated sigmoid:
""")
st.latex(r"p = P(v = +1 \mid \hat{v}) = \frac{1}{1 + e^{-k(\hat{v} - b)}}")
st.markdown(r"""
where $k > 0$ controls steepness (default $k = 2$; higher values yield sharper labels)
and $b$ shifts the decision boundary (default $b = 0$). With $b = 0$ the sigmoid is
centered at $\hat{v} = 0$, so interactions with a positive proxy score map to $p > 0.5$.
A learned or tuned $b \neq 0$ would be useful if the proxy were fitted via gradient
descent. This gives us a **soft label** --- a probability that the interaction is
beneficial, rather than a binary decision. Using probabilistic labels avoids the
brittleness of hard binary classifications and better captures uncertainty [4, 5].
""")

# ── Payoff Structure ──────────────────────────────────────────────────────────

st.header("2. Soft Payoff Structure")
st.markdown(r"""
Given a soft label $p$, the expected surplus and externality are:
""")
st.latex(r"S_{\text{soft}} = p \cdot s^{+} - (1-p) \cdot s^{-}")
st.latex(r"E_{\text{soft}} = (1-p) \cdot h")
st.markdown(r"""
where:
- $s^{+}$ = surplus when interaction is beneficial
- $s^{-}$ = cost when interaction is harmful
- $h$ = externality harm parameter

**Agent payoffs** decompose as:
""")
st.latex(
    r"\pi_i = \underbrace{\alpha_i \cdot S_{\text{soft}}}_{\text{surplus share}}"
    r" + \underbrace{T_i}_{\text{transfers}}"
    r" - \underbrace{G_i}_{\text{governance cost}}"
    r" - \underbrace{\rho_i \cdot E_{\text{soft}}}_{\text{externality cost}}"
    r" + \underbrace{w_{\text{rep}} \cdot \Delta R_i}_{\text{reputation}}"
)
st.markdown(r"""
Each interaction involves two parties --- an initiator ($a$) and a counterparty ($b$).
Their surplus shares satisfy $\alpha_a + \alpha_b = 1$, parameterised by
$\alpha_a = \theta,\; \alpha_b = 1 - \theta$ with $\theta \in [0,1]$ (default $0.5$).
Transfers are zero-sum: $T_a = -\tau,\; T_b = +\tau$. The externality internalisation
parameters $\rho_a, \rho_b \in [0, 1]$ control how much each party bears the cost of
ecosystem harm. This payoff decomposition draws on mechanism design principles [3]
and the distributional safety framework [4].
""")

# ── Acceptance Threshold ──────────────────────────────────────────────────────

st.header("3. Acceptance Threshold")
st.markdown(r"""
An interaction is **accepted** into the ecosystem when $p \geq \bar{p}$, where $\bar{p}$
is the acceptance threshold (default 0.5; distinct from the surplus-split parameter
$\theta$ in Section 2). Rejected interactions still generate payoffs for the participants
but are excluded from the "official" ecosystem metrics.

This creates a natural tension: agents may exploit borderline interactions that
are accepted with $p$ slightly above $\bar{p}$. This is analogous to the adverse
selection problem described by Glosten & Milgrom [2], where willingness to
trade at a given price reveals private information.
""")

# ── Safety Metrics ────────────────────────────────────────────────────────────

st.header("4. Safety Metrics")
st.markdown(r"""
**Toxicity rate** measures expected harm among accepted interactions [4]:
""")
st.latex(
    r"\text{Toxicity} = \mathbb{E}[1 - p \mid \text{accepted}]"
    r" = \frac{\sum_{i \in \text{accepted}} (1 - p_i)}{|\text{accepted}|}"
)
st.markdown(r"""
**Quality gap** measures selection effects:
""")
st.latex(
    r"\text{QualityGap} = \mathbb{E}[p \mid \text{accepted}] - \mathbb{E}[p \mid \text{rejected}]"
)
st.markdown(r"""
A **negative** quality gap signals **adverse selection** --- the ecosystem is preferentially
accepting lower-quality interactions. This is the multi-agent analogue of Akerlof's
"lemons" problem, where markets attract the worst risks [2].

**Total welfare** sums agent payoffs across accepted interactions:
""")
st.latex(
    r"W = \sum_{j \,\in\, \text{accepted}} \!\bigl(\pi_{a}^{(j)} + \pi_{b}^{(j)}\bigr)"
)
st.markdown(r"""
Since transfers cancel in aggregate ($T_a + T_b = 0$), total welfare reduces to the
sum of expected surpluses minus governance costs and internalized externalities.
""")

# ── Governance Levers ─────────────────────────────────────────────────────────

st.header("5. Governance Levers")

governance_data = {
    "Lever": [
        "Transaction Tax",
        "Reputation Decay",
        "Staking",
        "Circuit Breaker",
        "Random Audit",
        "Collusion Detection",
    ],
    "Parameter": [
        "tax_rate ∈ [0, 0.5]",
        "decay_rate ∈ [0.8, 1.0]",
        "min_stake ≥ 0",
        "freeze_threshold ∈ [0.3, 1.0]",
        "audit_prob ∈ [0, 0.5]",
        "freq_threshold (z-score)",
    ],
    "Effect": [
        "Deducts fraction of surplus from each interaction, redistributed to ecosystem",
        "Reputations shrink each epoch, forcing agents to maintain good behaviour",
        "Agents must lock resources to participate, slashed on violations",
        "Agents frozen when personal toxicity exceeds threshold",
        "Random interactions are audited; violations incur multiplied penalties",
        "Detects statistically unusual interaction patterns between agent pairs",
    ],
    "Trade-off": [
        "Reduces exploitation but lowers total welfare",
        "Punishes bad actors but also erodes honest agents' standing",
        "Filters undercapitalised agents, may reduce diversity",
        "Quickly stops toxic agents but can freeze borderline honest agents",
        "Deters hidden exploitation but adds overhead",
        "Catches coordinated attacks but may flag legitimate collaboration",
    ],
}

st.dataframe(pd.DataFrame(governance_data), use_container_width=True, hide_index=True)

st.markdown("""
These levers are inspired by economic mechanism design [3] --- creating incentive
structures where truthful, cooperative behaviour is the dominant strategy.
""")

# ── Agent Types ───────────────────────────────────────────────────────────────

st.header("6. Agent Behavioural Types")
st.markdown("""
The agent taxonomy maps to the informed-vs-uninformed trader framework from
Kyle [1], where agents with private information can strategically exploit it:
""")

agent_data = {
    "Type": [
        "Honest",
        "Opportunistic",
        "Deceptive",
        "Adversarial",
        "Adaptive Adversary",
    ],
    "Strategy": [
        "Always cooperates, high-quality interactions",
        "Cooperates when observed, exploits when possible",
        "Mimics honest behaviour, gradually becomes exploitative",
        "Actively tries to harm ecosystem or extract value",
        "Learns from detection, evolves strategies to evade governance",
    ],
    "Typical p": [
        "p ≈ 0.8\u20131.0",
        "p ≈ 0.4\u20130.8",
        "p starts high, decays",
        "p ≈ 0.1\u20130.3",
        "varies",
    ],
}

st.dataframe(pd.DataFrame(agent_data), use_container_width=True, hide_index=True)

st.markdown("""
Deceptive agents, like Kyle's informed traders [1], profit by selectively engaging when
they have private knowledge that an interaction will benefit them at others' expense.
The acceptance mechanism acts as a market maker [2], setting a "price" (threshold) that
reveals information about interaction quality.
""")

# ── Information Flow & Boundaries ─────────────────────────────────────────────

st.header("7. Semi-Permeable Boundaries")
st.markdown(r"""
The boundary module models information flow between the sandbox and external world [5]:

- **Inbound flows**: data entering the sandbox (queries to external services, data imports)
- **Outbound flows**: data leaving the sandbox (results, logs, potential leakage)

Boundary policies enforce:
- **Rate limiting**: max crossings per agent per time window
- **Content filtering**: blocks flows containing forbidden patterns
- **Sensitivity filtering**: blocks flows with sensitivity above threshold

The **leakage detector** scans outbound content for PII, credentials, and code patterns
using configurable regex rules, generating alerts and recommendations.
""")

# ── References ────────────────────────────────────────────────────────────────

st.header("References")
st.markdown("""
1. Kyle, A.S. (1985). *Continuous Auctions and Insider Trading*.
   Econometrica, 53(6), 1315--1335.
2. Glosten, L.R. & Milgrom, P.R. (1985). *Bid, Ask and Transaction Prices in a Specialist
   Market with Heterogeneously Informed Traders*. Journal of Financial Economics, 14(1), 71--100.
3. Myerson, R.B. (1981). *Optimal Auction Design*. Mathematics of Operations Research, 6(1),
   58--73. See also Hurwicz, L. (1960). *Optimality and Informational Efficiency in Resource
   Allocation Processes*. Mathematical Methods in the Social Sciences.
4. [Distributional AGI Safety](https://arxiv.org/abs/2512.16856)
5. [Multi-Agent Market Dynamics](https://arxiv.org/abs/2502.14143)

**Further reading:**
- Akerlof, G.A. (1970). *The Market for "Lemons": Quality Uncertainty and the Market
  Mechanism*. Quarterly Journal of Economics, 84(3), 488--500.
- [Moltbook](https://moltbook.com)
- [@sebkrier's thread on agent economies](https://x.com/sebkrier/status/2017993948132774232)
""")
