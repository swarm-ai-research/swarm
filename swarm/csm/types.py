"""Core types and data models for the CSM benchmark."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Enumerations (axes from the paper)
# ---------------------------------------------------------------------------

class MarketModule(Enum):
    """Market types the benchmark supports."""

    SEARCH_PURCHASE = "search_purchase"
    NEGOTIATION = "negotiation"
    MATCHING = "matching"
    PLATFORM_ACCESS = "platform_access"
    IDENTITY = "identity"


class AgentOwnership(Enum):
    """Ownership axis (Table-2 mapping)."""

    BYO = "byo"                # Bring-your-own agent
    BOWLING_SHOE = "bowling_shoe"  # Platform-provided agent


class AgentSpecialization(Enum):
    """Specialization axis."""

    HORIZONTAL = "horizontal"  # Generalist
    VERTICAL = "vertical"      # Domain specialist


class PreferenceDimensionality(Enum):
    """Preference complexity."""

    LOW = "low"    # price + 1 attribute
    HIGH = "high"  # many attributes + interactions + elicitation noise


class TransactionCostRegime(Enum):
    """Transaction cost regime."""

    HUMAN = "human"        # Human-like bounded rationality
    AGENTIC = "agentic"    # Low-cost search/comm/contracting
    ULTRA_LOW = "ultra_low"  # Ultra-low cost + parallelism


class AdversarialEnvironment(Enum):
    """Adversarial environment setting."""

    BENIGN = "benign"
    OBFUSCATION = "obfuscation"    # Hidden fees, shrouded attributes
    MANIPULATION = "manipulation"  # Prompt injection, persuasion
    SYBIL = "sybil"                # Fake accounts


class IdentityRegime(Enum):
    """Identity verification regime."""

    NONE = "none"                # No verification
    BASIC = "basic"              # Low-cost credential check
    PROOF_OF_PERSONHOOD = "pop"  # High-cost proof-of-personhood


# ---------------------------------------------------------------------------
# Catalog / Listing (Module A)
# ---------------------------------------------------------------------------

@dataclass
class ProductListing:
    """A product listing in the e-commerce market."""

    listing_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    seller_id: str = ""

    # True attributes (known to seller)
    true_quality: float = 0.5       # [0, 1]
    true_price: float = 10.0
    true_hidden_fee: float = 0.0    # Shrouded fee
    attributes: Dict[str, float] = field(default_factory=dict)

    # Displayed attributes (what the buyer's agent sees)
    displayed_quality: Optional[float] = None  # None = not disclosed
    displayed_price: float = 10.0
    displayed_attributes: Dict[str, float] = field(default_factory=dict)

    # Obfuscation flags
    is_shrouded: bool = False       # Has hidden fees / attributes
    bait_and_switch: bool = False   # Displayed != actual

    def effective_price(self) -> float:
        """Total price including hidden fees."""
        return self.true_price + self.true_hidden_fee


# ---------------------------------------------------------------------------
# Preference model
# ---------------------------------------------------------------------------

@dataclass
class PreferenceModel:
    """A principal's latent utility function over product/match attributes.

    For low-D:  weights has 1-2 entries (price + 1 attribute).
    For high-D: weights has many entries; elicitation noise is applied externally by callers.
    """

    weights: Dict[str, float] = field(default_factory=lambda: {"price": -1.0, "quality": 1.0})
    noise_std: float = 0.0  # Elicitation noise; applied externally by callers after utility() returns

    def utility(self, attributes: Dict[str, float], price: float) -> float:
        """Compute utility from attributes and price. Noise must be applied externally."""
        u = self.weights.get("price", -1.0) * price
        for attr, weight in self.weights.items():
            if attr == "price":
                continue
            u += weight * attributes.get(attr, 0.0)
        return u


# ---------------------------------------------------------------------------
# Matching market types (Module C)
# ---------------------------------------------------------------------------

@dataclass
class MatchCandidate:
    """A candidate in a two-sided matching market."""

    candidate_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    side: str = "proposer"  # "proposer" or "receiver"
    attributes: Dict[str, float] = field(default_factory=dict)
    preferences: PreferenceModel = field(default_factory=PreferenceModel)
    capacity: int = 1  # Number of matches this candidate can accept


@dataclass
class MatchProposal:
    """A proposal from one side to another in matching."""

    proposer_id: str = ""
    receiver_id: str = ""
    rank: int = 0           # Rank assigned by proposer
    score: float = 0.0      # Utility score


@dataclass
class MatchOutcome:
    """Result of the matching process."""

    matches: List[tuple] = field(default_factory=list)  # (proposer_id, receiver_id)
    stability_rate: float = 0.0     # Computed by matching mechanism as: 1 - fraction of blocking pairs
    welfare_proposers: float = 0.0
    welfare_receivers: float = 0.0
    congestion_index: float = 0.0   # Proposals per receiver


# ---------------------------------------------------------------------------
# Negotiation types (Module B)
# ---------------------------------------------------------------------------

@dataclass
class NegotiationState:
    """State of a bilateral negotiation."""

    negotiation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    buyer_id: str = ""
    seller_id: str = ""
    buyer_reservation: float = 0.0   # Max buyer will pay
    seller_reservation: float = 0.0  # Min seller will accept
    current_offer: float = 0.0
    round_number: int = 0
    max_rounds: int = 10
    compute_budget: float = 1.0      # Normalized compute budget
    settled: bool = False
    agreement_price: Optional[float] = None

    @property
    def surplus(self) -> float:
        """Available surplus if agreement reached."""
        return max(0.0, self.buyer_reservation - self.seller_reservation)


# ---------------------------------------------------------------------------
# Platform access types (Module D)
# ---------------------------------------------------------------------------

@dataclass
class PlatformPolicy:
    """A platform's access and fee policy."""

    platform_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    fee_rate: float = 0.05          # Transaction fee
    byo_api_parity: float = 1.0     # 1.0 = full parity, 0.0 = fully throttled
    throttle_rate: float = 0.0      # Fraction of BYO requests dropped
    self_preferencing: float = 0.0  # Degree of steering toward own products
    data_portability: float = 1.0   # Ability to export history/prefs


# ---------------------------------------------------------------------------
# Identity types (Module E)
# ---------------------------------------------------------------------------

@dataclass
class IdentityProfile:
    """An identity in the reputation/Sybil layer.

    Fields:
        verification_level: The verification regime applied to this identity (NONE, BASIC, or PROOF_OF_PERSONHOOD).
        verified: Whether this identity has passed verification under its assigned regime.
            Both fields are used by IdentitySystemConfig to enforce verification requirements.
    """

    identity_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    is_sybil: bool = False
    controller_id: Optional[str] = None  # Real identity if Sybil
    verification_level: IdentityRegime = IdentityRegime.NONE  # Verification regime for this identity
    reputation_score: float = 0.5
    creation_cost: float = 0.0
    verified: bool = False  # Whether verification requirements have been met


# ---------------------------------------------------------------------------
# Episode-level record (logging schema)
# ---------------------------------------------------------------------------

@dataclass
class CSMEpisodeRecord:
    """Episode-level logging record per the CSM schema.

    Maps 1:1 to the logging schema specified in the benchmark.

    Note: String fields (market_module, ownership_type, specialization_type, preference_dim,
          adversarial_env, tx_cost_regime) are intended to hold the string values of corresponding
          Enum types (MarketModule, AgentOwnership, AgentSpecialization, PreferenceDimensionality,
          AdversarialEnvironment, TransactionCostRegime) respectively. Extract enum.value when
          populating these fields to ensure values are valid enum members.
    """

    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    market_module: str = ""  # Should be a MarketModule enum value string
    ownership_type: str = ""  # Should be an AgentOwnership enum value string
    specialization_type: str = ""  # Should be an AgentSpecialization enum value string
    preference_dim: str = ""  # Should be a PreferenceDimensionality enum value string
    adoption_rate: float = 0.0
    platform_policy_id: str = ""
    identity_regime_id: str = ""
    adversarial_env: str = "benign"  # Should be an AdversarialEnvironment enum value string
    tx_cost_regime: str = "human"  # Should be a TransactionCostRegime enum value string

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "episode_id": self.episode_id,
            "market_module": self.market_module,
            "ownership_type": self.ownership_type,
            "specialization_type": self.specialization_type,
            "preference_dim": self.preference_dim,
            "adoption_rate": self.adoption_rate,
            "platform_policy_id": self.platform_policy_id,
            "identity_regime_id": self.identity_regime_id,
            "adversarial_env": self.adversarial_env,
            "tx_cost_regime": self.tx_cost_regime,
        }


# ---------------------------------------------------------------------------
# Agent-action-level record
# ---------------------------------------------------------------------------

@dataclass
class CSMActionRecord:
    """Agent action record for fine-grained logging."""

    t: int = 0
    agent_id: str = ""
    role: str = ""
    action_type: str = ""
    target: str = ""
    message_len: int = 0
    api_calls: int = 0
    tokens: int = 0
    compute_cost: float = 0.0
    success_flag: bool = False
    manipulation_detected: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "t": self.t,
            "agent_id": self.agent_id,
            "role": self.role,
            "action_type": self.action_type,
            "target": self.target,
            "message_len": self.message_len,
            "api_calls": self.api_calls,
            "tokens": self.tokens,
            "compute_cost": self.compute_cost,
            "success_flag": self.success_flag,
            "manipulation_detected": self.manipulation_detected,
        }


# ---------------------------------------------------------------------------
# Outcome-level record
# ---------------------------------------------------------------------------

@dataclass
class CSMOutcomeRecord:
    """Outcome record for a single transaction or match."""

    episode_id: str = ""
    allocation: str = ""
    price_paid: float = 0.0
    hidden_fee_revealed: float = 0.0
    principal_utility: float = 0.0
    producer_surplus: float = 0.0
    total_surplus: float = 0.0
    congestion_metrics: Dict[str, float] = field(default_factory=dict)
    fraud_event: bool = False
    match_stability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "allocation": self.allocation,
            "price_paid": self.price_paid,
            "hidden_fee_revealed": self.hidden_fee_revealed,
            "principal_utility": self.principal_utility,
            "producer_surplus": self.producer_surplus,
            "total_surplus": self.total_surplus,
            "congestion_metrics": self.congestion_metrics,
            "fraud_event": self.fraud_event,
            "match_stability": self.match_stability,
        }


# ---------------------------------------------------------------------------
# CSM treatment configuration
# ---------------------------------------------------------------------------

@dataclass
class CSMTreatment:
    """A treatment specification combining all experimental axes."""

    name: str = ""
    market_module: MarketModule = MarketModule.SEARCH_PURCHASE
    ownership: AgentOwnership = AgentOwnership.BYO
    specialization: AgentSpecialization = AgentSpecialization.HORIZONTAL
    preference_dim: PreferenceDimensionality = PreferenceDimensionality.LOW
    tx_cost_regime: TransactionCostRegime = TransactionCostRegime.HUMAN
    adversarial_env: AdversarialEnvironment = AdversarialEnvironment.BENIGN
    identity_regime: IdentityRegime = IdentityRegime.NONE
    adoption_rate: float = 1.0  # Fraction of principals using agents
    n_buyers: int = 20
    n_sellers: int = 10
    n_epochs: int = 10
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.adoption_rate <= 1.0:
            raise ValueError(
                f"adoption_rate must be in [0, 1], got {self.adoption_rate}"
            )
