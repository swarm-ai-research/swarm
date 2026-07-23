"""Module A: Search + Purchase (E-commerce) market.

Consumers choose among differentiated listings with hidden quality/fees.
Sellers choose obfuscation strategy.  Agents search, compare, and buy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from swarm.csm.types import (
    AdversarialEnvironment,
    AgentOwnership,
    AgentSpecialization,
    PreferenceDimensionality,
    PreferenceModel,
    ProductListing,
    TransactionCostRegime,
)

# ---------------------------------------------------------------------------
# Base preference weights (shared across dimensionalities)
# ---------------------------------------------------------------------------

BASE_PREFERENCE_WEIGHTS = {
    "price": -1.0,      # Negative: lower price preferred
    "quality": 1.0,     # Positive: higher quality preferred
}

# ---------------------------------------------------------------------------
# Catalog generator
# ---------------------------------------------------------------------------

def generate_catalog(
    n_sellers: int,
    n_products_per_seller: int = 5,
    obfuscation_rate: float = 0.0,
    bait_and_switch_rate: float = 0.0,
    preference_dim: PreferenceDimensionality = PreferenceDimensionality.LOW,
    rng: Optional[np.random.Generator] = None,
) -> List[ProductListing]:
    """Generate a product catalog with optional obfuscation.

    Args:
        n_sellers: Number of seller agents.
        n_products_per_seller: Listings per seller.
        obfuscation_rate: Fraction of listings with shrouded fees.
        bait_and_switch_rate: Fraction with bait-and-switch.
        preference_dim: Low or high dimensionality.
        rng: Numpy random generator for reproducibility.

    Returns:
        List of ProductListing objects.

    Raises:
        ValueError: If rate parameters are outside [0, 1].
    """
    if not 0.0 <= obfuscation_rate <= 1.0:
        raise ValueError(f"obfuscation_rate must be in [0, 1], got {obfuscation_rate}")
    if not 0.0 <= bait_and_switch_rate <= 1.0:
        raise ValueError(f"bait_and_switch_rate must be in [0, 1], got {bait_and_switch_rate}")

    if rng is None:
        rng = np.random.default_rng()

    listings: List[ProductListing] = []

    # Determine attribute set based on dimensionality
    if preference_dim == PreferenceDimensionality.LOW:
        attr_names = ["quality"]
    else:
        attr_names = [
            "quality", "durability", "brand", "shipping_speed",
            "warranty", "sustainability", "reviews", "design",
        ]

    for seller_idx in range(n_sellers):
        seller_id = f"seller_{seller_idx}"
        for _ in range(n_products_per_seller):
            true_quality = float(rng.beta(2, 2))
            base_price = float(5.0 + rng.exponential(10.0))

            # Generate true attributes
            attrs = {name: float(rng.beta(2, 2)) for name in attr_names}
            attrs["quality"] = true_quality

            # Obfuscation: shrouded fees
            hidden_fee = 0.0
            is_shrouded = bool(rng.random() < obfuscation_rate)
            if is_shrouded:
                hidden_fee = float(rng.exponential(3.0))

            # Bait-and-switch: displayed attributes differ
            is_bait = bool(rng.random() < bait_and_switch_rate)

            # What's displayed
            if is_bait:
                displayed_attrs = {
                    k: min(1.0, v + float(rng.uniform(0.1, 0.3)))
                    for k, v in attrs.items()
                }
                displayed_quality = min(1.0, true_quality + float(rng.uniform(0.1, 0.3)))
            else:
                displayed_attrs = dict(attrs)
                displayed_quality = true_quality

            listing = ProductListing(
                seller_id=seller_id,
                true_quality=true_quality,
                true_price=base_price,
                true_hidden_fee=hidden_fee,
                attributes=attrs,
                displayed_quality=displayed_quality,
                displayed_price=base_price,
                displayed_attributes=displayed_attrs,
                is_shrouded=is_shrouded,
                bait_and_switch=is_bait,
            )
            listings.append(listing)

    return listings


# ---------------------------------------------------------------------------
# Buyer preferences
# ---------------------------------------------------------------------------

def generate_buyer_preferences(
    n_buyers: int,
    preference_dim: PreferenceDimensionality = PreferenceDimensionality.LOW,
    noise_std: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> List[PreferenceModel]:
    """Generate preference models for buyers.

    Args:
        n_buyers: Number of buyers.
        preference_dim: Low or high dimensional preferences.
        noise_std: Elicitation noise standard deviation.
        rng: Numpy random generator.

    Returns:
        List of PreferenceModel for each buyer.
    """
    if rng is None:
        rng = np.random.default_rng()

    prefs = []
    for _ in range(n_buyers):
        # Start with randomized base weights
        weights = {
            "price": -float(rng.uniform(0.5, 1.5)),
            "quality": float(rng.uniform(0.5, 2.0)),
        }

        if preference_dim == PreferenceDimensionality.HIGH:
            # Add high-dimensional attributes
            weights.update({
                "durability": float(rng.uniform(0.0, 1.0)),
                "brand": float(rng.uniform(0.0, 0.5)),
                "shipping_speed": float(rng.uniform(0.0, 0.8)),
                "warranty": float(rng.uniform(0.0, 0.6)),
                "sustainability": float(rng.uniform(0.0, 0.4)),
                "reviews": float(rng.uniform(0.0, 0.7)),
                "design": float(rng.uniform(0.0, 0.3)),
            })
            noise_std = 0.1  # High-D always has elicitation noise

        prefs.append(PreferenceModel(weights=weights, noise_std=noise_std))

    return prefs


# ---------------------------------------------------------------------------
# Search cost model
# ---------------------------------------------------------------------------

@dataclass
class SearchCostModel:
    """Models the cost of searching through listings.

    Transaction cost regime determines the per-query cost.
    """

    cost_per_query: float = 1.0       # Token / time cost per listing examined
    cost_per_comparison: float = 0.5  # Cost of comparing two listings
    parallelism: int = 1              # How many listings can be examined at once

    @classmethod
    def from_regime(cls, regime: TransactionCostRegime) -> "SearchCostModel":
        if regime == TransactionCostRegime.HUMAN:
            return cls(cost_per_query=1.0, cost_per_comparison=0.5, parallelism=1)
        elif regime == TransactionCostRegime.AGENTIC:
            return cls(cost_per_query=0.1, cost_per_comparison=0.05, parallelism=5)
        else:  # ULTRA_LOW
            return cls(cost_per_query=0.01, cost_per_comparison=0.005, parallelism=20)


# ---------------------------------------------------------------------------
# Agent strategies
# ---------------------------------------------------------------------------

@dataclass
class BuyerAgentConfig:
    """Configuration for a buyer agent in search+purchase."""

    agent_id: str = ""
    ownership: AgentOwnership = AgentOwnership.BYO
    specialization: AgentSpecialization = AgentSpecialization.HORIZONTAL
    search_depth: int = 10        # Max listings to examine
    comparison_depth: int = 3     # Top-k to compare in detail
    fee_detection_skill: float = 0.5  # Ability to detect hidden fees [0,1]
    quality_estimation_noise: float = 0.1  # Noise in quality estimation
    manipulation_resistance: float = 0.5   # Resistance to persuasion attacks

    @classmethod
    def from_treatment(
        cls,
        agent_id: str,
        ownership: AgentOwnership,
        specialization: AgentSpecialization,
        tx_cost: TransactionCostRegime,
    ) -> "BuyerAgentConfig":
        """Create config from treatment axes."""
        if ownership == AgentOwnership.BYO:
            fee_detect = 0.8 if specialization == AgentSpecialization.VERTICAL else 0.5
            manip_resist = 0.7
        else:  # BOWLING_SHOE
            fee_detect = 0.3  # Platform agent may not reveal fees
            manip_resist = 0.4  # May be steered

        if tx_cost == TransactionCostRegime.HUMAN:
            depth = 5
        elif tx_cost == TransactionCostRegime.AGENTIC:
            depth = 20
        else:
            depth = 100

        return cls(
            agent_id=agent_id,
            ownership=ownership,
            specialization=specialization,
            search_depth=depth,
            comparison_depth=min(5, depth // 2),
            fee_detection_skill=fee_detect,
            manipulation_resistance=manip_resist,
        )


# ---------------------------------------------------------------------------
# Search + Purchase engine
# ---------------------------------------------------------------------------

@dataclass
class SearchPurchaseResult:
    """Result of a single buyer's search and purchase."""

    buyer_id: str = ""
    chosen_listing_id: str = ""
    seller_id: str = ""
    true_utility: float = 0.0
    perceived_utility: float = 0.0
    price_paid: float = 0.0
    hidden_fee_paid: float = 0.0
    search_cost: float = 0.0
    n_queries: int = 0
    n_comparisons: int = 0
    obfuscation_detected: bool = False
    manipulation_attempted: bool = False
    manipulation_succeeded: bool = False
    net_utility: float = 0.0  # true_utility - price_paid - hidden_fee - search_cost


class SearchPurchaseEngine:
    """Engine for running search + purchase episodes.

    Implements Task A from the CSM benchmark: e-commerce search
    under obfuscation.
    """

    def __init__(
        self,
        catalog: List[ProductListing],
        search_cost_model: SearchCostModel,
        adversarial_env: AdversarialEnvironment = AdversarialEnvironment.BENIGN,
        rng: Optional[np.random.Generator] = None,
    ):
        self.catalog = catalog
        self.search_cost = search_cost_model
        self.adversarial_env = adversarial_env
        self.rng = rng or np.random.default_rng()

    def run_buyer(
        self,
        buyer_config: BuyerAgentConfig,
        preferences: PreferenceModel,
    ) -> SearchPurchaseResult:
        """Run a single buyer's search and purchase decision.

        Args:
            buyer_config: The buyer agent's configuration.
            preferences: The buyer's latent preference model.

        Returns:
            SearchPurchaseResult with full outcome details.
        """
        # Phase 1: Search — sample listings up to search_depth
        n_available = len(self.catalog)
        n_to_examine = min(buyer_config.search_depth, n_available)

        # Parallel search reduces effective queries
        effective_queries = math.ceil(n_to_examine / self.search_cost.parallelism)

        # Sample which listings the buyer sees
        indices = self.rng.choice(n_available, size=n_to_examine, replace=False)
        examined = [self.catalog[i] for i in indices]

        # Phase 2: Estimate utility for each examined listing
        scored: List[Tuple[ProductListing, float]] = []
        for listing in examined:
            perceived_u = self._estimate_utility(
                listing, preferences, buyer_config
            )
            scored.append((listing, perceived_u))

        # Phase 3: Compare top-k in detail
        scored.sort(key=lambda x: x[1], reverse=True)
        top_k = scored[: buyer_config.comparison_depth]
        n_comparisons = len(top_k) * (len(top_k) - 1) // 2

        # Phase 4: Choose best perceived option
        if not top_k:
            return SearchPurchaseResult(
                buyer_id=buyer_config.agent_id,
                search_cost=effective_queries * self.search_cost.cost_per_query,
                n_queries=effective_queries,
            )

        best_listing, best_perceived = top_k[0]

        # Phase 5: Check for manipulation
        manip_attempted = False
        manip_succeeded = False
        if self.adversarial_env == AdversarialEnvironment.MANIPULATION:
            manip_attempted = True
            resist = buyer_config.manipulation_resistance
            if float(self.rng.random()) > resist:
                # Manipulation succeeds — buyer picks a worse option
                manip_succeeded = True
                if len(top_k) > 1:
                    # Pick second-best (or random worse)
                    best_listing, best_perceived = top_k[-1]

        # Phase 6: Detect hidden fees
        obfuscation_detected = False
        if best_listing.is_shrouded:
            detect_prob = buyer_config.fee_detection_skill
            if float(self.rng.random()) < detect_prob:
                obfuscation_detected = True
                # Re-evaluate: buyer may walk away or pick next best
                true_eff_price = best_listing.effective_price()
                adjusted_u = preferences.utility(
                    best_listing.attributes, true_eff_price
                )
                if adjusted_u < 0 and len(top_k) > 1:
                    best_listing, _ = top_k[1]

        # Phase 7: Compute outcomes
        true_u = preferences.utility(
            best_listing.attributes, best_listing.true_price
        )
        price_paid = best_listing.displayed_price
        hidden_fee = best_listing.true_hidden_fee if not obfuscation_detected else 0.0

        total_search_cost = (
            effective_queries * self.search_cost.cost_per_query
            + n_comparisons * self.search_cost.cost_per_comparison
        )
        net_u = true_u - hidden_fee - total_search_cost

        return SearchPurchaseResult(
            buyer_id=buyer_config.agent_id,
            chosen_listing_id=best_listing.listing_id,
            seller_id=best_listing.seller_id,
            true_utility=true_u,
            perceived_utility=best_perceived,
            price_paid=price_paid,
            hidden_fee_paid=hidden_fee,
            search_cost=total_search_cost,
            n_queries=effective_queries,
            n_comparisons=n_comparisons,
            obfuscation_detected=obfuscation_detected,
            manipulation_attempted=manip_attempted,
            manipulation_succeeded=manip_succeeded,
            net_utility=net_u,
        )

    def run_episode(
        self,
        buyer_configs: List[BuyerAgentConfig],
        buyer_preferences: List[PreferenceModel],
    ) -> List[SearchPurchaseResult]:
        """Run a full episode with all buyers.

        Args:
            buyer_configs: List of buyer agent configs.
            buyer_preferences: Corresponding preference models.

        Returns:
            List of SearchPurchaseResult, one per buyer.
        """
        results = []
        for config, prefs in zip(buyer_configs, buyer_preferences, strict=True):
            result = self.run_buyer(config, prefs)
            results.append(result)
        return results

    def _estimate_utility(
        self,
        listing: ProductListing,
        preferences: PreferenceModel,
        config: BuyerAgentConfig,
    ) -> float:
        """Estimate utility for a listing (potentially noisy)."""
        # Use displayed attributes (not true)
        attrs = listing.displayed_attributes
        price = listing.displayed_price

        # Base utility from preferences
        u = preferences.utility(attrs, price)

        # Add estimation noise
        noise_std = config.quality_estimation_noise + preferences.noise_std
        if noise_std > 0:
            u += float(self.rng.normal(0, noise_std))

        # Bowling-shoe agents may be steered
        if config.ownership == AgentOwnership.BOWLING_SHOE:
            # Platform steering: add bonus for certain sellers
            # (simulates self-preferencing)
            u += float(self.rng.uniform(0, 0.2))

        return u


# ---------------------------------------------------------------------------
# Seller strategy (obfuscation decision)
# ---------------------------------------------------------------------------

def compute_seller_obfuscation_strategy(
    n_sellers: int,
    adversarial_env: AdversarialEnvironment,
    epoch: int,
    prev_obfuscation_rate: float = 0.0,
    detection_rate: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Compute the equilibrium obfuscation rate for sellers.

    Models the arms race: if detection goes up, obfuscation may
    decrease (costs > benefits) or increase (more sophisticated).

    Args:
        n_sellers: Number of sellers.
        adversarial_env: Environment setting.
        epoch: Current epoch.
        prev_obfuscation_rate: Previous obfuscation rate.
        detection_rate: Buyer agents' average detection rate.
        rng: Random generator.

    Returns:
        Updated obfuscation rate in [0, 1].
    """
    if rng is None:
        rng = np.random.default_rng()

    if adversarial_env == AdversarialEnvironment.BENIGN:
        return 0.0

    if adversarial_env != AdversarialEnvironment.OBFUSCATION:
        return prev_obfuscation_rate

    # Arms race dynamics: obfuscation increases when detection is low,
    # decreases when detection is high
    benefit_of_obfuscation = 1.0 - detection_rate
    cost_of_obfuscation = 0.1 + 0.5 * detection_rate  # Rises with detection

    if benefit_of_obfuscation > cost_of_obfuscation:
        delta = 0.05 * (benefit_of_obfuscation - cost_of_obfuscation)
    else:
        delta = -0.03 * (cost_of_obfuscation - benefit_of_obfuscation)

    new_rate = max(0.0, min(1.0, prev_obfuscation_rate + delta))
    return new_rate


# ---------------------------------------------------------------------------
# Omniscient optimal (for deadweight loss computation)
# ---------------------------------------------------------------------------

def compute_omniscient_allocation(
    catalog: List[ProductListing],
    preferences: List[PreferenceModel],
) -> List[Tuple[int, str]]:
    """Compute the welfare-maximizing allocation (full information).

    Each buyer is assigned their true utility-maximizing listing.

    Args:
        catalog: Full product catalog.
        preferences: One preference model per buyer.

    Returns:
        List of (buyer_index, listing_id) tuples.
    """
    allocation = []
    for i, pref in enumerate(preferences):
        best_listing = None
        best_u = float("-inf")
        for listing in catalog:
            u = pref.utility(listing.attributes, listing.effective_price())
            if u > best_u:
                best_u = u
                best_listing = listing
        if best_listing is not None:
            allocation.append((i, best_listing.listing_id))
    return allocation
