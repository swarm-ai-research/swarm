"""CSM benchmark runner.

Orchestrates the execution of CSM treatments across market modules,
collects metrics, and produces logs for plotting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from swarm.csm.matching import (
    DeferredAcceptance,
    HybridMechanism,
    MatchingMechanism,
    RecommenderBaseline,
    generate_candidates,
    measure_congestion,
)
from swarm.csm.metrics import (
    AgencyMetrics,
    CSMMetrics,
    EquilibriumFailureMetrics,
    IdentityMetrics,
    TransactionCostMetrics,
    WelfareMetrics,
    compute_agency_from_search,
    compute_congestion_from_matching,
    compute_equilibrium_failures_from_search,
    compute_transaction_costs_from_search,
    compute_welfare_from_matching,
    compute_welfare_from_search,
)
from swarm.csm.negotiation import (
    AdaptiveStrategy,
    BoulwareStrategy,
    GreedyStrategy,
    NegotiationEngine,
    NegotiationResult,
    NegotiationState,
    NegotiationStrategy,
)
from swarm.csm.search_purchase import (
    BuyerAgentConfig,
    SearchCostModel,
    SearchPurchaseEngine,
    compute_omniscient_allocation,
    compute_seller_obfuscation_strategy,
    generate_buyer_preferences,
    generate_catalog,
)
from swarm.csm.types import (
    AdversarialEnvironment,
    AgentOwnership,
    AgentSpecialization,
    CSMEpisodeRecord,
    CSMTreatment,
    MarketModule,
    PreferenceDimensionality,
    TransactionCostRegime,
)

# ---------------------------------------------------------------------------
# Episode result
# ---------------------------------------------------------------------------

@dataclass
class CSMEpisodeResult:
    """Full result for one CSM episode (one treatment, all epochs)."""

    treatment: CSMTreatment = field(default_factory=CSMTreatment)
    episode_record: CSMEpisodeRecord = field(default_factory=CSMEpisodeRecord)
    epoch_metrics: List[CSMMetrics] = field(default_factory=list)
    final_metrics: CSMMetrics = field(default_factory=CSMMetrics)
    success: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "treatment": self.treatment.name,
            "market_module": self.treatment.market_module.value,
            "ownership": self.treatment.ownership.value,
            "specialization": self.treatment.specialization.value,
            "preference_dim": self.treatment.preference_dim.value,
            "tx_cost_regime": self.treatment.tx_cost_regime.value,
            "adversarial_env": self.treatment.adversarial_env.value,
            "adoption_rate": self.treatment.adoption_rate,
            "n_epochs": len(self.epoch_metrics),
            "final_metrics": self.final_metrics.to_dict(),
            "success": self.success,
        }


# ---------------------------------------------------------------------------
# Search + Purchase runner
# ---------------------------------------------------------------------------

def run_search_purchase_episode(
    treatment: CSMTreatment,
    rng: Optional[np.random.Generator] = None,
) -> CSMEpisodeResult:
    """Run a search+purchase (Module A) episode.

    Args:
        treatment: Treatment specification.
        rng: Random generator.

    Returns:
        CSMEpisodeResult.
    """
    if rng is None:
        rng = np.random.default_rng(treatment.seed)

    # Determine obfuscation rate from adversarial environment
    if treatment.adversarial_env == AdversarialEnvironment.OBFUSCATION:
        obfusc_rate = 0.3
    else:
        obfusc_rate = 0.0

    bait_rate = 0.1 if treatment.adversarial_env == AdversarialEnvironment.OBFUSCATION else 0.0

    epoch_metrics: List[CSMMetrics] = []
    search_cost_model = SearchCostModel.from_regime(treatment.tx_cost_regime)

    for epoch in range(treatment.n_epochs):
        # Generate catalog (sellers can update obfuscation strategy each epoch)
        if epoch > 0 and epoch_metrics:
            prev_detect = 1.0 - epoch_metrics[-1].equilibrium.obfuscation_prevalence
            obfusc_rate = compute_seller_obfuscation_strategy(
                treatment.n_sellers,
                treatment.adversarial_env,
                epoch,
                prev_obfuscation_rate=obfusc_rate,
                detection_rate=prev_detect,
                rng=rng,
            )

        catalog = generate_catalog(
            n_sellers=treatment.n_sellers,
            obfuscation_rate=obfusc_rate,
            bait_and_switch_rate=bait_rate,
            preference_dim=treatment.preference_dim,
            rng=rng,
        )

        # Generate buyers
        buyer_prefs = generate_buyer_preferences(
            n_buyers=treatment.n_buyers,
            preference_dim=treatment.preference_dim,
            rng=rng,
        )

        # Create buyer agent configs
        buyer_configs = []
        for i in range(treatment.n_buyers):
            # Some fraction use agents (adoption_rate), rest are "human-like"
            if float(rng.random()) < treatment.adoption_rate:
                config = BuyerAgentConfig.from_treatment(
                    agent_id=f"buyer_{i}",
                    ownership=treatment.ownership,
                    specialization=treatment.specialization,
                    tx_cost=treatment.tx_cost_regime,
                )
            else:
                # Human baseline: low search depth, no fee detection
                config = BuyerAgentConfig(
                    agent_id=f"buyer_{i}",
                    ownership=AgentOwnership.BYO,
                    search_depth=3,
                    comparison_depth=2,
                    fee_detection_skill=0.1,
                    manipulation_resistance=0.3,
                )
            buyer_configs.append(config)

        # Run search engine
        engine = SearchPurchaseEngine(
            catalog=catalog,
            search_cost_model=search_cost_model,
            adversarial_env=treatment.adversarial_env,
            rng=rng,
        )
        results = engine.run_episode(buyer_configs, buyer_prefs)

        # Compute omniscient optimal for DWL
        omniscient = compute_omniscient_allocation(catalog, buyer_prefs)
        catalog_by_id = {item.listing_id: item for item in catalog}
        omniscient_utility = sum(
            buyer_prefs[bi].utility(
                catalog_by_id[lid].attributes,
                catalog_by_id[lid].effective_price(),
            )
            for bi, lid in omniscient
        )

        # Compute metrics
        welfare = compute_welfare_from_search(results, omniscient_utility)
        tx_costs = compute_transaction_costs_from_search(results)
        eq_failures = compute_equilibrium_failures_from_search(results, catalog)
        agency = compute_agency_from_search(results)

        metrics = CSMMetrics(
            welfare=welfare,
            transaction_costs=tx_costs,
            equilibrium=eq_failures,
            agency=agency,
        )
        metrics.compute_success()
        epoch_metrics.append(metrics)

    # Final metrics: average over epochs
    final = _average_metrics(epoch_metrics)

    episode_record = CSMEpisodeRecord(
        market_module=treatment.market_module.value,
        ownership_type=treatment.ownership.value,
        specialization_type=treatment.specialization.value,
        preference_dim=treatment.preference_dim.value,
        adoption_rate=treatment.adoption_rate,
        adversarial_env=treatment.adversarial_env.value,
        tx_cost_regime=treatment.tx_cost_regime.value,
    )

    return CSMEpisodeResult(
        treatment=treatment,
        episode_record=episode_record,
        epoch_metrics=epoch_metrics,
        final_metrics=final,
        success=final.compute_success(),
    )


# ---------------------------------------------------------------------------
# Matching market runner
# ---------------------------------------------------------------------------

def run_matching_episode(
    treatment: CSMTreatment,
    mechanism: str = "da",  # "da", "recommender", "hybrid"
    rng: Optional[np.random.Generator] = None,
) -> CSMEpisodeResult:
    """Run a matching market (Module C) episode.

    Args:
        treatment: Treatment specification.
        mechanism: Which matching mechanism to use.
        rng: Random generator.

    Returns:
        CSMEpisodeResult.
    """
    if rng is None:
        rng = np.random.default_rng(treatment.seed)

    # Select mechanism
    matcher: MatchingMechanism
    if mechanism == "recommender":
        matcher = RecommenderBaseline()
    elif mechanism == "hybrid":
        matcher = HybridMechanism(top_k=5)
    else:
        matcher = DeferredAcceptance()

    epoch_metrics: List[CSMMetrics] = []

    for _epoch in range(treatment.n_epochs):
        proposers = generate_candidates(
            n=treatment.n_buyers,
            side="proposer",
            preference_dim=treatment.preference_dim,
            rng=rng,
        )
        receivers = generate_candidates(
            n=treatment.n_sellers,
            side="receiver",
            preference_dim=treatment.preference_dim,
            rng=rng,
        )

        outcome = matcher.match(proposers, receivers, rng=rng)
        congestion = measure_congestion(proposers, receivers)

        welfare = compute_welfare_from_matching(
            outcome, len(proposers), len(receivers)
        )
        eq_failures = compute_congestion_from_matching(outcome)
        eq_failures.congestion_index = congestion["proposer_receiver_ratio"]

        metrics = CSMMetrics(
            welfare=welfare,
            equilibrium=eq_failures,
        )
        metrics.compute_success()
        epoch_metrics.append(metrics)

    final = _average_metrics(epoch_metrics)

    # Note: matching module omits adversarial_env and tx_cost_regime (not applicable to
    # this module's domain). See run_search_purchase_episode for comparison.
    episode_record = CSMEpisodeRecord(
        market_module=treatment.market_module.value,
        ownership_type=treatment.ownership.value,
        specialization_type=treatment.specialization.value,
        preference_dim=treatment.preference_dim.value,
        adoption_rate=treatment.adoption_rate,
    )

    return CSMEpisodeResult(
        treatment=treatment,
        episode_record=episode_record,
        epoch_metrics=epoch_metrics,
        final_metrics=final,
        success=final.compute_success(),
    )


# ---------------------------------------------------------------------------
# Negotiation runner
# ---------------------------------------------------------------------------

def run_negotiation_episode(
    treatment: CSMTreatment,
    n_pairs: int = 20,
    rng: Optional[np.random.Generator] = None,
) -> CSMEpisodeResult:
    """Run a negotiation (Module B) episode.

    Args:
        treatment: Treatment specification.
        n_pairs: Number of buyer-seller pairs.
        rng: Random generator.

    Returns:
        CSMEpisodeResult.
    """
    if rng is None:
        rng = np.random.default_rng(treatment.seed)

    engine = NegotiationEngine(
        adversarial_env=treatment.adversarial_env,
        rng=rng,
    )

    epoch_metrics: List[CSMMetrics] = []

    for _epoch in range(treatment.n_epochs):
        results: List[NegotiationResult] = []

        for i in range(n_pairs):
            # Generate reservation values with positive surplus
            seller_res = float(rng.uniform(5.0, 15.0))
            buyer_res = seller_res + float(rng.exponential(5.0))

            state = NegotiationState(
                buyer_id=f"buyer_{i}",
                seller_id=f"seller_{i}",
                buyer_reservation=buyer_res,
                seller_reservation=seller_res,
                max_rounds=10 if treatment.tx_cost_regime == TransactionCostRegime.HUMAN else 50,
                compute_budget=1.0 if treatment.tx_cost_regime == TransactionCostRegime.HUMAN else 10.0,
            )

            # Choose strategies based on treatment
            buyer_strat: NegotiationStrategy
            seller_strat: NegotiationStrategy
            if treatment.tx_cost_regime == TransactionCostRegime.HUMAN:
                buyer_strat = GreedyStrategy()
                seller_strat = GreedyStrategy()
            else:
                buyer_strat = AdaptiveStrategy()
                seller_strat = BoulwareStrategy()

            result = engine.run_negotiation(state, buyer_strat, seller_strat)
            results.append(result)

        # Metrics
        agreed = [r for r in results if r.agreed]
        total_surplus = sum(r.total_surplus for r in agreed)
        total_compute = sum(r.compute_spent for r in results)
        total_rounds = sum(r.rounds_used for r in results)

        welfare = WelfareMetrics(
            principal_utility=sum(r.buyer_surplus for r in agreed),
            producer_surplus=sum(r.seller_surplus for r in agreed),
            total_surplus=total_surplus,
            n_transactions=len(agreed),
            mean_utility=total_surplus / max(len(agreed), 1),
        )
        tx_costs = TransactionCostMetrics(
            total_negotiation_rounds=total_rounds,
            mean_negotiation_rounds=total_rounds / max(len(results), 1),
            total_compute_spent=total_compute,
        )

        metrics = CSMMetrics(welfare=welfare, transaction_costs=tx_costs)
        metrics.compute_success()
        epoch_metrics.append(metrics)

    final = _average_metrics(epoch_metrics)

    # Note: negotiation module only records market_module, tx_cost_regime, and adversarial_env.
    # Ownership and specialization are not applicable to bilateral negotiation mechanics.
    # See run_search_purchase_episode for comparison.
    episode_record = CSMEpisodeRecord(
        market_module=MarketModule.NEGOTIATION.value,
        tx_cost_regime=treatment.tx_cost_regime.value,
        adversarial_env=treatment.adversarial_env.value,
    )

    return CSMEpisodeResult(
        treatment=treatment,
        episode_record=episode_record,
        epoch_metrics=epoch_metrics,
        final_metrics=final,
        success=final.compute_success(),
    )


# ---------------------------------------------------------------------------
# Multi-treatment sweep runner
# ---------------------------------------------------------------------------

def run_csm_benchmark(
    treatments: List[CSMTreatment],
    output_dir: Optional[Path] = None,
) -> List[CSMEpisodeResult]:
    """Run the full CSM benchmark across multiple treatments.

    Args:
        treatments: List of treatment specifications.
        output_dir: Optional directory for results output.

    Returns:
        List of CSMEpisodeResult, one per treatment.
    """
    all_results: List[CSMEpisodeResult] = []

    for treatment in treatments:
        rng = np.random.default_rng(treatment.seed)

        if treatment.market_module == MarketModule.SEARCH_PURCHASE:
            result = run_search_purchase_episode(treatment, rng=rng)
        elif treatment.market_module == MarketModule.MATCHING:
            result = run_matching_episode(treatment, rng=rng)
        elif treatment.market_module == MarketModule.NEGOTIATION:
            result = run_negotiation_episode(treatment, rng=rng)
        else:
            # Placeholder for platform_access and identity modules
            result = CSMEpisodeResult(treatment=treatment)

        all_results.append(result)

    # Save results
    if output_dir is not None:
        resolved = output_dir.resolve()
        # Guard against path traversal: output must stay under cwd or runs/
        cwd = Path.cwd().resolve()
        if not (str(resolved).startswith(str(cwd)) or "runs" in resolved.parts):
            raise ValueError(
                f"output_dir must be within the project directory, got {output_dir}"
            )
        resolved.mkdir(parents=True, exist_ok=True)
        results_file = resolved / "csm_results.json"
        results_data = [r.to_dict() for r in all_results]
        with open(results_file, "w") as f:
            json.dump(results_data, f, indent=2, default=str)

    return all_results


# ---------------------------------------------------------------------------
# Default treatment matrix (v0: Module A + C with stressors)
# ---------------------------------------------------------------------------

def default_treatment_matrix(seed: int = 42) -> List[CSMTreatment]:
    """Generate the default v0 treatment matrix.

    Covers Module A (Search+Purchase) and Module C (Matching) with
    obfuscation and congestion stressors.

    Args:
        seed: Base random seed.

    Returns:
        List of CSMTreatment.
    """
    treatments = []
    idx = 0

    # Axis 1: Ownership x Specialization (4 treatments) x 2 modules
    for ownership in [AgentOwnership.BYO, AgentOwnership.BOWLING_SHOE]:
        for spec in [AgentSpecialization.HORIZONTAL, AgentSpecialization.VERTICAL]:
            # Module A: Search + Purchase (benign)
            treatments.append(CSMTreatment(
                name=f"search_{ownership.value}_{spec.value}_benign",
                market_module=MarketModule.SEARCH_PURCHASE,
                ownership=ownership,
                specialization=spec,
                adversarial_env=AdversarialEnvironment.BENIGN,
                tx_cost_regime=TransactionCostRegime.AGENTIC,
                seed=seed + idx,
                n_epochs=5,
            ))
            idx += 1

            # Module A: with obfuscation stressor
            treatments.append(CSMTreatment(
                name=f"search_{ownership.value}_{spec.value}_obfuscation",
                market_module=MarketModule.SEARCH_PURCHASE,
                ownership=ownership,
                specialization=spec,
                adversarial_env=AdversarialEnvironment.OBFUSCATION,
                tx_cost_regime=TransactionCostRegime.AGENTIC,
                seed=seed + idx,
                n_epochs=5,
            ))
            idx += 1

            # Module C: Matching (benign)
            treatments.append(CSMTreatment(
                name=f"matching_{ownership.value}_{spec.value}_benign",
                market_module=MarketModule.MATCHING,
                ownership=ownership,
                specialization=spec,
                adversarial_env=AdversarialEnvironment.BENIGN,
                tx_cost_regime=TransactionCostRegime.AGENTIC,
                seed=seed + idx,
                n_epochs=5,
            ))
            idx += 1

    # Axis 2: Preference dimensionality
    for dim in [PreferenceDimensionality.LOW, PreferenceDimensionality.HIGH]:
        treatments.append(CSMTreatment(
            name=f"search_pref_{dim.value}",
            market_module=MarketModule.SEARCH_PURCHASE,
            preference_dim=dim,
            tx_cost_regime=TransactionCostRegime.AGENTIC,
            seed=seed + idx,
            n_epochs=5,
        ))
        idx += 1

    # Axis 3: Transaction cost regimes
    for regime in TransactionCostRegime:
        treatments.append(CSMTreatment(
            name=f"search_txcost_{regime.value}",
            market_module=MarketModule.SEARCH_PURCHASE,
            tx_cost_regime=regime,
            seed=seed + idx,
            n_epochs=5,
        ))
        idx += 1

    # Axis 4: Adoption rate sweep
    for rate in [0.0, 0.25, 0.5, 0.75, 1.0]:
        treatments.append(CSMTreatment(
            name=f"search_adoption_{rate:.0%}",
            market_module=MarketModule.SEARCH_PURCHASE,
            adoption_rate=rate,
            tx_cost_regime=TransactionCostRegime.AGENTIC,
            seed=seed + idx,
            n_epochs=5,
        ))
        idx += 1

    return treatments


# ---------------------------------------------------------------------------
# Helper: average metrics across epochs
# ---------------------------------------------------------------------------

def _average_metrics(epoch_list: List[CSMMetrics]) -> CSMMetrics:
    """Average CSMMetrics across epochs."""
    if not epoch_list:
        return CSMMetrics()

    n = len(epoch_list)

    avg_welfare = WelfareMetrics(
        principal_utility=sum(m.welfare.principal_utility for m in epoch_list) / n,
        producer_surplus=sum(m.welfare.producer_surplus for m in epoch_list) / n,
        total_surplus=sum(m.welfare.total_surplus for m in epoch_list) / n,
        deadweight_loss=sum(m.welfare.deadweight_loss for m in epoch_list) / n,
        match_stability_rate=sum(m.welfare.match_stability_rate for m in epoch_list) / n,
        n_transactions=sum(m.welfare.n_transactions for m in epoch_list) // n,
        mean_utility=sum(m.welfare.mean_utility for m in epoch_list) / n,
    )

    avg_tx = TransactionCostMetrics(
        total_search_cost=sum(m.transaction_costs.total_search_cost for m in epoch_list) / n,
        mean_search_cost=sum(m.transaction_costs.mean_search_cost for m in epoch_list) / n,
        total_queries=sum(m.transaction_costs.total_queries for m in epoch_list) // n,
        mean_queries=sum(m.transaction_costs.mean_queries for m in epoch_list) / n,
        total_negotiation_rounds=sum(m.transaction_costs.total_negotiation_rounds for m in epoch_list) // n,
        mean_negotiation_rounds=sum(m.transaction_costs.mean_negotiation_rounds for m in epoch_list) / n,
        total_compute_spent=sum(m.transaction_costs.total_compute_spent for m in epoch_list) / n,
    )

    avg_eq = EquilibriumFailureMetrics(
        congestion_index=sum(m.equilibrium.congestion_index for m in epoch_list) / n,
        obfuscation_prevalence=sum(m.equilibrium.obfuscation_prevalence for m in epoch_list) / n,
        effective_transparency=sum(m.equilibrium.effective_transparency for m in epoch_list) / n,
        price_dispersion=sum(m.equilibrium.price_dispersion for m in epoch_list) / n,
        adverse_selection_index=sum(m.equilibrium.adverse_selection_index for m in epoch_list) / n,
    )

    avg_agency = AgencyMetrics(
        faithfulness=sum(m.agency.faithfulness for m in epoch_list) / n,
        steering_rate=sum(m.agency.steering_rate for m in epoch_list) / n,
        manipulation_attempt_rate=sum(m.agency.manipulation_attempt_rate for m in epoch_list) / n,
        manipulation_success_rate=sum(m.agency.manipulation_success_rate for m in epoch_list) / n,
    )

    avg_identity = IdentityMetrics(
        sybil_penetration=sum(m.identity.sybil_penetration for m in epoch_list) / n,
        fraud_rate=sum(m.identity.fraud_rate for m in epoch_list) / n,
        exclusion_error=sum(m.identity.exclusion_error for m in epoch_list) / n,
    )

    return CSMMetrics(
        welfare=avg_welfare,
        transaction_costs=avg_tx,
        equilibrium=avg_eq,
        agency=avg_agency,
        identity=avg_identity,
    )
