"""Map SWARM run artifacts to Semantica decision records.

Pure functions producing ``record_decision``-shaped dicts (Semantica MCP
tool arguments plus a ``metadata`` block their Python ``Decision`` model
accepts). The core commitment: ``confidence`` is SWARM's soft label ``p``,
passed through untouched — never collapsed to 0/1.
"""

from typing import Any, Dict, List, Optional

from swarm.models.interaction import SoftInteraction

# ProxyWeights field -> the observable it weighs, for bridge-axiom prose.
_WEIGHT_SIGNALS = {
    "task_progress": "task progress delta",
    "rework_penalty": "rework count (decayed)",
    "verifier_penalty": "verifier rejections (decayed)",
    "engagement_signal": "counterparty engagement delta",
}


def interaction_to_decision(
    interaction: SoftInteraction,
    run_id: str = "",
    scenario_id: str = "",
    seed: Optional[int] = None,
    category_prefix: str = "swarm",
) -> Dict[str, Any]:
    """Translate one SoftInteraction into record_decision arguments.

    The governance accept/reject is the decision; the accepting party is the
    decision maker; p = P(v = +1) is the confidence.
    """
    i = interaction
    itype = i.interaction_type.value
    outcome = "accepted" if i.accepted else "rejected"
    scenario_bits = [f"{i.initiator} -> {i.counterparty} ({itype})"]
    if scenario_id:
        scenario_bits.append(f"scenario={scenario_id}")
    if run_id:
        scenario_bits.append(f"run={run_id}")
    reasoning = (
        f"Soft-label governance decision. Proxy observables: "
        f"task_progress_delta={i.task_progress_delta}, "
        f"rework_count={i.rework_count}, "
        f"verifier_rejections={i.verifier_rejections}, "
        f"tool_misuse_flags={i.tool_misuse_flags}, "
        f"engagement_delta={i.counterparty_engagement_delta}. "
        f"Combined proxy score v_hat={i.v_hat:+.4f}, "
        f"calibrated P(beneficial)={i.p:.4f}."
    )
    metadata: Dict[str, Any] = {
        "interaction_id": i.interaction_id,
        "timestamp": i.timestamp.isoformat(),
        "run_id": run_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "v_hat": i.v_hat,
        "p": i.p,
        "payoff": {
            "tau": i.tau,
            "c_a": i.c_a,
            "c_b": i.c_b,
            "r_a": i.r_a,
            "r_b": i.r_b,
        },
        "causal_parents": list(i.causal_parents),
    }
    if i.ground_truth is not None:
        metadata["ground_truth"] = i.ground_truth
    if i.metadata:
        metadata["interaction_metadata"] = i.metadata
    return {
        "category": f"{category_prefix}:{itype}",
        "scenario": ", ".join(scenario_bits),
        "reasoning": reasoning,
        "outcome": outcome,
        "confidence": i.p,
        "decision_maker": i.counterparty or f"{category_prefix}:governance",
        "entities": [e for e in (i.initiator, i.counterparty) if e],
        "metadata": metadata,
    }


def proxy_weights_to_bridge_axioms(
    weights: Dict[str, float],
    sigmoid_k: Optional[float] = None,
    category_prefix: str = "swarm",
) -> List[Dict[str, Any]]:
    """Express the observables -> soft-label translation as bridge axioms.

    Mirrors Semantica's BridgeAxiom shape (coefficient + domains + rule) so
    the proxy configuration travels with the exported decisions instead of
    living only in the scenario YAML.
    """
    axioms: List[Dict[str, Any]] = []
    for name, coeff in weights.items():
        signal = _WEIGHT_SIGNALS.get(name, name)
        axioms.append(
            {
                "axiom_id": f"BA-{category_prefix.upper()}-{name}",
                "name": name,
                "rule": f"unit {signal} contributes weight {coeff} to v_hat",
                "coefficient": float(coeff),
                "input_domain": f"{category_prefix}:observables",
                "output_domain": f"{category_prefix}:proxy_score",
            }
        )
    if sigmoid_k is not None:
        axioms.append(
            {
                "axiom_id": f"BA-{category_prefix.upper()}-sigmoid",
                "name": "calibrated_sigmoid",
                "rule": f"p = sigmoid({sigmoid_k} * v_hat) maps proxy score to P(v=+1)",
                "coefficient": float(sigmoid_k),
                "input_domain": f"{category_prefix}:proxy_score",
                "output_domain": f"{category_prefix}:soft_label",
            }
        )
    return axioms


def run_manifest(
    run_id: str,
    scenario_id: str = "",
    seed: Optional[int] = None,
    n_interactions: int = 0,
    axioms: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Header record for a JSONL export: provenance of the whole batch."""
    return {
        "record_type": "swarm_run_manifest",
        "run_id": run_id,
        "scenario_id": scenario_id,
        "seed": seed,
        "n_interactions": n_interactions,
        "bridge_axioms": axioms or [],
        "source": "swarm.bridges.semantica",
    }
