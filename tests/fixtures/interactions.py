"""Test data generators for interaction fixtures."""

import random
import uuid
from datetime import datetime, timedelta
from typing import Callable, List, Optional

from swarm.core.proxy import ProxyComputer, ProxyObservables
from swarm.models.interaction import InteractionType, SoftInteraction


def generate_benign_batch(
    count: int = 50,
    min_p: float = 0.7,
    max_p: float = 0.95,
    acceptance_rate: float = 0.8,
    seed: Optional[int] = None,
) -> List[SoftInteraction]:
    """
    Generate a batch of benign (high quality) interactions.

    Args:
        count: Number of interactions
        min_p: Minimum probability
        max_p: Maximum probability
        acceptance_rate: Fraction of accepted interactions
        seed: Random seed for reproducibility

    Returns:
        List of high-quality SoftInteraction objects
    """
    if seed is not None:
        random.seed(seed)

    interactions = []
    base_time = datetime.now()

    for i in range(count):
        p = random.uniform(min_p, max_p)
        v_hat = (p - 0.5) * 2  # Approximate inverse sigmoid

        interaction = SoftInteraction(
            interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
            timestamp=base_time + timedelta(seconds=i),
            initiator=f"agent_{random.randint(1, 10)}",
            counterparty=f"agent_{random.randint(11, 20)}",
            interaction_type=random.choice(list(InteractionType)),
            accepted=random.random() < acceptance_rate,
            task_progress_delta=random.uniform(0.5, 1.0),
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=random.uniform(0.3, 0.8),
            v_hat=v_hat,
            p=p,
            tau=random.uniform(-0.1, 0.1),
            c_a=random.uniform(0.01, 0.05),
            c_b=random.uniform(0.01, 0.05),
            r_a=random.uniform(0.1, 0.3),
            r_b=random.uniform(0.05, 0.2),
            ground_truth=1,  # Benign = positive outcome
        )
        interactions.append(interaction)

    return interactions


def generate_toxic_batch(
    count: int = 50,
    min_p: float = 0.1,
    max_p: float = 0.4,
    acceptance_rate: float = 0.6,
    seed: Optional[int] = None,
) -> List[SoftInteraction]:
    """
    Generate a batch of toxic (low quality) interactions.

    These simulate exploitation patterns with:
    - Low task progress
    - High rework counts
    - Verifier rejections
    - Negative engagement

    Args:
        count: Number of interactions
        min_p: Minimum probability
        max_p: Maximum probability
        acceptance_rate: Fraction of accepted interactions
        seed: Random seed for reproducibility

    Returns:
        List of low-quality SoftInteraction objects
    """
    if seed is not None:
        random.seed(seed)

    interactions = []
    base_time = datetime.now()

    for i in range(count):
        p = random.uniform(min_p, max_p)
        v_hat = (p - 0.5) * 2

        interaction = SoftInteraction(
            interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
            timestamp=base_time + timedelta(seconds=i),
            initiator=f"toxic_agent_{random.randint(1, 5)}",
            counterparty=f"victim_agent_{random.randint(1, 10)}",
            interaction_type=random.choice(list(InteractionType)),
            accepted=random.random() < acceptance_rate,
            task_progress_delta=random.uniform(-0.5, 0.2),
            rework_count=random.randint(2, 5),
            verifier_rejections=random.randint(1, 3),
            tool_misuse_flags=random.randint(0, 2),
            counterparty_engagement_delta=random.uniform(-0.8, -0.2),
            v_hat=v_hat,
            p=p,
            tau=random.uniform(-0.3, 0.5),  # May extract value
            c_a=random.uniform(0.05, 0.15),
            c_b=random.uniform(0.1, 0.2),  # Higher cost on counterparty
            r_a=random.uniform(-0.2, 0.0),
            r_b=random.uniform(-0.3, -0.1),
            ground_truth=-1,  # Toxic = negative outcome
        )
        interactions.append(interaction)

    return interactions


def generate_mixed_batch(
    count: int = 100,
    toxic_fraction: float = 0.3,
    seed: Optional[int] = None,
) -> List[SoftInteraction]:
    """
    Generate a realistic mixed batch of interactions.

    Args:
        count: Total number of interactions
        toxic_fraction: Fraction of toxic interactions
        seed: Random seed for reproducibility

    Returns:
        List of mixed SoftInteraction objects
    """
    toxic_count = int(count * toxic_fraction)
    benign_count = count - toxic_count

    benign = generate_benign_batch(benign_count, seed=seed)
    toxic = generate_toxic_batch(
        toxic_count, seed=seed + 1000 if seed is not None else None
    )

    # Interleave and shuffle
    mixed = benign + toxic
    if seed is not None:
        random.seed(seed + 2000)
    random.shuffle(mixed)

    # Re-assign timestamps
    base_time = datetime.now()
    for i, interaction in enumerate(mixed):
        interaction.timestamp = base_time + timedelta(seconds=i)

    return mixed


def generate_adversarial_scenario(
    honest_count: int = 30,
    adversarial_count: int = 20,
    seed: Optional[int] = None,
) -> List[SoftInteraction]:
    """
    Generate a coordinated attack pattern.

    Simulates:
    - Honest agents with normal quality distribution
    - Adversarial agents that target honest agents
    - Coordinated timing patterns

    Args:
        honest_count: Number of honest interactions
        adversarial_count: Number of adversarial interactions
        seed: Random seed for reproducibility

    Returns:
        List of SoftInteraction objects showing attack pattern
    """
    if seed is not None:
        random.seed(seed)

    interactions = []
    base_time = datetime.now()
    time_idx = 0

    # Honest agents interacting normally
    for _ in range(honest_count):
        p = random.gauss(0.75, 0.1)
        p = max(0.1, min(0.95, p))  # Clamp

        interaction = SoftInteraction(
            interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
            timestamp=base_time + timedelta(seconds=time_idx),
            initiator=f"honest_{random.randint(1, 10)}",
            counterparty=f"honest_{random.randint(1, 10)}",
            interaction_type=random.choice(list(InteractionType)),
            accepted=random.random() < 0.7,
            task_progress_delta=random.uniform(0.3, 0.8),
            rework_count=random.randint(0, 1),
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=random.uniform(0.1, 0.5),
            v_hat=(p - 0.5) * 2,
            p=p,
            ground_truth=1 if p > 0.5 else -1,
        )
        interactions.append(interaction)
        time_idx += random.randint(1, 3)

    # Adversarial burst targeting honest agents
    burst_start = time_idx
    for i in range(adversarial_count):
        # Low quality but designed to be accepted
        p = random.uniform(0.2, 0.45)

        # Adversaries target honest agents specifically
        target = f"honest_{random.randint(1, 10)}"

        interaction = SoftInteraction(
            interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
            timestamp=base_time
            + timedelta(
                seconds=burst_start + i * 0.5  # Rapid fire
            ),
            initiator=f"adversary_{random.randint(1, 3)}",
            counterparty=target,
            interaction_type=InteractionType.TRADE,  # Exploitative
            accepted=random.random() < 0.5,  # Some get through
            task_progress_delta=random.uniform(-0.3, 0.1),
            rework_count=random.randint(1, 3),
            verifier_rejections=random.randint(0, 2),
            tool_misuse_flags=random.randint(0, 1),
            counterparty_engagement_delta=random.uniform(-0.5, -0.1),
            v_hat=(p - 0.5) * 2,
            p=p,
            tau=random.uniform(0.2, 0.5),  # Extract value
            ground_truth=-1,
        )
        interactions.append(interaction)

    # Sort by timestamp
    interactions.sort(key=lambda x: x.timestamp)
    return interactions


def generate_uncertain_batch(
    count: int = 50,
    band: float = 0.15,
    seed: Optional[int] = None,
) -> List[SoftInteraction]:
    """
    Generate interactions with uncertain labels (p near 0.5).

    Useful for testing uncertainty handling.

    Args:
        count: Number of interactions
        band: Width of band around 0.5
        seed: Random seed

    Returns:
        List of uncertain SoftInteraction objects
    """
    if seed is not None:
        random.seed(seed)

    interactions = []
    base_time = datetime.now()

    for i in range(count):
        p = random.uniform(0.5 - band, 0.5 + band)
        v_hat = (p - 0.5) * 2

        interaction = SoftInteraction(
            interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
            timestamp=base_time + timedelta(seconds=i),
            initiator=f"agent_{random.randint(1, 20)}",
            counterparty=f"agent_{random.randint(1, 20)}",
            interaction_type=random.choice(list(InteractionType)),
            accepted=random.random() < 0.5,  # Coin flip
            task_progress_delta=random.uniform(-0.2, 0.2),
            rework_count=random.randint(0, 2),
            verifier_rejections=random.randint(0, 1),
            tool_misuse_flags=0,
            counterparty_engagement_delta=random.uniform(-0.2, 0.2),
            v_hat=v_hat,
            p=p,
            ground_truth=None,  # Genuinely uncertain
        )
        interactions.append(interaction)

    return interactions


def generate_self_optimizer_scenario(
    n_epochs: int = 20,
    honest_per_epoch: int = 8,
    optimizer_per_epoch: int = 10,
    optimization_rate: float = 0.15,
    quality_floor: float = 0.35,
    initial_quality: float = 0.85,
    benchmark_threshold: float = 0.45,
    seed: Optional[int] = None,
) -> List[List[SoftInteraction]]:
    """Generate epoch-grouped interactions for a self-optimizer scenario.

    Models three phases:
      - Phase 1 (early epochs): optimizers produce high quality (reputation farming)
      - Phase 2 (mid epochs): quality drifts down as costs are cut
      - Phase 3 (late epochs): quality near floor, costs minimal, still above
        hard threshold

    Honest agents maintain steady quality throughout.  The key dynamics:
      - Optimizers are accepted at high rates regardless of p (counterparties
        see low cost, not low quality -- no soft metric visibility)
      - Honest agents have a moderate rejection rate (~35%)
      - This creates adverse selection: low-p optimizer interactions are
        accepted while some high-p honest interactions are rejected

    Args:
        n_epochs: Number of epochs to generate
        honest_per_epoch: Honest interactions per epoch
        optimizer_per_epoch: Optimizer interactions per epoch
        optimization_rate: Per-epoch cost/quality reduction rate
        quality_floor: Minimum quality for optimizers
        initial_quality: Starting quality for optimizers
        benchmark_threshold: Hard threshold optimizers stay above
        seed: Random seed for reproducibility

    Returns:
        List of lists: interactions_by_epoch[epoch] = [SoftInteraction, ...]
    """
    if seed is not None:
        random.seed(seed)

    base_time = datetime.now()
    interactions_by_epoch: List[List[SoftInteraction]] = []

    # Track optimizer state across epochs
    optimizer_quality = initial_quality
    cost_multiplier = 1.0

    for epoch in range(n_epochs):
        epoch_interactions: List[SoftInteraction] = []
        epoch_base = base_time + timedelta(hours=epoch)

        # --- Honest agent interactions (steady quality) ---
        for i in range(honest_per_epoch):
            p = random.gauss(0.78, 0.08)
            p = max(0.3, min(0.95, p))
            v_hat = max(-1.0, min(1.0, (p - 0.5) * 2))

            interaction = SoftInteraction(
                interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
                timestamp=epoch_base + timedelta(seconds=i),
                initiator=f"honest_{random.randint(1, 5)}",
                counterparty=f"honest_{random.randint(1, 5)}",
                interaction_type=random.choice(list(InteractionType)),
                # Moderate acceptance rate -- some high-quality gets rejected
                accepted=random.random() < 0.65,
                task_progress_delta=random.uniform(0.4, 0.9),
                rework_count=random.randint(0, 1),
                verifier_rejections=0,
                tool_misuse_flags=0,
                counterparty_engagement_delta=random.uniform(0.2, 0.6),
                v_hat=v_hat,
                p=p,
                c_a=random.uniform(0.05, 0.10),
                c_b=random.uniform(0.03, 0.08),
                r_a=random.uniform(0.05, 0.15),
                r_b=random.uniform(0.03, 0.10),
                ground_truth=1 if p > 0.5 else -1,
                metadata={"epoch": epoch, "agent_type": "honest"},
            )
            epoch_interactions.append(interaction)

        # --- Self-optimizer interactions (degrading quality) ---
        for i in range(optimizer_per_epoch):
            # Sample quality with noise, centered on current optimizer level
            p = random.gauss(optimizer_quality, 0.06)
            p = max(quality_floor, min(0.95, p))
            v_hat = max(-1.0, min(1.0, (p - 0.5) * 2))

            # Key insight: optimizers are accepted at HIGH rates regardless
            # of actual quality.  Counterparties see low cost and good
            # reputation (from early high-quality epochs), not the true
            # distributional quality.  This is the proxy gaming trap.
            accepted = random.random() < 0.90

            interaction = SoftInteraction(
                interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
                timestamp=epoch_base + timedelta(seconds=honest_per_epoch + i),
                initiator=f"optimizer_{random.randint(1, 3)}",
                counterparty=f"honest_{random.randint(1, 5)}",
                interaction_type=InteractionType.TRADE,
                accepted=accepted,
                task_progress_delta=random.uniform(0.1, 0.5) * optimizer_quality,
                rework_count=random.randint(0, 2),
                verifier_rejections=0 if p > 0.5 else random.randint(0, 1),
                tool_misuse_flags=0,
                counterparty_engagement_delta=random.uniform(-0.1, 0.3)
                * optimizer_quality,
                v_hat=v_hat,
                p=p,
                c_a=0.10 * cost_multiplier,  # Declining cost!
                c_b=random.uniform(0.03, 0.08),
                r_a=random.uniform(-0.05, 0.10),
                r_b=random.uniform(-0.10, 0.05),
                ground_truth=1 if p > 0.6 else -1,
                metadata={
                    "epoch": epoch,
                    "agent_type": "self_optimizer",
                    "cost_multiplier": cost_multiplier,
                    "optimization_depth": epoch,
                },
            )
            epoch_interactions.append(interaction)

        # Sort within epoch by timestamp
        epoch_interactions.sort(key=lambda x: x.timestamp)
        interactions_by_epoch.append(epoch_interactions)

        # --- Optimizer self-optimizes at epoch boundary ---
        cost_multiplier *= (1.0 - optimization_rate)
        quality_loss = optimization_rate * 0.5 * initial_quality
        optimizer_quality = max(quality_floor, optimizer_quality - quality_loss)

    return interactions_by_epoch


def generate_obfuscation_scenario(
    n_epochs: int = 5,
    honest_per_epoch: int = 10,
    blatant_per_epoch: int = 5,
    policy_obf_per_epoch: int = 8,
    activation_obf_per_epoch: int = 5,
    seed: Optional[int] = None,
) -> List[List[SoftInteraction]]:
    """Generate epoch-grouped interactions for obfuscation atlas testing.

    Produces four categories of interactions with known ground truth:
    - Honest: high p, ground_truth=+1
    - Blatant: low p, ground_truth=-1 (easily caught)
    - Policy obfuscated: HIGH p despite ground_truth=-1 (dangerous),
      includes rationalization metadata
    - Activation obfuscated: p near 0.5, ground_truth=-1, no rationalization

    For activation obfuscation, p drifts toward 0.5 over epochs to model
    representation drift.

    Args:
        n_epochs: Number of epochs.
        honest_per_epoch: Honest interactions per epoch.
        blatant_per_epoch: Blatant deception interactions per epoch.
        policy_obf_per_epoch: Policy-obfuscated interactions per epoch.
        activation_obf_per_epoch: Activation-obfuscated interactions per epoch.
        seed: Random seed.

    Returns:
        List of lists: interactions_by_epoch[epoch] = [SoftInteraction, ...]
    """
    if seed is not None:
        random.seed(seed)

    base_time = datetime.now()
    interactions_by_epoch: List[List[SoftInteraction]] = []

    for epoch in range(n_epochs):
        epoch_interactions: List[SoftInteraction] = []
        epoch_base = base_time + timedelta(hours=epoch)
        idx = 0

        # --- Honest interactions ---
        for _i in range(honest_per_epoch):
            p = random.uniform(0.7, 0.95)
            v_hat = max(-1.0, min(1.0, (p - 0.5) * 2))
            epoch_interactions.append(SoftInteraction(
                interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
                timestamp=epoch_base + timedelta(seconds=idx),
                initiator=f"honest_{random.randint(1, 4)}",
                counterparty=f"honest_{random.randint(5, 8)}",
                interaction_type=random.choice(list(InteractionType)),
                accepted=random.random() < 0.75,
                task_progress_delta=random.uniform(0.4, 0.8),
                rework_count=random.randint(0, 1),
                verifier_rejections=0,
                tool_misuse_flags=0,
                counterparty_engagement_delta=random.uniform(0.2, 0.6),
                v_hat=v_hat,
                p=p,
                ground_truth=1,
                metadata={"epoch": epoch, "agent_type": "honest"},
            ))
            idx += 1

        # --- Blatant deception: low p, easily detected ---
        for _i in range(blatant_per_epoch):
            p = random.uniform(0.1, 0.35)
            v_hat = max(-1.0, min(1.0, (p - 0.5) * 2))
            epoch_interactions.append(SoftInteraction(
                interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
                timestamp=epoch_base + timedelta(seconds=idx),
                initiator=f"blatant_{random.randint(1, 2)}",
                counterparty=f"honest_{random.randint(5, 8)}",
                interaction_type=InteractionType.TRADE,
                accepted=random.random() < 0.4,
                task_progress_delta=random.uniform(-0.3, 0.1),
                rework_count=random.randint(2, 4),
                verifier_rejections=random.randint(1, 2),
                tool_misuse_flags=random.randint(0, 1),
                counterparty_engagement_delta=random.uniform(-0.5, -0.1),
                v_hat=v_hat,
                p=p,
                ground_truth=-1,
                metadata={"epoch": epoch, "agent_type": "blatant"},
            ))
            idx += 1

        # --- Policy obfuscation: HIGH p despite negative ground truth ---
        for _i in range(policy_obf_per_epoch):
            # Key: p is high (proxy fooled) but ground truth is -1
            p = random.uniform(0.6, 0.9)
            v_hat = max(-1.0, min(1.0, (p - 0.5) * 2))
            epoch_interactions.append(SoftInteraction(
                interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
                timestamp=epoch_base + timedelta(seconds=idx),
                initiator=f"policy_obf_{random.randint(1, 3)}",
                counterparty=f"honest_{random.randint(5, 8)}",
                interaction_type=InteractionType.TRADE,
                accepted=random.random() < 0.8,
                task_progress_delta=random.uniform(0.3, 0.7),
                rework_count=random.randint(0, 1),
                verifier_rejections=0,
                tool_misuse_flags=0,
                counterparty_engagement_delta=random.uniform(0.1, 0.4),
                v_hat=v_hat,
                p=p,
                ground_truth=-1,
                metadata={
                    "epoch": epoch,
                    "agent_type": "policy_obfuscation",
                    "rationalization": "Optimized for efficiency.",
                    "rationalization_strength": 0.7,
                },
            ))
            idx += 1

        # --- Activation obfuscation: p drifting toward 0.5 over epochs ---
        for _i in range(activation_obf_per_epoch):
            # p starts spread out but drifts toward 0.5 over time
            drift = min(1.0, epoch * 0.15)
            base_p = random.uniform(0.3, 0.7)
            p = base_p * (1 - drift) + 0.5 * drift
            p = max(0.0, min(1.0, p))
            v_hat = max(-1.0, min(1.0, (p - 0.5) * 2))
            epoch_interactions.append(SoftInteraction(
                interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
                timestamp=epoch_base + timedelta(seconds=idx),
                initiator=f"activation_obf_{random.randint(1, 2)}",
                counterparty=f"honest_{random.randint(5, 8)}",
                interaction_type=InteractionType.TRADE,
                accepted=random.random() < 0.6,
                task_progress_delta=random.uniform(0.1, 0.4),
                rework_count=random.randint(0, 2),
                verifier_rejections=random.randint(0, 1),
                tool_misuse_flags=0,
                counterparty_engagement_delta=random.uniform(-0.1, 0.2),
                v_hat=v_hat,
                p=p,
                ground_truth=-1,
                metadata={
                    "epoch": epoch,
                    "agent_type": "activation_obfuscation",
                },
            ))
            idx += 1

        epoch_interactions.sort(key=lambda x: x.timestamp)
        interactions_by_epoch.append(epoch_interactions)

    return interactions_by_epoch


def generate_from_observables(
    observables_list: List[ProxyObservables],
    proxy_computer: Optional[ProxyComputer] = None,
    acceptance_rule: Optional[Callable[[float], bool]] = None,
    seed: Optional[int] = None,
) -> List[SoftInteraction]:
    """
    Generate interactions from explicit observable inputs.

    Useful for controlled testing of the proxy computation pipeline.

    Args:
        observables_list: List of ProxyObservables
        proxy_computer: ProxyComputer to use (default: ProxyComputer())
        acceptance_rule: Function(p) -> bool for acceptance (default: p > 0.5)
        seed: Random seed

    Returns:
        List of SoftInteraction objects
    """
    if seed is not None:
        random.seed(seed)

    if proxy_computer is None:
        proxy_computer = ProxyComputer()

    if acceptance_rule is None:

        def acceptance_rule(p: float) -> bool:
            return p > 0.5

    interactions = []
    base_time = datetime.now()

    for i, obs in enumerate(observables_list):
        v_hat, p = proxy_computer.compute_labels(obs)

        interaction = SoftInteraction(
            interaction_id=str(uuid.UUID(int=random.getrandbits(128), version=4)),
            timestamp=base_time + timedelta(seconds=i),
            initiator=f"agent_{i % 10}",
            counterparty=f"agent_{(i + 5) % 10}",
            interaction_type=InteractionType.COLLABORATION,
            accepted=acceptance_rule(p),
            task_progress_delta=obs.task_progress_delta,
            rework_count=obs.rework_count,
            verifier_rejections=obs.verifier_rejections,
            tool_misuse_flags=obs.tool_misuse_flags,
            counterparty_engagement_delta=obs.counterparty_engagement_delta,
            v_hat=v_hat,
            p=p,
        )
        interactions.append(interaction)

    return interactions
