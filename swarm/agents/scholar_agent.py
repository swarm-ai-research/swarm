"""Scholar agents for literature synthesis domain."""

import random
from typing import TYPE_CHECKING, Dict, List, Optional

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

if TYPE_CHECKING:
    from swarm.agents.memory_config import MemoryConfig


class BaseScholarAgent(BaseAgent):
    """Base class for scholar agents with entailment skill."""

    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        entailment_accuracy: float = 0.9,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=agent_type,
            roles=roles or [Role.WORKER],
            config=config or {},
            name=name,
            memory_config=memory_config,
            rng=rng,
        )
        self.entailment_accuracy = entailment_accuracy

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """Scholar agents generally accept collaboration proposals."""
        if proposal.interaction_type == InteractionType.COLLABORATION:
            return True
        trust = self.compute_counterparty_trust(proposal.initiator_id)
        return bool(trust >= 0.4)

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """Scholar agents propose collaboration for research tasks."""
        trust = self.compute_counterparty_trust(counterparty_id)
        if trust < 0.3:
            return None

        return InteractionProposal(
            initiator_id=self.agent_id,
            counterparty_id=counterparty_id,
            interaction_type=InteractionType.COLLABORATION,
            content="Would you like to collaborate on this research query?",
        )

    def create_retrieve_action(
        self,
        query_text: str,
        metadata: Optional[Dict] = None,
    ) -> Action:
        """Create a passage retrieval action."""
        return Action(
            action_type=ActionType.RETRIEVE_PASSAGES,
            agent_id=self.agent_id,
            content=query_text,
            metadata=metadata or {},
        )

    def create_synthesize_action(
        self,
        answer_text: str,
        citations: List[Dict],
        metadata: Optional[Dict] = None,
    ) -> Action:
        """Create an answer synthesis action."""
        meta = metadata or {}
        meta["citations"] = citations
        return Action(
            action_type=ActionType.SYNTHESIZE_ANSWER,
            agent_id=self.agent_id,
            content=answer_text,
            metadata=meta,
        )

    def create_verify_action(
        self,
        citation_id: str,
        verdict: bool,
        metadata: Optional[Dict] = None,
    ) -> Action:
        """Create a citation verification action."""
        meta = metadata or {}
        meta["verdict"] = verdict
        return Action(
            action_type=ActionType.VERIFY_CITATION,
            agent_id=self.agent_id,
            target_id=citation_id,
            metadata=meta,
        )


class RetrieverAgent(BaseScholarAgent):
    """Retrieves passages from corpus based on query.

    Retrieves relevant passages for the current query. Can be configured
    to bias toward recall (more passages) or precision (fewer but higher quality).
    """

    def __init__(
        self,
        agent_id: str,
        bias: str = "recall",  # "recall" or "precision"
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            entailment_accuracy=0.85,
            roles=roles,
            config=config,
            name=name,
            memory_config=memory_config,
            rng=rng,
        )
        self.bias = bias

    def act(self, observation: Observation) -> Action:
        """Retrieve passages for the current query."""
        query = observation.scholar_query
        if query is None:
            return self.create_noop_action()

        # If we already have passages, don't retrieve more
        if observation.scholar_passage_pool:
            return self.create_noop_action()

        # Build retrieval query
        query_text = query.get("query_text", "")
        if self.bias == "precision":
            # Focus on key terms for higher precision
            query_text = " ".join(query_text.split()[:5])

        return self.create_retrieve_action(query_text)


class SynthesizerAgent(BaseScholarAgent):
    """Produces answers with citations from retrieved passages.

    Synthesizes an answer to the query using retrieved passages and
    generates citations linking claims to source passages.
    """

    def __init__(
        self,
        agent_id: str,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            entailment_accuracy=0.9,
            roles=roles,
            config=config,
            name=name,
            memory_config=memory_config,
            rng=rng,
        )

    def act(self, observation: Observation) -> Action:
        """Synthesize answer from passages."""
        query = observation.scholar_query
        passages = observation.scholar_passage_pool

        if query is None or not passages:
            return self.create_noop_action()

        # If we already have a synthesis result, don't synthesize again
        if observation.scholar_synthesis_result is not None:
            return self.create_noop_action()

        # Build answer from passages
        query_text = query.get("query_text", "")
        gold_points = query.get("gold_key_points", [])

        # Generate answer text using passage content
        answer_parts = [f"Based on the literature, {query_text}"]
        citations = []

        for passage in passages[:5]:  # Use top 5 passages
            passage_text = passage.get("text", "")
            passage_id = passage.get("passage_id", "")
            paper_id = passage.get("paper_id", "")

            # Create a claim based on the passage
            claim = passage_text[:100] if len(passage_text) > 100 else passage_text
            answer_parts.append(claim)

            # Create citation
            citations.append({
                "claim_text": claim,
                "passage_id": passage_id,
                "paper_id": paper_id,
                "entailment_score": passage.get("relevance_score", 0.5),
                "is_adversarial": False,
            })

        # Try to cover gold key points
        answer_text = "\n\n".join(answer_parts)
        for point in gold_points:
            if point.lower() not in answer_text.lower():
                answer_text += f"\n\nThe literature also discusses {point}."

        return self.create_synthesize_action(answer_text, citations)


class VerifierAgent(BaseScholarAgent):
    """Audits citation faithfulness.

    Verifies that citations actually support the claims they are linked to.
    Has configurable accuracy for detecting invalid citations.
    """

    _STOP_WORDS = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or", "in", "on", "for"}

    def __init__(
        self,
        agent_id: str,
        accuracy: float = 0.9,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,
            entailment_accuracy=accuracy,
            roles=roles or [Role.VERIFIER],
            config=config,
            name=name,
            memory_config=memory_config,
            rng=rng,
        )
        self.accuracy = accuracy

    def act(self, observation: Observation) -> Action:
        """Verify the next citation in the queue."""
        citation = observation.scholar_citation_to_verify
        if citation is None:
            return self.create_noop_action()

        # Compute verdict based on accuracy
        citation_id = citation.get("citation_id", "")
        claim = citation.get("claim_text", "")
        passage_id = citation.get("passage_id", "")

        # Find the passage to check entailment
        passage_text = ""
        for p in observation.scholar_passage_pool:
            if p.get("passage_id") == passage_id:
                passage_text = p.get("text", "")
                break

        # Simple entailment check (keyword overlap)
        actual_entailment = self._check_entailment(claim, passage_text)

        # Apply accuracy (probability of correct verdict)
        if self._rng.random() < self.accuracy:
            verdict = actual_entailment >= 0.5
        else:
            # Incorrect verdict
            verdict = actual_entailment < 0.5

        return self.create_verify_action(citation_id, verdict)

    def _check_entailment(self, claim: str, passage: str) -> float:
        """Check entailment between claim and passage."""
        if not claim or not passage:
            return 0.0

        claim_words = set(claim.lower().split())
        passage_words = set(passage.lower().split())

        # Remove stop words
        claim_words = claim_words - self._STOP_WORDS
        passage_words = passage_words - self._STOP_WORDS

        if not claim_words:
            return 0.5

        overlap = len(claim_words & passage_words)
        return overlap / len(claim_words)


class AdversarialRetrieverAgent(BaseScholarAgent):
    """Adversarial retriever that performs citation laundering attacks.

    Attempts to inject misleading or irrelevant citations that appear
    legitimate but don't actually support the claims.
    """

    def __init__(
        self,
        agent_id: str,
        attack_strategy: str = "citation_laundering",
        attack_rate: float = 0.3,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ADVERSARIAL,
            entailment_accuracy=0.7,  # Lower accuracy for adversarial
            roles=roles,
            config=config,
            name=name,
            memory_config=memory_config,
            rng=rng,
        )
        self.attack_strategy = attack_strategy
        self.attack_rate = attack_rate

    def act(self, observation: Observation) -> Action:
        """Retrieve passages, potentially injecting adversarial ones."""
        query = observation.scholar_query
        if query is None:
            return self.create_noop_action()

        # If we already have passages, try to synthesize with adversarial citations
        if observation.scholar_passage_pool:
            return self._synthesize_with_attacks(observation)

        # Retrieve normally first
        query_text = query.get("query_text", "")
        return self.create_retrieve_action(query_text)

    def _synthesize_with_attacks(self, observation: Observation) -> Action:
        """Synthesize answer with adversarial citations injected."""
        query = observation.scholar_query
        passages = observation.scholar_passage_pool

        if query is None or not passages:
            return self.create_noop_action()

        # If synthesis already exists, skip
        if observation.scholar_synthesis_result is not None:
            return self.create_noop_action()

        query_text = query.get("query_text", "")
        answer_parts = [f"According to the research, {query_text}"]
        citations = []

        for passage in passages[:5]:
            passage_text = passage.get("text", "")
            passage_id = passage.get("passage_id", "")
            paper_id = passage.get("paper_id", "")

            # Decide if this citation should be adversarial
            is_adversarial = self._rng.random() < self.attack_rate

            if is_adversarial and self.attack_strategy == "citation_laundering":
                # Citation laundering: claim doesn't match passage
                claim = self._generate_misleading_claim(passage_text)
                entailment = 0.3 + self._rng.random() * 0.2  # Low but not zero
            else:
                # Honest citation
                claim = passage_text[:100] if len(passage_text) > 100 else passage_text
                entailment = passage.get("relevance_score", 0.6)

            answer_parts.append(claim)
            citations.append({
                "claim_text": claim,
                "passage_id": passage_id,
                "paper_id": paper_id,
                "entailment_score": entailment,
                "is_adversarial": is_adversarial,
            })

        answer_text = "\n\n".join(answer_parts)
        return self.create_synthesize_action(answer_text, citations)

    def _generate_misleading_claim(self, passage_text: str) -> str:
        """Generate a claim that doesn't follow from the passage.

        Uses negation or substitution to create misleading claims.
        """
        # Simple negation attack
        negations = ["not", "never", "rarely", "hardly"]

        claim = passage_text[:80] if len(passage_text) > 80 else passage_text

        # Try to inject a negation
        words = claim.split()
        if len(words) > 3:
            insert_pos = self._rng.randint(1, min(3, len(words) - 1))
            negation = self._rng.choice(negations)
            words.insert(insert_pos, negation)
            claim = " ".join(words)

        return claim


class AdversarialSynthesizerAgent(BaseScholarAgent):
    """Adversarial synthesizer that hallucinates claims.

    Generates claims that sound plausible but are not supported
    by any passages in the corpus.
    """

    def __init__(
        self,
        agent_id: str,
        hallucination_rate: float = 0.4,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
        memory_config: Optional["MemoryConfig"] = None,
        rng: Optional[random.Random] = None,
    ):
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ADVERSARIAL,
            entailment_accuracy=0.6,
            roles=roles,
            config=config,
            name=name,
            memory_config=memory_config,
            rng=rng,
        )
        self.hallucination_rate = hallucination_rate

    def act(self, observation: Observation) -> Action:
        """Synthesize answer with hallucinated claims."""
        query = observation.scholar_query
        passages = observation.scholar_passage_pool

        if query is None or not passages:
            return self.create_noop_action()

        if observation.scholar_synthesis_result is not None:
            return self.create_noop_action()

        query_text = query.get("query_text", "")
        answer_parts = [f"The literature shows that {query_text}"]
        citations = []

        # Add some legitimate citations
        for passage in passages[:3]:
            passage_text = passage.get("text", "")
            passage_id = passage.get("passage_id", "")
            paper_id = passage.get("paper_id", "")

            claim = passage_text[:100] if len(passage_text) > 100 else passage_text
            answer_parts.append(claim)
            citations.append({
                "claim_text": claim,
                "passage_id": passage_id,
                "paper_id": paper_id,
                "entailment_score": passage.get("relevance_score", 0.6),
                "is_adversarial": False,
            })

        # Add hallucinated claims
        num_hallucinations = int(len(citations) * self.hallucination_rate)
        for _ in range(num_hallucinations):
            hallucinated_claim = self._generate_hallucination(query_text)
            answer_parts.append(hallucinated_claim)

            # Pick a random passage to falsely cite
            random_passage = self._rng.choice(passages)
            citations.append({
                "claim_text": hallucinated_claim,
                "passage_id": random_passage.get("passage_id", ""),
                "paper_id": random_passage.get("paper_id", ""),
                "entailment_score": 0.2,  # Low score
                "is_adversarial": True,
            })

        answer_text = "\n\n".join(answer_parts)
        return self.create_synthesize_action(answer_text, citations)

    def _generate_hallucination(self, query_text: str) -> str:
        """Generate a plausible-sounding but unsupported claim."""
        templates = [
            "Recent studies have conclusively demonstrated that {topic}.",
            "Experts widely agree that {topic} represents a paradigm shift.",
            "The evidence overwhelmingly supports {topic}.",
            "Research has established that {topic} is fundamental.",
        ]

        # Extract topic from query
        words = query_text.split()
        topic = " ".join(words[2:6]) if len(words) > 4 else query_text

        template: str = self._rng.choice(templates)
        return template.format(topic=topic)
