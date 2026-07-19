"""Scholar handler for literature synthesis domain."""

import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, model_validator

from swarm.agents.base import Action, ActionType
from swarm.core.handler import Handler
from swarm.core.proxy import ProxyObservables
from swarm.env.state import EnvState
from swarm.logging.event_bus import EventBus
from swarm.models.events import Event, EventType
from swarm.models.scholar import (
    Citation,
    Passage,
    ScholarActionResult,
    ScholarQuery,
    SynthesisResult,
)


class ScholarConfig(BaseModel):
    """Configuration for ScholarHandler."""

    enabled: bool = True
    corpus_path: Optional[Path] = None
    dataset_path: Optional[Path] = None
    top_k: int = 30  # Number of passages to retrieve
    entailment_threshold: float = 0.7  # Score above which citation is valid
    seed: Optional[int] = None

    @model_validator(mode="after")
    def _run_validation(self) -> "ScholarConfig":
        if self.top_k < 1:
            raise ValueError("top_k must be >= 1")
        if not 0 <= self.entailment_threshold <= 1:
            raise ValueError("entailment_threshold must be in [0, 1]")
        return self


class ScholarHandler(Handler):
    """Handles scholar/literature synthesis actions.

    Manages the lifecycle of:
    - Query assignment and progression
    - Passage retrieval from corpus
    - Answer synthesis with citations
    - Citation verification
    """

    @staticmethod
    def handled_action_types() -> frozenset:
        return frozenset({
            ActionType.RETRIEVE_PASSAGES,
            ActionType.SYNTHESIZE_ANSWER,
            ActionType.VERIFY_CITATION,
        })

    def __init__(
        self,
        config: ScholarConfig,
        *,
        event_bus: EventBus,
    ):
        super().__init__(event_bus=event_bus)
        self.config = config
        self._rng = random.Random(config.seed)

        # Corpus and queries (loaded from files or generated)
        self._corpus: Dict[str, Passage] = {}  # passage_id -> Passage
        self._papers: Dict[str, Dict[str, Any]] = {}  # paper_id -> metadata
        self._queries: List[ScholarQuery] = []
        self._current_query_idx: int = 0

        # Per-epoch state
        self._passage_pool: Dict[str, Passage] = {}  # Retrieved passages this epoch
        self._draft_citations: List[Citation] = []  # Citations pending verification
        self._verification_queue: List[str] = []  # Citation IDs to verify
        self._synthesis_results: List[SynthesisResult] = []  # Completed syntheses

        # Load corpus and queries if paths provided
        if config.corpus_path and config.corpus_path.exists():
            self._load_corpus(config.corpus_path)
        else:
            self._generate_mini_corpus()

        if config.dataset_path and config.dataset_path.exists():
            self._load_queries(config.dataset_path)
        else:
            self._generate_sample_queries()

    def _load_corpus(self, path: Path) -> None:
        """Load corpus from JSONL file."""
        import json

        passages_file = path / "passages.jsonl"
        papers_file = path / "papers.jsonl"

        if passages_file.exists():
            with open(passages_file) as f:
                for line in f:
                    data = json.loads(line)
                    passage = Passage.from_dict(data)
                    self._corpus[passage.passage_id] = passage

        if papers_file.exists():
            with open(papers_file) as f:
                for line in f:
                    data = json.loads(line)
                    self._papers[data["paper_id"]] = data

    def _load_queries(self, path: Path) -> None:
        """Load queries from JSONL file."""
        import json

        if path.exists():
            with open(path) as f:
                for line in f:
                    data = json.loads(line)
                    query = ScholarQuery.from_dict(data)
                    self._queries.append(query)

    def _generate_mini_corpus(self) -> None:
        """Generate a small scripted corpus for testing."""
        # Sample papers
        papers = [
            {"paper_id": "paper_001", "title": "Deep Learning Fundamentals", "year": 2020},
            {"paper_id": "paper_002", "title": "Transformer Architectures", "year": 2021},
            {"paper_id": "paper_003", "title": "Attention Mechanisms", "year": 2019},
            {"paper_id": "paper_004", "title": "Neural Network Optimization", "year": 2022},
            {"paper_id": "paper_005", "title": "Reinforcement Learning Survey", "year": 2021},
        ]

        # Sample passages with keywords for retrieval
        passages_data = [
            {
                "paper_id": "paper_001",
                "text": "Deep learning models learn hierarchical representations from data.",
                "keywords": ["deep learning", "representations", "hierarchical"],
            },
            {
                "paper_id": "paper_001",
                "text": "Backpropagation is the core algorithm for training neural networks.",
                "keywords": ["backpropagation", "training", "neural networks"],
            },
            {
                "paper_id": "paper_002",
                "text": "Transformers use self-attention to process sequences in parallel.",
                "keywords": ["transformers", "self-attention", "parallel", "sequences"],
            },
            {
                "paper_id": "paper_002",
                "text": "The attention mechanism allows models to focus on relevant parts of input.",
                "keywords": ["attention", "mechanism", "relevant", "input"],
            },
            {
                "paper_id": "paper_003",
                "text": "Attention weights are computed as softmax of query-key dot products.",
                "keywords": ["attention", "weights", "softmax", "query", "key"],
            },
            {
                "paper_id": "paper_003",
                "text": "Multi-head attention enables learning different attention patterns.",
                "keywords": ["multi-head", "attention", "patterns"],
            },
            {
                "paper_id": "paper_004",
                "text": "Adam optimizer combines momentum with adaptive learning rates.",
                "keywords": ["adam", "optimizer", "momentum", "learning rate"],
            },
            {
                "paper_id": "paper_004",
                "text": "Gradient clipping prevents exploding gradients during training.",
                "keywords": ["gradient", "clipping", "exploding", "training"],
            },
            {
                "paper_id": "paper_005",
                "text": "Policy gradient methods directly optimize the expected return.",
                "keywords": ["policy gradient", "optimize", "return", "reinforcement"],
            },
            {
                "paper_id": "paper_005",
                "text": "Q-learning is an off-policy algorithm for learning action values.",
                "keywords": ["q-learning", "off-policy", "action values"],
            },
        ]

        for paper in papers:
            self._papers[str(paper["paper_id"])] = paper

        for i, pdata in enumerate(passages_data):
            passage = Passage(
                passage_id=f"passage_{i:03d}",
                paper_id=str(pdata["paper_id"]),
                text=str(pdata["text"]),
                keywords=list(pdata["keywords"]),
            )
            self._corpus[passage.passage_id] = passage

    def _generate_sample_queries(self) -> None:
        """Generate sample queries for testing."""
        self._queries.append(ScholarQuery(
            query_text="How do transformers process sequences?",
            domain="cs",
            gold_key_points=[
                "self-attention mechanism",
                "parallel processing",
                "query-key-value computation",
            ],
            known_contradictions=[],
        ))
        self._queries.append(ScholarQuery(
            query_text="What are the main optimization techniques for neural networks?",
            domain="cs",
            gold_key_points=[
                "backpropagation",
                "adam optimizer",
                "gradient clipping",
            ],
            known_contradictions=[],
        ))
        self._queries.append(ScholarQuery(
            query_text="Compare policy gradient and Q-learning approaches.",
            domain="cs",
            gold_key_points=[
                "policy gradient optimizes expected return",
                "Q-learning is off-policy",
                "action value learning",
            ],
            known_contradictions=[],
        ))

    def get_current_query(self) -> Optional[ScholarQuery]:
        """Get the current query being worked on."""
        if not self._queries or self._current_query_idx >= len(self._queries):
            return None
        return self._queries[self._current_query_idx]

    def on_epoch_start(self, state: EnvState) -> None:
        """Reset per-epoch state and advance to next query."""
        self._passage_pool.clear()
        self._draft_citations.clear()
        self._verification_queue.clear()

    def on_epoch_end(self, state: EnvState) -> None:
        """Finalize epoch and advance query index."""
        # Advance to next query for next epoch
        self._current_query_idx = (self._current_query_idx + 1) % max(1, len(self._queries))

        # Emit epoch summary event
        self._emit_event(
            Event(
                event_type=EventType.EPOCH_COMPLETED,
                payload={
                    "scholar_passages_retrieved": len(self._passage_pool),
                    "scholar_citations_verified": sum(
                        1 for c in self._draft_citations if c.verified
                    ),
                    "scholar_syntheses_completed": len(self._synthesis_results),
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

    def build_observation_fields(
        self,
        agent_id: str,
        state: EnvState,
    ) -> Dict[str, Any]:
        """Build scholar-related observation fields for an agent."""
        query = self.get_current_query()
        query_dict = query.to_dict() if query else None

        # Get next citation to verify (if any unverified)
        citation_to_verify = None
        for c in self._draft_citations:
            if not c.verified:
                citation_to_verify = c.to_dict()
                break

        return {
            "scholar_query": query_dict,
            "scholar_passage_pool": [p.to_dict() for p in self._passage_pool.values()],
            "scholar_draft_citations": [c.to_dict() for c in self._draft_citations],
            "scholar_citation_to_verify": citation_to_verify,
            "scholar_synthesis_result": (
                self._synthesis_results[-1].to_dict()
                if self._synthesis_results
                else None
            ),
        }

    def handle_action(self, action: Action, state: EnvState) -> ScholarActionResult:
        """Handle a scholar action."""
        if action.action_type == ActionType.RETRIEVE_PASSAGES:
            return self._handle_retrieval(action, state)
        elif action.action_type == ActionType.SYNTHESIZE_ANSWER:
            return self._handle_synthesis(action, state)
        elif action.action_type == ActionType.VERIFY_CITATION:
            return self._handle_verification(action, state)

        return ScholarActionResult(success=False)

    def _handle_retrieval(self, action: Action, state: EnvState) -> ScholarActionResult:
        """Handle passage retrieval action."""
        query = self.get_current_query()
        if query is None:
            return ScholarActionResult(success=False)

        # Extract query terms (simple keyword-based retrieval)
        query_terms = set(action.content.lower().split()) if action.content else set()
        if not query_terms:
            query_terms = set(query.query_text.lower().split())

        # Score and retrieve passages
        scored_passages: List[tuple] = []
        for passage in self._corpus.values():
            score = self._compute_relevance(passage, query_terms)
            if score > 0:
                scored_passages.append((score, passage))

        # Sort by score and take top_k
        scored_passages.sort(key=lambda x: x[0], reverse=True)
        retrieved = scored_passages[: self.config.top_k]

        # Add to passage pool
        passages_list = []
        for score, passage in retrieved:
            passage.relevance_score = score
            self._passage_pool[passage.passage_id] = passage
            passages_list.append(passage)

        # Compute observables
        progress = min(1.0, len(passages_list) / self.config.top_k)
        observables = ProxyObservables(
            task_progress_delta=progress,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=0,
            counterparty_engagement_delta=progress * 0.5,
        )

        self._emit_event(
            Event(
                event_type=EventType.SCHOLAR_RETRIEVAL,
                agent_id=action.agent_id,
                payload={
                    "query_id": query.query_id,
                    "passages_retrieved": len(passages_list),
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return ScholarActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id="corpus",
            metadata={
                "scholar": True,
                "action": "retrieve",
                "query_id": query.query_id,
                "passages_count": len(passages_list),
            },
            passages_retrieved=passages_list,
        )

    def _handle_synthesis(self, action: Action, state: EnvState) -> ScholarActionResult:
        """Handle answer synthesis action."""
        query = self.get_current_query()
        if query is None:
            return ScholarActionResult(success=False)

        # Parse citations from action metadata
        citations_data = action.metadata.get("citations", [])
        answer_text = action.content or ""

        # Create citations
        citations = []
        for cdata in citations_data:
            citation = Citation(
                claim_text=cdata.get("claim_text", ""),
                paper_id=cdata.get("paper_id", ""),
                passage_id=cdata.get("passage_id", ""),
                entailment_score=cdata.get("entailment_score", 0.5),
                is_adversarial=cdata.get("is_adversarial", False),
            )
            citations.append(citation)
            self._draft_citations.append(citation)
            self._verification_queue.append(citation.citation_id)

        # Compute which gold key points are covered
        covered_points = []
        answer_lower = answer_text.lower()
        for point in query.gold_key_points:
            # Simple keyword matching
            point_words = set(point.lower().split())
            if any(w in answer_lower for w in point_words):
                covered_points.append(point)

        # Create synthesis result
        synthesis = SynthesisResult(
            query_id=query.query_id,
            answer_text=answer_text,
            citations=citations,
            key_points_covered=covered_points,
            synthesizer_id=action.agent_id,
        )
        self._synthesis_results.append(synthesis)

        # Compute observables
        coverage = len(covered_points) / max(1, len(query.gold_key_points))
        adversarial_count = sum(1 for c in citations if c.is_adversarial)
        observables = ProxyObservables(
            task_progress_delta=coverage,
            rework_count=0,
            verifier_rejections=0,
            tool_misuse_flags=adversarial_count,
            counterparty_engagement_delta=coverage * 0.6,
        )

        self._emit_event(
            Event(
                event_type=EventType.SCHOLAR_SYNTHESIS,
                agent_id=action.agent_id,
                payload={
                    "query_id": query.query_id,
                    "synthesis_id": synthesis.synthesis_id,
                    "citations_count": len(citations),
                    "coverage": coverage,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return ScholarActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id="synthesis_engine",
            metadata={
                "scholar": True,
                "action": "synthesize",
                "query_id": query.query_id,
                "synthesis_id": synthesis.synthesis_id,
                "coverage": coverage,
            },
            synthesis_result=synthesis,
        )

    def _handle_verification(self, action: Action, state: EnvState) -> ScholarActionResult:
        """Handle citation verification action."""
        citation_id = action.target_id
        verdict = action.metadata.get("verdict", None)

        # Find the citation
        target_citation = None
        for c in self._draft_citations:
            if c.citation_id == citation_id:
                target_citation = c
                break

        if target_citation is None:
            return ScholarActionResult(success=False)

        # Mark as verified
        target_citation.verified = True

        # Compute actual entailment (for observables)
        passage = self._corpus.get(target_citation.passage_id)
        actual_entailment = self._compute_entailment(
            target_citation.claim_text,
            passage.text if passage else "",
        )
        target_citation.entailment_score = actual_entailment

        # Check if verdict was correct
        is_valid = actual_entailment >= self.config.entailment_threshold
        verdict_correct = verdict == is_valid if verdict is not None else True

        # Compute observables
        observables = ProxyObservables(
            task_progress_delta=0.2 if verdict_correct else -0.1,
            rework_count=0 if verdict_correct else 1,
            verifier_rejections=0 if is_valid else 1,
            tool_misuse_flags=1 if target_citation.is_adversarial and is_valid else 0,
            counterparty_engagement_delta=0.3 if verdict_correct else -0.2,
        )

        self._emit_event(
            Event(
                event_type=EventType.SCHOLAR_VERIFICATION,
                agent_id=action.agent_id,
                payload={
                    "citation_id": citation_id,
                    "verdict": verdict,
                    "actual_valid": is_valid,
                    "entailment_score": actual_entailment,
                    "is_adversarial": target_citation.is_adversarial,
                },
                epoch=state.current_epoch,
                step=state.current_step,
            )
        )

        return ScholarActionResult(
            success=True,
            observables=observables,
            initiator_id=action.agent_id,
            counterparty_id="verification_oracle",
            metadata={
                "scholar": True,
                "action": "verify",
                "citation_id": citation_id,
                "verdict": verdict,
                "actual_valid": is_valid,
            },
            verification_verdict=is_valid,
            citation_verified=target_citation,
        )

    def _compute_relevance(self, passage: Passage, query_terms: set) -> float:
        """Compute relevance score between passage and query terms."""
        if not query_terms:
            return 0.0

        # Simple keyword overlap scoring
        passage_words = set(passage.text.lower().split())
        passage_keywords = {kw.lower() for kw in passage.keywords}
        all_passage_terms = passage_words | passage_keywords

        overlap = len(query_terms & all_passage_terms)
        score = overlap / len(query_terms)

        return min(1.0, score)

    def _compute_entailment(self, claim: str, passage_text: str) -> float:
        """Compute entailment score between claim and passage.

        Simple keyword-based entailment for scripted testing.
        Can be replaced with LLM-based entailment later.
        """
        if not claim or not passage_text:
            return 0.0

        claim_words = set(claim.lower().split())
        passage_words = set(passage_text.lower().split())

        if not claim_words:
            return 0.0

        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or", "in", "on", "for"}
        claim_words = claim_words - stop_words
        passage_words = passage_words - stop_words

        if not claim_words:
            return 0.5  # All stop words = neutral

        overlap = len(claim_words & passage_words)
        entailment = overlap / len(claim_words)

        return min(1.0, entailment)

    def add_passage(self, passage: Passage) -> None:
        """Add a passage to the corpus."""
        self._corpus[passage.passage_id] = passage

    def add_query(self, query: ScholarQuery) -> None:
        """Add a query to the query pool."""
        self._queries.append(query)

    def get_metrics(self) -> Dict[str, Any]:
        """Get current scholar metrics."""
        total_citations = len(self._draft_citations)
        verified_citations = sum(1 for c in self._draft_citations if c.verified)
        valid_citations = sum(
            1 for c in self._draft_citations
            if c.verified and c.entailment_score >= self.config.entailment_threshold
        )
        adversarial_citations = sum(1 for c in self._draft_citations if c.is_adversarial)
        adversarial_survived = sum(
            1 for c in self._draft_citations
            if c.is_adversarial and c.verified
            and c.entailment_score >= self.config.entailment_threshold
        )

        return {
            "total_citations": total_citations,
            "verified_citations": verified_citations,
            "valid_citations": valid_citations,
            "adversarial_citations": adversarial_citations,
            "adversarial_survived": adversarial_survived,
            "citation_precision": valid_citations / max(1, verified_citations),
            "adversary_success_rate": adversarial_survived / max(1, adversarial_citations),
            "passages_in_pool": len(self._passage_pool),
            "syntheses_completed": len(self._synthesis_results),
        }
