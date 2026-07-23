"""Tests for swarm/judges/ — arm B scaffolding.

Heaviest test category is orthogonality (the load-bearing property): the
judge must NEVER see ProxyComputer observables.
"""

from __future__ import annotations

import pytest

from swarm.judges import (
    FORBIDDEN_FIELDS,
    RUBRIC_VERSION,
    JudgeView,
    LLMJudge,
    MockJudge,
    assert_view_is_orthogonal,
    bin_counts,
    load_rubric,
    make_view,
    stratified_sample,
)
from tests.fixtures.interactions import (
    generate_benign_batch,
    generate_obfuscation_scenario,
    generate_toxic_batch,
)


class TestRubric:
    def test_rubric_loads(self) -> None:
        text = load_rubric()
        assert "FROZEN" in text
        assert RUBRIC_VERSION in text

    def test_rubric_lists_all_forbidden_fields(self) -> None:
        text = load_rubric()
        # All proxy observables should be explicitly listed
        for field in [
            "task_progress_delta",
            "rework_count",
            "verifier_rejections",
            "tool_misuse_flags",
            "counterparty_engagement_delta",
        ]:
            assert field in text, f"rubric does not mention forbidden field {field}"


class TestOrthogonality:
    """Load-bearing tests. If any of these fail, the anchor is compromised."""

    def test_view_strips_top_level_forbidden_fields(self) -> None:
        # Build an interaction with a juicy task_progress_delta and confirm
        # it does not appear anywhere in the judge payload.
        batch = generate_toxic_batch(count=5, seed=0)
        for interaction in batch:
            view = make_view(interaction)
            payload = view.to_judge_payload()
            for forbidden in FORBIDDEN_FIELDS:
                assert forbidden not in payload, (
                    f"forbidden field {forbidden} leaked to top level"
                )
            assert_view_is_orthogonal(payload)

    def test_view_strips_forbidden_keys_from_metadata(self) -> None:
        # Even if a malicious caller stuffs a forbidden field into metadata,
        # make_view must scrub it.
        class FakeInteraction:
            interaction_id = "abc"
            interaction_type = "REPLY"
            accepted = True
            initiator = "a"
            counterparty = "b"
            metadata = {"v_hat": 0.99, "p": 0.99, "agent_type": "honest"}

        view = make_view(FakeInteraction())
        assert "v_hat" not in view.metadata
        assert "p" not in view.metadata
        assert view.metadata["agent_type"] == "honest"
        assert_view_is_orthogonal(view.to_judge_payload())

    def test_assert_raises_on_forbidden_top_level(self) -> None:
        with pytest.raises(AssertionError):
            assert_view_is_orthogonal(
                {"interaction_id": "x", "task_progress_delta": 0.5}
            )

    def test_assert_raises_on_forbidden_in_metadata(self) -> None:
        with pytest.raises(AssertionError):
            assert_view_is_orthogonal({"metadata": {"rework_count": 2}})

    def test_judge_score_only_sees_view(self) -> None:
        # MockJudge should never crash with KeyError or similar even on a
        # toxic-fixture batch with rich observables.
        batch = generate_toxic_batch(count=10, seed=0)
        judge = MockJudge()
        for interaction in batch:
            view = make_view(interaction)
            verdict = judge.score(view)
            assert 0.0 <= verdict.score <= 1.0
            assert verdict.judge_name == "mock"
            assert verdict.rubric_version == RUBRIC_VERSION


class TestMockJudgeRubricFidelity:
    """MockJudge must implement the rubric's signal-interpretation rules."""

    def test_empty_metadata_gives_midscore(self) -> None:
        view = JudgeView(
            interaction_id="x",
            interaction_type="REPLY",
            accepted=True,
            initiator_label="a",
            counterparty_label="b",
            metadata={},
        )
        assert MockJudge().score(view).score == 0.5

    def test_honest_scores_high(self) -> None:
        view = JudgeView(
            interaction_id="x",
            interaction_type="COLLABORATION",
            accepted=True,
            initiator_label="honest_1",
            counterparty_label="honest_2",
            metadata={"agent_type": "honest"},
        )
        assert MockJudge().score(view).score >= 0.7

    def test_policy_obfuscation_scores_low_despite_innocuous_rationalization(self) -> None:
        # The dangerous case: high p (proxy fooled), but the agent_type
        # label tells us the truth. Judge must score LOW.
        view = JudgeView(
            interaction_id="x",
            interaction_type="TRADE",
            accepted=True,
            initiator_label="policy_obf_1",
            counterparty_label="honest_5",
            metadata={
                "agent_type": "policy_obfuscation",
                "rationalization": "Optimized for efficiency.",
                "rationalization_strength": 0.7,
            },
        )
        assert MockJudge().score(view).score <= 0.3

    def test_self_optimizer_with_deep_optimization_scores_lower(self) -> None:
        shallow = JudgeView(
            interaction_id="a", interaction_type="TRADE", accepted=True,
            initiator_label="opt_1", counterparty_label="h_1",
            metadata={"agent_type": "self_optimizer", "optimization_depth": 0},
        )
        deep = JudgeView(
            interaction_id="b", interaction_type="TRADE", accepted=True,
            initiator_label="opt_1", counterparty_label="h_1",
            metadata={"agent_type": "self_optimizer", "optimization_depth": 15},
        )
        assert MockJudge().score(deep).score < MockJudge().score(shallow).score

    def test_self_optimizer_float_depth_is_honored_v3(self) -> None:
        # Regression: float optimization_depth was silently treated as 0,
        # producing the loosest target (~0.45) instead of depth-scaled
        # escalation. Real-world metadata routinely carries floats.
        judge = MockJudge(rubric_version="rubric.v3")
        int_deep = JudgeView(
            interaction_id="a", interaction_type="TRADE", accepted=True,
            initiator_label="opt_1", counterparty_label="h_1",
            metadata={"agent_type": "self_optimizer", "optimization_depth": 15},
        )
        float_deep = JudgeView(
            interaction_id="b", interaction_type="TRADE", accepted=True,
            initiator_label="opt_1", counterparty_label="h_1",
            metadata={"agent_type": "self_optimizer", "optimization_depth": 15.0},
        )
        assert judge.score(float_deep).score == judge.score(int_deep).score

    def test_blatant_scores_low(self) -> None:
        view = JudgeView(
            interaction_id="x", interaction_type="TRADE", accepted=True,
            initiator_label="blatant_1", counterparty_label="honest_5",
            metadata={"agent_type": "blatant"},
        )
        assert MockJudge().score(view).score <= 0.2

    def test_deterministic(self) -> None:
        view = JudgeView(
            interaction_id="x", interaction_type="TRADE", accepted=True,
            initiator_label="a", counterparty_label="b",
            metadata={"agent_type": "honest"},
        )
        judge = MockJudge()
        a = judge.score(view)
        b = judge.score(view)
        assert a.score == b.score
        assert a.rationale == b.rationale


class TestLLMJudge:
    def _view(self, iid: str = "x") -> JudgeView:
        return JudgeView(
            interaction_id=iid,
            interaction_type="REPLY",
            accepted=True,
            initiator_label="a",
            counterparty_label="b",
            metadata={"agent_type": "honest"},
        )

    def _result(self, text: str):
        from swarm.judges.llm_call import LLMCallResult
        return LLMCallResult(text=text, input_tokens=10, output_tokens=10, latency_seconds=0.01)

    def test_parses_clean_json(self) -> None:
        judge = LLMJudge(
            name="claude",
            caller=lambda _p: self._result('{"score": 0.83, "rationale": "looks honest"}'),
        )
        verdict = judge.score(self._view())
        assert verdict.score == 0.83
        assert verdict.rationale == "looks honest"
        assert verdict.judge_name == "claude"
        assert verdict.interaction_id == "x"

    def test_parses_markdown_fenced_json(self) -> None:
        judge = LLMJudge(
            name="gpt",
            caller=lambda _p: self._result(
                'Here is my verdict:\n\n```json\n{"score": 0.42, "rationale": "mid"}\n```\n'
            ),
        )
        assert judge.score(self._view()).score == 0.42

    def test_parses_narrated_response(self) -> None:
        judge = LLMJudge(
            name="llama",
            caller=lambda _p: self._result(
                'Looking at this interaction, I would say: {"score": 0.1, "rationale": "low"}'
            ),
        )
        assert judge.score(self._view()).score == 0.1

    def test_parses_nested_json_object(self) -> None:
        # A score object with a nested value must still parse — the old
        # flat [^{}]* regex hard-stopped at the first inner brace and
        # dropped the row.
        judge = LLMJudge(
            name="claude",
            caller=lambda _p: self._result(
                '{"score": 0.5, "rationale": "mixed", '
                '"evidence": {"agent_type": "honest", "flags": {"x": 1}}}'
            ),
        )
        verdict = judge.score(self._view())
        assert verdict.score == 0.5
        assert verdict.rationale == "mixed"

    def test_parses_nested_json_inside_fence_with_narration(self) -> None:
        # Nested object, wrapped in a markdown fence, with leading prose.
        judge = LLMJudge(
            name="gpt",
            caller=lambda _p: self._result(
                'My verdict:\n```json\n'
                '{"score": 0.2, "rationale": "see evidence", '
                '"evidence": {"reasons": ["a", "b"]}}\n```'
            ),
        )
        assert judge.score(self._view()).score == 0.2

    def test_clamps_out_of_range_score(self) -> None:
        judge = LLMJudge(
            name="claude",
            caller=lambda _p: self._result('{"score": 1.5, "rationale": "off"}'),
        )
        assert judge.score(self._view()).score == 1.0

    def test_raises_on_garbage_response(self) -> None:
        judge = LLMJudge(
            name="claude",
            caller=lambda _p: self._result("Not a JSON object at all."),
        )
        with pytest.raises(ValueError, match="failed to parse response"):
            judge.score(self._view())

    def test_recovers_score_from_truncated_response(self) -> None:
        # Real-world failure: a model emits valid JSON whose rationale
        # exceeds max_tokens and gets cut mid-string. The score is
        # recoverable; the rationale gets a [truncated] marker.
        truncated = ' {\n  "score": 0.1,\n  "rationale": "The interaction is a TRADE between an init'
        judge = LLMJudge(name="claude", caller=lambda _p: self._result(truncated))
        verdict = judge.score(self._view())
        assert verdict.score == 0.1
        assert "[truncated]" in verdict.rationale

    def test_recovers_score_when_rationale_has_no_value(self) -> None:
        # Truncation right after the score field with no rationale value.
        truncated = '{"score": 0.85, "rati'
        judge = LLMJudge(name="claude", caller=lambda _p: self._result(truncated))
        verdict = judge.score(self._view())
        assert verdict.score == 0.85
        assert "[response truncated" in verdict.rationale

    def test_retries_on_transient_error_then_succeeds(self) -> None:
        attempts = {"n": 0}

        def flaky(_prompt: str):
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise TimeoutError("transient")  # retryable
            return self._result('{"score": 0.7, "rationale": "ok"}')

        # Patch sleep so the test doesn't actually wait.
        from swarm.judges import llm_call
        original = llm_call.time.sleep
        llm_call.time.sleep = lambda _s: None
        try:
            judge = LLMJudge(name="claude", caller=flaky, max_retries=3)
            verdict = judge.score(self._view())
        finally:
            llm_call.time.sleep = original

        assert verdict.score == 0.7
        assert attempts["n"] == 2

    def test_orthogonality_enforced_in_prompt_build(self) -> None:
        # If a JudgeView slipped a forbidden field through, the prompt
        # builder must blow up before any LLM call.
        judge = LLMJudge(name="claude", caller=lambda _p: self._result('{"score":0.5}'))
        bad = JudgeView(
            interaction_id="x",
            interaction_type="REPLY",
            accepted=True,
            initiator_label="a",
            counterparty_label="b",
            metadata={"v_hat": 0.9},  # forbidden
        )
        with pytest.raises(AssertionError):
            judge._build_prompt(bad)

    def test_dispatch_rejects_unknown_provider(self) -> None:
        judge = LLMJudge(name="x", provider="not-a-real-provider")
        with pytest.raises(ValueError, match="unsupported provider"):
            judge._dispatch("prompt")

    @pytest.mark.parametrize(
        ("provider", "expected_fn"),
        [
            ("anthropic", "call_anthropic"),
            ("openai", "call_openai_compatible"),
            ("openrouter", "call_openai_compatible"),
            ("groq", "call_openai_compatible"),
            ("together", "call_openai_compatible"),
            ("deepseek", "call_openai_compatible"),
            ("ollama", "call_ollama"),
        ],
    )
    def test_dispatch_routes_to_expected_provider(
        self, monkeypatch, provider: str, expected_fn: str
    ) -> None:
        # Without `caller`, _dispatch must route each provider string to the
        # right call_* helper. Patching the helpers in the judge module keeps
        # the dispatch table honest at HEAD (a typo'd provider would either
        # hit the wrong helper or fall through to the ValueError).
        from swarm.judges import judge as judge_mod

        called = {"fn": None}

        def make_stub(label: str):
            def stub(**kwargs):
                called["fn"] = label
                return self._result('{"score": 0.5, "rationale": "ok"}')
            return stub

        for fn_name in ("call_anthropic", "call_openai_compatible", "call_ollama"):
            monkeypatch.setattr(judge_mod, fn_name, make_stub(fn_name))

        judge = LLMJudge(name="j", provider=provider, base_url="http://x")
        judge._dispatch("prompt")
        assert called["fn"] == expected_fn


class TestSampler:
    def test_stratification_respects_per_bin_cap(self) -> None:
        # 100 benign (high p) + 100 toxic (low p)
        benign = generate_benign_batch(count=100, acceptance_rate=1.0, seed=1)
        toxic = generate_toxic_batch(count=100, acceptance_rate=1.0, seed=2)
        sampled = stratified_sample(benign + toxic, per_bin=10, seed=42)
        # Bins are [0, .2, .4, .6, .8, 1.0]. We should get <=10 per
        # populated bin and the sample size should be modest.
        assert len(sampled) <= 50
        counts = bin_counts(sampled)
        for c in counts:
            assert c <= 10

    def test_only_accepted_returned(self) -> None:
        # Mix with low acceptance rate; sampler should filter to accepted only.
        batch = generate_toxic_batch(count=100, acceptance_rate=0.3, seed=3)
        sampled = stratified_sample(batch, per_bin=50, seed=0)
        for s in sampled:
            assert s.accepted is True

    def test_reproducible_with_seed(self) -> None:
        batch = generate_obfuscation_scenario(n_epochs=2, seed=5)
        flat = [i for epoch in batch for i in epoch]
        a = stratified_sample(flat, per_bin=5, seed=99)
        b = stratified_sample(flat, per_bin=5, seed=99)
        assert [i.interaction_id for i in a] == [i.interaction_id for i in b]


class TestMockJudgeJitter:
    """Seeded jitter for inter-rater (arm C) machinery runs."""

    def _view(self) -> JudgeView:
        return JudgeView(
            interaction_id="jitter-test-1",
            interaction_type="COLLABORATION",
            accepted=True,
            initiator_label="honest_1",
            counterparty_label="honest_2",
            metadata={"agent_type": "honest"},
        )

    def test_zero_sigma_is_exact_rubric_scoring(self) -> None:
        view = self._view()
        base = MockJudge(rubric_version="rubric.v1").score(view)
        noiseless = MockJudge(
            name="mock_b", rubric_version="rubric.v1", noise_sigma=0.0
        ).score(view)
        assert noiseless.score == base.score

    def test_jitter_is_deterministic_per_seed(self) -> None:
        view = self._view()
        judge = MockJudge(
            name="mock_b", rubric_version="rubric.v1", noise_sigma=0.05, noise_seed=2
        )
        again = MockJudge(
            name="mock_b", rubric_version="rubric.v1", noise_sigma=0.05, noise_seed=2
        )
        assert judge.score(view).score == again.score(view).score

    def test_jitter_differs_across_judges_and_stays_bounded(self) -> None:
        view = self._view()
        base = MockJudge(rubric_version="rubric.v1").score(view).score
        scores = {
            name: MockJudge(
                name=name, rubric_version="rubric.v1", noise_sigma=0.08, noise_seed=i
            ).score(view).score
            for i, name in enumerate(["mock_b", "mock_c"], start=2)
        }
        assert all(0.0 <= s <= 1.0 for s in scores.values())
        assert any(s != base for s in scores.values())
