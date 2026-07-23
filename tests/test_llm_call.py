"""Tests for swarm/judges/llm_call.py — provider-dispatch helpers.

The provider call functions themselves require network + API keys and
are not exercised here. We test the surface that's deterministic and
provider-agnostic: retry semantics, API-key resolution.
"""

from __future__ import annotations

import pytest

from swarm.judges.llm_call import (
    PROVIDER_BASE_URLS,
    LLMCallResult,
    _resolve_api_key,
    call_with_retries,
    default_retryable,
)


class TestResolveApiKey:
    def test_explicit_wins(self, monkeypatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "from-env")
        assert _resolve_api_key("anthropic", "explicit") == "explicit"

    def test_falls_back_to_env(self, monkeypatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "from-env")
        assert _resolve_api_key("openai", None) == "from-env"

    def test_returns_none_when_missing(self, monkeypatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert _resolve_api_key("anthropic", None) is None

    def test_ollama_does_not_need_key(self) -> None:
        # Ollama is keyless; the function should just return None and the
        # caller (call_ollama) ignores the key entirely.
        assert _resolve_api_key("ollama", None) is None


class TestCallWithRetries:
    def _ok(self):
        return LLMCallResult(text="ok", input_tokens=1, output_tokens=1, latency_seconds=0.01)

    def test_no_retry_on_success(self) -> None:
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            return self._ok()

        result = call_with_retries(fn, max_retries=3, sleep=lambda _s: None)
        assert result.text == "ok"
        assert attempts["n"] == 1

    def test_retries_then_succeeds(self) -> None:
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise TimeoutError("transient")  # retryable
            return self._ok()

        result = call_with_retries(fn, max_retries=5, sleep=lambda _s: None)
        assert result.text == "ok"
        assert attempts["n"] == 3

    def test_exhausts_and_raises(self) -> None:
        def fn():
            raise TimeoutError("flaky network")  # retryable, exhausts budget

        with pytest.raises(TimeoutError, match="flaky network"):
            call_with_retries(fn, max_retries=2, sleep=lambda _s: None)

    def test_backoff_doubles(self) -> None:
        delays: list[float] = []

        def fn():
            raise TimeoutError("nope")  # retryable so backoff actually runs

        with pytest.raises(TimeoutError):
            call_with_retries(
                fn, max_retries=3, base_delay=1.0, sleep=lambda s: delays.append(s)
            )
        # delays should be [1, 2, 4]: 2^0, 2^1, 2^2 scaled by base_delay
        assert delays == [1.0, 2.0, 4.0]

    def test_fails_fast_on_non_retryable(self) -> None:
        # Terminal misconfiguration (missing key -> RuntimeError) must not
        # consume the retry budget; it should raise on the first attempt.
        attempts = {"n": 0}
        delays: list[float] = []

        def fn():
            attempts["n"] += 1
            raise RuntimeError("No Anthropic API key")

        with pytest.raises(RuntimeError, match="No Anthropic API key"):
            call_with_retries(
                fn, max_retries=5, sleep=lambda s: delays.append(s)
            )
        assert attempts["n"] == 1  # no retries
        assert delays == []  # never slept

    def test_custom_retryable_predicate(self) -> None:
        # A caller can opt a normally-terminal error back into retrying.
        attempts = {"n": 0}

        def fn():
            attempts["n"] += 1
            if attempts["n"] < 2:
                raise RuntimeError("retry me")
            return self._ok()

        result = call_with_retries(
            fn,
            max_retries=3,
            sleep=lambda _s: None,
            retryable=lambda err: isinstance(err, RuntimeError),
        )
        assert result.text == "ok"
        assert attempts["n"] == 2


class TestDefaultRetryable:
    def test_timeout_is_retryable(self) -> None:
        assert default_retryable(TimeoutError("slow")) is True

    def test_config_errors_are_terminal(self) -> None:
        assert default_retryable(RuntimeError("missing key")) is False
        assert default_retryable(ImportError("no sdk")) is False
        assert default_retryable(ValueError("bad provider")) is False

    def test_5xx_and_429_retryable_4xx_terminal(self) -> None:
        class _HttpErr(Exception):
            def __init__(self, status: int) -> None:
                self.status_code = status

        assert default_retryable(_HttpErr(503)) is True
        assert default_retryable(_HttpErr(429)) is True  # rate limit
        assert default_retryable(_HttpErr(401)) is False  # auth
        assert default_retryable(_HttpErr(400)) is False  # bad request

    def test_status_via_response_attribute(self) -> None:
        # httpx.HTTPStatusError exposes status on `.response`.
        class _Resp:
            status_code = 500

        class _Err(Exception):
            response = _Resp()

        assert default_retryable(_Err()) is True

    def test_retryable_by_class_name(self) -> None:
        # SDK exception types are matched by name without importing them.
        class APIConnectionError(Exception):
            pass

        assert default_retryable(APIConnectionError()) is True


class TestProviderBaseUrls:
    def test_known_compat_providers_have_endpoints(self) -> None:
        for provider in ("openrouter", "groq", "together", "deepseek"):
            assert provider in PROVIDER_BASE_URLS
            assert PROVIDER_BASE_URLS[provider].startswith("https://")

    def test_compat_provider_without_endpoint_is_rejected(self) -> None:
        # A provider with a resolvable key but no built-in/explicit endpoint
        # must be rejected rather than silently hitting api.openai.com. The
        # check runs before the SDK import, so it holds without `openai`.
        from swarm.judges import llm_call

        with pytest.raises(ValueError, match="no built-in endpoint"):
            llm_call.call_openai_compatible(
                model="m", prompt="p", api_key="explicit-key", provider="foo"
            )
