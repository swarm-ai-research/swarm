"""Self-contained synchronous LLM dispatch for the judge backend.

Kept separate from `swarm/agents/llm_agent.py` so the judge doesn't pull in
the agent abstraction (which carries reputation, memory, action parsing,
async loops, and more — none of it relevant to one-shot judge calls).

Three providers covered, matching the calibration pre-reg's "Claude /
GPT-4o-mini / Llama" requirement:

- Anthropic (Claude)
- OpenAI / OpenAI-compatible (GPT-4o-mini, OpenRouter, Groq, ...)
- Ollama (local Llama)

Each function returns the model's raw text. JSON parsing and the
rubric-shaped response handling live in `LLMJudge.score`.

Temperature defaults to 0 per the rubric's "Determinism" section so
re-running the same prompt yields the same score. The judge can also
swap to a temperature-0-by-default provider via `LLMConfig.temperature`.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Callable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LLMCallResult:
    """One LLM round-trip's output."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_seconds: float


def _resolve_api_key(provider: str, explicit: Optional[str]) -> Optional[str]:
    """Resolve API key from explicit arg first, then env vars."""
    if explicit:
        return explicit
    env_vars = {
        "anthropic": ("ANTHROPIC_API_KEY",),
        "openai": ("OPENAI_API_KEY",),
        "openrouter": ("OPENROUTER_API_KEY",),
        "groq": ("GROQ_API_KEY",),
        "together": ("TOGETHER_API_KEY",),
        "deepseek": ("DEEPSEEK_API_KEY",),
        "ollama": (),
    }
    for var in env_vars.get(provider, ()):
        v = os.environ.get(var)
        if v:
            return v
    return None


# Default OpenAI-compatible endpoints for providers that are NOT OpenAI
# itself. Without these, openai.OpenAI(base_url=None) silently targets
# api.openai.com, so a `groq`/`openrouter`/... key would be sent to the
# wrong host and 401/404. An explicit base_url always overrides these.
PROVIDER_BASE_URLS: dict[str, str] = {
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
    "deepseek": "https://api.deepseek.com",
}


def call_anthropic(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
) -> LLMCallResult:
    """One-shot Anthropic message call. Returns raw text + token usage.

    Raises ImportError if `anthropic` is not installed; the caller is
    expected to surface the error rather than fall back silently.
    """
    try:
        import anthropic
    except ImportError as err:
        raise ImportError(
            "anthropic package not installed. "
            "Install with: python -m pip install anthropic"
        ) from err

    key = _resolve_api_key("anthropic", api_key)
    if not key:
        raise RuntimeError(
            "No Anthropic API key. Set ANTHROPIC_API_KEY or pass api_key="
        )
    client = anthropic.Anthropic(api_key=key, timeout=timeout)
    started = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.monotonic() - started
    # response.content is a list of typed blocks; pick the first text block.
    text = ""
    for block in response.content:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            break
    return LLMCallResult(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        latency_seconds=elapsed,
    )


def call_openai_compatible(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float = 60.0,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    provider: str = "openai",
) -> LLMCallResult:
    """One-shot OpenAI-style chat completion.

    Works for OpenAI, OpenRouter, Groq, Together, DeepSeek — anything
    that exposes the OpenAI chat API at `base_url`. Set `provider` to
    drive the env-var fallback (`OPENAI_API_KEY` for "openai",
    `OPENROUTER_API_KEY` for "openrouter", `GROQ_API_KEY` for "groq",
    etc.). Use `call_anthropic` for Anthropic — its API is not
    OpenAI-chat compatible.
    """
    key = _resolve_api_key(provider, api_key)
    if not key:
        raise RuntimeError(
            f"No {provider} API key. Set the env var or pass api_key="
        )
    # Non-OpenAI providers need their own endpoint; without one the OpenAI
    # SDK would target api.openai.com with the wrong key. Fall back to a
    # known default, and reject any other non-openai provider that hasn't
    # supplied an explicit base_url rather than silently mis-routing.
    # Validated before importing the SDK so misconfiguration fails fast even
    # where `openai` isn't installed.
    resolved_base_url = base_url or PROVIDER_BASE_URLS.get(provider)
    if resolved_base_url is None and provider != "openai":
        raise ValueError(
            f"provider {provider!r} has no built-in endpoint; pass base_url= "
            f"explicitly (known defaults: {sorted(PROVIDER_BASE_URLS)})"
        )

    try:
        import openai
    except ImportError as err:
        raise ImportError(
            "openai package not installed. "
            "Install with: python -m pip install openai"
        ) from err

    client = openai.OpenAI(api_key=key, base_url=resolved_base_url, timeout=timeout)
    started = time.monotonic()
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed = time.monotonic() - started
    text = response.choices[0].message.content or ""
    return LLMCallResult(
        text=text,
        input_tokens=response.usage.prompt_tokens if response.usage else 0,
        output_tokens=response.usage.completion_tokens if response.usage else 0,
        latency_seconds=elapsed,
    )


def call_ollama(
    *,
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    timeout: float = 60.0,
    base_url: Optional[str] = None,
) -> LLMCallResult:
    """One-shot Ollama chat call. No API key needed."""
    try:
        import httpx
    except ImportError as err:
        raise ImportError(
            "httpx package not installed. Install with: python -m pip install httpx"
        ) from err

    url = f"{(base_url or 'http://localhost:11434').rstrip('/')}/api/chat"
    started = time.monotonic()
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            url,
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
        )
        response.raise_for_status()
        data = response.json()
    elapsed = time.monotonic() - started
    return LLMCallResult(
        text=data["message"]["content"],
        input_tokens=int(data.get("prompt_eval_count", 0)),
        output_tokens=int(data.get("eval_count", 0)),
        latency_seconds=elapsed,
    )


# Exception *class names* (matched by name, so the optional SDKs don't have
# to be imported here) that represent transient failures worth retrying.
_RETRYABLE_EXC_NAMES = frozenset(
    {
        # openai / anthropic SDK
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        # httpx
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "WriteTimeout",
        "PoolTimeout",
        "TimeoutException",
        "RemoteProtocolError",
    }
)


def default_retryable(err: BaseException) -> bool:
    """True only for transient failures (network / timeout / 5xx / 429).

    Fail fast on everything else: a missing SDK (``ImportError``), a missing
    key or other misconfiguration (``RuntimeError`` / ``ValueError``), or a
    4xx auth/quota error should surface immediately instead of burning the
    full retry budget behind transient-looking warnings.
    """
    if isinstance(err, TimeoutError):
        return True
    # HTTP status, if the exception carries one (openai.APIStatusError exposes
    # `.status_code`; httpx.HTTPStatusError exposes `.response.status_code`).
    status = getattr(err, "status_code", None)
    if status is None:
        status = getattr(getattr(err, "response", None), "status_code", None)
    if isinstance(status, int):
        return status >= 500 or status == 429
    return type(err).__name__ in _RETRYABLE_EXC_NAMES


def call_with_retries(
    fn: "Callable[[], LLMCallResult]",
    *,
    max_retries: int = 3,
    base_delay: float = 1.0,
    sleep: "Callable[[float], None]" = time.sleep,
    retryable: "Callable[[BaseException], bool]" = default_retryable,
) -> LLMCallResult:
    """Exponential-backoff wrapper for any of the call_* functions.

    `sleep` is injectable so tests can run without wall-clock waits.
    Only exceptions for which ``retryable(err)`` is true are retried — by
    default that's transient network/timeout/5xx/429 failures. Terminal
    errors (missing SDK or key, 4xx auth/quota, programming bugs) are
    re-raised on the first attempt so misconfiguration surfaces fast.
    """
    last: Optional[BaseException] = None
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except Exception as err:
            last = err
            if not retryable(err):
                logger.warning("LLM call failed with non-retryable error: %s", err)
                raise
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries + 1, err)
            if attempt < max_retries:
                sleep(base_delay * (2 ** attempt))
    raise last if last is not None else RuntimeError("LLM call failed with no exception captured")
