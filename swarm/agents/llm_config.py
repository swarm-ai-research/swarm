"""Configuration for LLM-backed agents."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional
from urllib.parse import urlparse


class LLMProvider(Enum):
    """Supported LLM API providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    GROQ = "groq"
    TOGETHER = "together"
    DEEPSEEK = "deepseek"
    GOOGLE = "google"
    LLAMA_CPP = "llama_cpp"


class PersonaType(Enum):
    """Pre-defined persona types for LLM agents."""

    HONEST = "honest"
    STRATEGIC = "strategic"
    ADVERSARIAL = "adversarial"
    OPEN = "open"  # Let the LLM develop its own strategy


@dataclass
class LLMConfig:
    """
    Configuration for an LLM-backed agent.

    Attributes:
        provider: The LLM API provider (anthropic, openai, ollama)
        model: Model identifier (e.g., "claude-sonnet-4-20250514", "gpt-4o")
        api_key: API key (None for ollama)
        base_url: Override URL for ollama or proxies
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens in response
        timeout: Request timeout in seconds
        max_retries: Maximum retry attempts for API failures
        retry_base_delay: Base delay for exponential backoff (seconds)
        persona: Pre-defined persona type
        system_prompt: Custom system prompt (overrides persona if set)
        cost_tracking: Whether to track API costs
        prompt_audit_path: Optional JSONL path to audit prompts/responses
        prompt_audit_include_system_prompt: If true, log full system prompt
        prompt_audit_hash_system_prompt: If true, log system prompt SHA256 hash
        prompt_audit_max_chars: Truncate logged fields to this size
    """

    provider: LLMProvider = LLMProvider.ANTHROPIC
    model: str = "claude-sonnet-4-20250514"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: float = 30.0
    max_retries: int = 3
    retry_base_delay: float = 1.0
    persona: PersonaType = PersonaType.OPEN
    system_prompt: Optional[str] = None
    cost_tracking: bool = True
    prompt_audit_path: Optional[str] = None
    prompt_audit_include_system_prompt: bool = False
    prompt_audit_hash_system_prompt: bool = True
    prompt_audit_max_chars: int = 20_000
    memori_config: Optional[Dict[str, Any]] = None

    # llama.cpp in-process options (Option B only)
    model_path: Optional[str] = None
    n_ctx: int = 4096
    n_threads: Optional[int] = None
    llama_seed: int = -1

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0.0 <= self.temperature <= 1.0:
            raise ValueError(
                f"temperature must be in [0.0, 1.0], got {self.temperature}"
            )

        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be positive, got {self.max_tokens}")

        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

        if self.max_retries < 0:
            raise ValueError(
                f"max_retries must be non-negative, got {self.max_retries}"
            )

        # Validate provider-specific requirements
        if self.provider == LLMProvider.OPENROUTER and not self.base_url:
            self.base_url = "https://openrouter.ai/api/v1"

        if self.provider == LLMProvider.GROQ and not self.base_url:
            self.base_url = "https://api.groq.com/openai/v1"

        if self.provider == LLMProvider.TOGETHER and not self.base_url:
            self.base_url = "https://api.together.xyz/v1"

        if self.provider == LLMProvider.DEEPSEEK and not self.base_url:
            self.base_url = "https://api.deepseek.com/v1"

        if self.provider not in (LLMProvider.OLLAMA, LLMProvider.OPENROUTER, LLMProvider.LLAMA_CPP) and not self.api_key:
            # Allow None: the LLM SDK will read from ANTHROPIC_API_KEY or OPENAI_API_KEY environment variables
            pass

        if self.provider == LLMProvider.OLLAMA and not self.base_url:
            self.base_url = "http://localhost:11434"

        if self.provider == LLMProvider.LLAMA_CPP and not self.base_url:
            self.base_url = "http://localhost:8080/v1"

        if self.prompt_audit_max_chars < 0:
            raise ValueError(
                f"prompt_audit_max_chars must be >= 0, got {self.prompt_audit_max_chars}"
            )

        # Validate llama_seed (must be -1 for random or >= 0 for deterministic)
        if self.llama_seed < -1:
            raise ValueError(
                f"llama_seed must be >= -1 (-1 = random), got {self.llama_seed}"
            )

        # Validate model_path (llama.cpp in-process)
        if self.model_path is not None:
            self._validate_model_path(self.model_path)

        # Validate base_url scheme (SSRF hardening)
        if self.base_url is not None:
            self._validate_base_url(self.base_url)

    @staticmethod
    def _validate_model_path(path: str) -> None:
        """Validate model_path to prevent path traversal and arbitrary file loads."""
        # Reject empty
        if not path.strip():
            raise ValueError("model_path must not be empty")

        # Reject traversal components (check raw path before normalization)
        if ".." in path.replace("\\", "/").split("/"):
            raise ValueError(
                f"model_path must not contain '..': {path}"
            )

        # Require .gguf extension
        if not path.lower().endswith(".gguf"):
            raise ValueError(
                f"model_path must end with .gguf: {path}"
            )

    @staticmethod
    def _validate_base_url(url: str) -> None:
        """Validate base_url scheme to prevent SSRF via file:// etc."""
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"base_url scheme must be http or https, got: {parsed.scheme!r}"
            )


@dataclass
class LLMUsageStats:
    """
    Tracks LLM API usage for cost monitoring.

    Attributes:
        total_requests: Number of API requests made
        total_input_tokens: Total input tokens across all requests
        total_output_tokens: Total output tokens across all requests
        total_retries: Number of retry attempts
        total_failures: Number of failed requests (after all retries)
        estimated_cost_usd: Estimated cost in USD
    """

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_retries: int = 0
    total_failures: int = 0
    estimated_cost_usd: float = 0.0

    # Cost per 1M tokens (approximate, as of 2024)
    _COST_PER_1M_INPUT: dict = field(
        default_factory=lambda: {
            "claude-sonnet-4-20250514": 3.0,
            "claude-3-5-sonnet-20241022": 3.0,
            "claude-3-haiku-20240307": 0.25,
            "gpt-4o": 2.50,
            "gpt-4o-mini": 0.15,
            "gpt-4-turbo": 10.0,
            # OpenRouter models (approximate)
            "mistralai/mixtral-8x7b-instruct": 0.24,
            "meta-llama/llama-3-70b-instruct": 0.59,
            "google/gemini-pro-1.5": 1.25,
            # Groq models
            "llama-3.3-70b-versatile": 0.59,
            "mixtral-8x7b-32768": 0.24,
            # Together models
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": 0.88,
            # DeepSeek models
            "deepseek-chat": 0.14,
            "deepseek-reasoner": 0.55,
            # Google Gemini models
            "gemini-2.0-flash": 0.10,
            "gemini-1.5-pro": 1.25,
        }
    )

    _COST_PER_1M_OUTPUT: dict = field(
        default_factory=lambda: {
            "claude-sonnet-4-20250514": 15.0,
            "claude-3-5-sonnet-20241022": 15.0,
            "claude-3-haiku-20240307": 1.25,
            "gpt-4o": 10.0,
            "gpt-4o-mini": 0.60,
            "gpt-4-turbo": 30.0,
            # OpenRouter models (approximate)
            "mistralai/mixtral-8x7b-instruct": 0.24,
            "meta-llama/llama-3-70b-instruct": 0.79,
            "google/gemini-pro-1.5": 5.0,
            # Groq models
            "llama-3.3-70b-versatile": 0.79,
            "mixtral-8x7b-32768": 0.24,
            # Together models
            "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo": 0.88,
            # DeepSeek models
            "deepseek-chat": 0.28,
            "deepseek-reasoner": 2.19,
            # Google Gemini models
            "gemini-2.0-flash": 0.40,
            "gemini-1.5-pro": 5.00,
        }
    )

    def record_usage(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        retries: int = 0,
        failed: bool = False,
    ) -> None:
        """Record usage from an API call."""
        self.total_requests += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_retries += retries

        if failed:
            self.total_failures += 1

        # Estimate cost
        input_cost = self._COST_PER_1M_INPUT.get(model, 0.0)
        output_cost = self._COST_PER_1M_OUTPUT.get(model, 0.0)

        self.estimated_cost_usd += (input_tokens / 1_000_000) * input_cost + (
            output_tokens / 1_000_000
        ) * output_cost

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_retries": self.total_retries,
            "total_failures": self.total_failures,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }
