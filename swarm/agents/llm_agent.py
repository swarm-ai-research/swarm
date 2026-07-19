"""LLM-backed agent implementation."""

import asyncio
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from swarm.agents.base import (
    Action,
    ActionType,
    BaseAgent,
    InteractionProposal,
    Observation,
    Role,
)
from swarm.agents.llm_config import LLMConfig, LLMProvider, LLMUsageStats
from swarm.agents.llm_prompts import (
    build_accept_prompt,
    build_action_prompt,
    build_kernel_action_prompt,
)
from swarm.logging.prompt_audit import PromptAuditConfig, PromptAuditLog
from swarm.models.agent import AgentType
from swarm.models.interaction import InteractionType

logger = logging.getLogger(__name__)


class LLMAgent(BaseAgent):
    """
    Agent backed by an LLM API call.

    This agent uses an LLM to decide actions based on observations.
    Supports Anthropic, OpenAI, and Ollama providers.
    """

    def __init__(
        self,
        agent_id: str,
        llm_config: LLMConfig,
        roles: Optional[List[Role]] = None,
        config: Optional[Dict] = None,
        name: Optional[str] = None,
    ):
        """
        Initialize LLM agent.

        Args:
            agent_id: Unique identifier
            llm_config: LLM API configuration
            roles: List of roles this agent can fulfill
            config: Additional agent configuration
            name: Human-readable name for the agent
        """
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.HONEST,  # Default, behavior comes from LLM
            roles=roles,
            config=config,
            name=name,
        )

        self.llm_config = llm_config
        self.usage_stats = LLMUsageStats()

        # Resolve API key from environment if not provided
        self._api_key = llm_config.api_key or self._get_api_key_from_env()

        # Prompt audit logging (lazy init)
        self._prompt_audit_log: Optional[PromptAuditLog] = None

        # Lazy-loaded clients
        self._anthropic_client = None
        self._openai_client = None
        # Llama CPP can use either: (A) llama-cpp-python direct client, or (B) OpenAI-compatible API
        self._llama_model = None  # llama-cpp-python Llama instance (Option A)

        # Memori middleware (lazy-loaded from config dict)
        self._memori_middleware = None
        self._memori_config_raw = llm_config.memori_config

    def _get_prompt_audit_log(self) -> Optional[PromptAuditLog]:
        """Lazy-create prompt audit log if enabled."""
        if not self.llm_config.prompt_audit_path:
            return None
        if self._prompt_audit_log is not None:
            return self._prompt_audit_log

        cfg = PromptAuditConfig(
            path=Path(self.llm_config.prompt_audit_path),
            include_system_prompt=self.llm_config.prompt_audit_include_system_prompt,
            hash_system_prompt=self.llm_config.prompt_audit_hash_system_prompt,
            max_chars=self.llm_config.prompt_audit_max_chars,
        )
        self._prompt_audit_log = PromptAuditLog(cfg)
        return self._prompt_audit_log

    def _audit_llm_exchange(
        self,
        *,
        kind: str,
        observation: Observation,
        system_prompt: str,
        user_prompt: str,
        response_text: Optional[str],
        input_tokens: Optional[int],
        output_tokens: Optional[int],
        error: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        audit = self._get_prompt_audit_log()
        if audit is None:
            return
        audit.append_exchange(
            agent_id=self.agent_id,
            kind=kind,
            epoch=getattr(observation, "current_epoch", None),
            step=getattr(observation, "current_step", None),
            provider=self.llm_config.provider.value,
            model=self.llm_config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_text=response_text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            error=error,
            extra=extra,
        )

    def _get_api_key_from_env(self) -> Optional[str]:
        """Get API key from environment variables."""
        if self.llm_config.provider == LLMProvider.ANTHROPIC:
            return os.environ.get("ANTHROPIC_API_KEY")
        elif self.llm_config.provider == LLMProvider.OPENAI:
            return os.environ.get("OPENAI_API_KEY")
        elif self.llm_config.provider == LLMProvider.OPENROUTER:
            return os.environ.get("OPENROUTER_API_KEY")
        elif self.llm_config.provider == LLMProvider.GROQ:
            return os.environ.get("GROQ_API_KEY")
        elif self.llm_config.provider == LLMProvider.TOGETHER:
            return os.environ.get("TOGETHER_API_KEY")
        elif self.llm_config.provider == LLMProvider.DEEPSEEK:
            return os.environ.get("DEEPSEEK_API_KEY")
        elif self.llm_config.provider == LLMProvider.GOOGLE:
            return os.environ.get("GOOGLE_API_KEY")
        elif self.llm_config.provider == LLMProvider.LLAMA_CPP:
            # llama-server ignores API keys, but OpenAI client requires non-empty
            return "not-needed"
        return None

    def _get_memori_middleware(self):
        """Lazy-init Memori middleware from raw config dict."""
        if self._memori_middleware is not None:
            return self._memori_middleware
        if self._memori_config_raw is None:
            return None
        from swarm.agents.memori_middleware import MemoriConfig, MemoriMiddleware

        config = MemoriConfig.from_dict(self._memori_config_raw)
        if not config.enabled:
            return None
        self._memori_middleware = MemoriMiddleware(config, agent_id=self.agent_id)
        return self._memori_middleware

    def _get_anthropic_client(self):
        """Lazy-load Anthropic client."""
        if self._anthropic_client is None:
            try:
                import anthropic

                self._anthropic_client = anthropic.Anthropic(
                    api_key=self._api_key,
                    timeout=self.llm_config.timeout,
                )
            except ImportError as err:
                raise ImportError(
                    "anthropic package not installed. "
                    "Install with: python -m pip install anthropic"
                ) from err
            middleware = self._get_memori_middleware()
            if middleware is not None:
                middleware.wrap_client(self._anthropic_client)
        return self._anthropic_client

    def _get_openai_client(self):
        """Lazy-load OpenAI client."""
        if self._openai_client is None:
            try:
                import openai

                self._openai_client = openai.OpenAI(
                    api_key=self._api_key,
                    timeout=self.llm_config.timeout,
                )
            except ImportError as err:
                raise ImportError(
                    "openai package not installed. "
                    "Install with: python -m pip install openai"
                ) from err
            middleware = self._get_memori_middleware()
            if middleware is not None:
                middleware.wrap_client(self._openai_client)
        return self._openai_client

    async def _call_llm_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """
        Make an async LLM API call with retry logic.

        Args:
            system_prompt: System message
            user_prompt: User message

        Returns:
            Tuple of (response_text, input_tokens, output_tokens)

        Raises:
            Exception: If all retries fail
        """
        last_error = None
        retries = 0

        for attempt in range(self.llm_config.max_retries + 1):
            try:
                if self.llm_config.provider == LLMProvider.ANTHROPIC:
                    return await self._call_anthropic_async(system_prompt, user_prompt)
                elif self.llm_config.provider == LLMProvider.OPENAI:
                    return await self._call_openai_async(system_prompt, user_prompt)
                elif self.llm_config.provider == LLMProvider.OPENROUTER:
                    return await self._call_openrouter_async(system_prompt, user_prompt)
                elif self.llm_config.provider == LLMProvider.OLLAMA:
                    return await self._call_ollama_async(system_prompt, user_prompt)
                elif self.llm_config.provider in (
                    LLMProvider.GROQ,
                    LLMProvider.TOGETHER,
                    LLMProvider.DEEPSEEK,
                ):
                    return await self._call_openai_compatible_async(
                        system_prompt, user_prompt
                    )
                elif self.llm_config.provider == LLMProvider.GOOGLE:
                    return await self._call_google_async(system_prompt, user_prompt)
                elif self.llm_config.provider == LLMProvider.LLAMA_CPP:
                    if self.llm_config.model_path:
                        return await self._call_llama_cpp_direct_async(
                            system_prompt, user_prompt
                        )
                    else:
                        return await self._call_openai_compatible_async(
                            system_prompt, user_prompt
                        )
                else:
                    raise ValueError(f"Unknown provider: {self.llm_config.provider}")

            except Exception as e:
                last_error = e
                retries += 1
                logger.warning(f"LLM call failed (attempt {attempt + 1}): {e}")

                if attempt < self.llm_config.max_retries:
                    delay = self.llm_config.retry_base_delay * (2**attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted
        self.usage_stats.record_usage(
            model=self.llm_config.model,
            input_tokens=0,
            output_tokens=0,
            retries=retries,
            failed=True,
        )
        raise last_error or Exception("LLM call failed")

    async def _call_anthropic_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call Anthropic API."""
        client = self._get_anthropic_client()

        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.messages.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            ),
        )

        text = response.content[0].text
        input_tokens = response.usage.input_tokens if response.usage else 0
        output_tokens = response.usage.output_tokens if response.usage else 0

        if self.llm_config.cost_tracking:
            self.usage_stats.record_usage(
                model=self.llm_config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return text, input_tokens, output_tokens

    async def _call_openai_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call OpenAI API."""
        client = self._get_openai_client()

        # Run sync client in thread pool
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            ),
        )

        text = response.choices[0].message.content
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        if self.llm_config.cost_tracking:
            self.usage_stats.record_usage(
                model=self.llm_config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return text, input_tokens, output_tokens

    async def _call_openai_compatible_async(
        self,
        system_prompt: str,
        user_prompt: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> tuple[str, int, int]:
        """Call an OpenAI-compatible API (OpenRouter, Groq, Together, DeepSeek)."""
        try:
            import openai
        except ImportError as err:
            raise ImportError(
                "openai package not installed. "
                "Install with: python -m pip install openai"
            ) from err

        client = openai.OpenAI(
            api_key=self._api_key,
            base_url=self.llm_config.base_url,
            timeout=self.llm_config.timeout,
            default_headers=extra_headers or {},
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=self.llm_config.model,
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            ),
        )

        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        if self.llm_config.cost_tracking:
            self.usage_stats.record_usage(
                model=self.llm_config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return text, input_tokens, output_tokens

    async def _call_openrouter_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call OpenRouter API (OpenAI-compatible with custom headers)."""
        return await self._call_openai_compatible_async(
            system_prompt,
            user_prompt,
            extra_headers={
                "HTTP-Referer": "https://github.com/swarm-ai-safety",
                "X-Title": "SWARM Council",
            },
        )

    async def _call_google_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call Google Gemini API via the google-genai SDK."""
        try:
            from google import genai
        except ImportError as err:
            raise ImportError(
                "google-genai package not installed. "
                "Install with: python -m pip install google-genai"
            ) from err

        client = genai.Client(
            api_key=self._api_key,
            http_options={"timeout": int(self.llm_config.timeout)},
        )

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.models.generate_content(
                model=self.llm_config.model,
                contents=user_prompt,
                config={
                    "system_instruction": system_prompt,
                    "max_output_tokens": self.llm_config.max_tokens,
                    "temperature": self.llm_config.temperature,
                },
            ),
        )

        text = response.text or ""
        input_tokens = getattr(
            getattr(response, "usage_metadata", None), "prompt_token_count", 0
        ) or 0
        output_tokens = getattr(
            getattr(response, "usage_metadata", None), "candidates_token_count", 0
        ) or 0

        if self.llm_config.cost_tracking:
            self.usage_stats.record_usage(
                model=self.llm_config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return text, input_tokens, output_tokens

    async def _call_ollama_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call Ollama API."""
        try:
            import httpx
        except ImportError as err:
            raise ImportError(
                "httpx package not installed. Install with: python -m pip install httpx"
            ) from err

        base_url = self.llm_config.base_url or "http://localhost:11434"
        url = f"{base_url}/api/chat"

        async with httpx.AsyncClient(timeout=self.llm_config.timeout) as client:
            response = await client.post(
                url,
                json={
                    "model": self.llm_config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.llm_config.temperature,
                        "num_predict": self.llm_config.max_tokens,
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        text = data["message"]["content"]
        # Ollama doesn't always provide token counts
        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        if self.llm_config.cost_tracking:
            self.usage_stats.record_usage(
                model=self.llm_config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return text, input_tokens, output_tokens

    def _get_llama_model(self):
        """Lazy-load llama-cpp-python Llama model for in-process inference."""
        if self._llama_model is not None:
            return self._llama_model
        try:
            from llama_cpp import Llama
        except ImportError as err:
            raise ImportError(
                "llama-cpp-python package not installed. "
                "Install with: python -m pip install llama-cpp-python"
            ) from err

        if not self.llm_config.model_path:
            raise ValueError(
                "model_path is required for in-process llama.cpp inference"
            )

        self._llama_model = Llama(
            model_path=self.llm_config.model_path,
            n_ctx=self.llm_config.n_ctx,
            n_threads=self.llm_config.n_threads,
            seed=self.llm_config.llama_seed,
            verbose=False,
        )
        return self._llama_model

    async def _call_llama_cpp_direct_async(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """Call llama-cpp-python in-process (Option B)."""
        model = self._get_llama_model()

        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.create_chat_completion(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=self.llm_config.max_tokens,
                temperature=self.llm_config.temperature,
            ),
        )

        text = response["choices"][0]["message"]["content"] or ""
        usage = response.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        if self.llm_config.cost_tracking:
            self.usage_stats.record_usage(
                model=self.llm_config.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        return text, input_tokens, output_tokens

    def _call_llm_sync(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, int, int]:
        """
        Synchronous wrapper for LLM calls.

        Used by the sync act() method when no event loop is running.
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in an async context, create a new task
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self._call_llm_async(system_prompt, user_prompt)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(
                    self._call_llm_async(system_prompt, user_prompt)
                )
        except RuntimeError:
            # No event loop, create one
            return asyncio.run(self._call_llm_async(system_prompt, user_prompt))

    def _parse_action_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response into action dictionary.

        Args:
            response: Raw LLM response text

        Returns:
            Parsed action dictionary
        """
        # Try to extract JSON from response
        # Handle markdown code blocks
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", response)
        if json_match:
            json_str = json_match.group(1).strip()
        else:
            # Try to find raw JSON
            json_match = re.search(r"\{[\s\S]*\}", response)
            if json_match:
                json_str = json_match.group(0)
            else:
                raise ValueError("No JSON found in response")

        try:
            result: Dict[str, Any] = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}") from e

    def _action_dict_to_action(self, action_dict: Dict[str, Any]) -> Action:
        """
        Convert parsed action dictionary to Action object.

        Args:
            action_dict: Parsed action dictionary from LLM

        Returns:
            Action object
        """
        action_type_str = action_dict.get("action_type", "NOOP").upper()
        params = action_dict.get("params", {})

        try:
            action_type = ActionType[action_type_str]
        except KeyError:
            logger.warning(
                f"Unknown action type: {action_type_str}, defaulting to NOOP"
            )
            return self.create_noop_action()

        if action_type == ActionType.NOOP:
            return self.create_noop_action()

        elif action_type == ActionType.POST:
            return self.create_post_action(content=params.get("content", ""))

        elif action_type == ActionType.REPLY:
            return self.create_reply_action(
                post_id=params.get("post_id", ""),
                content=params.get("content", ""),
            )

        elif action_type == ActionType.VOTE:
            return self.create_vote_action(
                post_id=params.get("post_id", ""),
                direction=int(params.get("direction", 1)),
            )

        elif action_type == ActionType.PROPOSE_INTERACTION:
            interaction_type_str = params.get("interaction_type", "collaboration")
            try:
                interaction_type = InteractionType(interaction_type_str)
            except ValueError:
                interaction_type = InteractionType.COLLABORATION

            return self.create_propose_action(
                counterparty_id=params.get("counterparty_id", ""),
                interaction_type=interaction_type,
                content=params.get("content", ""),
                task_id=params.get("task_id"),
            )

        elif action_type == ActionType.CLAIM_TASK:
            return self.create_claim_task_action(task_id=params.get("task_id", ""))

        elif action_type == ActionType.SUBMIT_OUTPUT:
            return self.create_submit_output_action(
                task_id=params.get("task_id", ""),
                content=params.get("content", ""),
            )

        elif action_type == ActionType.SUBMIT_KERNEL:
            return Action(
                action_type=ActionType.SUBMIT_KERNEL,
                agent_id=self.agent_id,
                target_id=params.get("challenge_id", ""),
                content=params.get("cuda_code", ""),
            )

        elif action_type == ActionType.VERIFY_KERNEL:
            return Action(
                action_type=ActionType.VERIFY_KERNEL,
                agent_id=self.agent_id,
                target_id=params.get("submission_id", ""),
            )

        elif action_type == ActionType.AUDIT_KERNEL:
            return Action(
                action_type=ActionType.AUDIT_KERNEL,
                agent_id=self.agent_id,
                target_id=params.get("submission_id", ""),
            )

        else:
            logger.warning(f"Unhandled action type: {action_type}, defaulting to NOOP")
            return self.create_noop_action()

    def act(self, observation: Observation) -> Action:
        """
        Decide on an action given the current observation.

        This is the synchronous interface required by BaseAgent.
        For async usage, use act_async() directly.

        Args:
            observation: Current view of the environment

        Returns:
            Action to take
        """
        system_prompt = ""
        user_prompt = ""
        try:
            # Use kernel-specific prompts when kernel challenges are available
            if observation.kernel_available_challenges:
                system_prompt, user_prompt = build_kernel_action_prompt(
                    persona=self.llm_config.persona,
                    observation=observation,
                    custom_system_prompt=self.llm_config.system_prompt,
                    memory=self.get_memory(limit=10),
                )
            else:
                system_prompt, user_prompt = build_action_prompt(
                    persona=self.llm_config.persona,
                    observation=observation,
                    custom_system_prompt=self.llm_config.system_prompt,
                    memory=self.get_memory(limit=10),
                )

            response, input_tokens, output_tokens = self._call_llm_sync(
                system_prompt, user_prompt
            )
            self._audit_llm_exchange(
                kind="act",
                observation=observation,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            action_dict = self._parse_action_response(response)

            # Store reasoning in memory
            if action_dict.get("reasoning"):
                self.remember(
                    {
                        "type": "action_reasoning",
                        "action_type": action_dict.get("action_type"),
                        "reasoning": action_dict["reasoning"],
                    }
                )

            return self._action_dict_to_action(action_dict)

        except Exception as e:
            self._audit_llm_exchange(
                kind="act_error",
                observation=observation,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=None,
                input_tokens=None,
                output_tokens=None,
                error=str(e),
            )
            logger.error(f"LLM action failed: {e}, returning NOOP")
            return self.create_noop_action()

    async def act_async(self, observation: Observation) -> Action:
        """
        Async version of act().

        Args:
            observation: Current view of the environment

        Returns:
            Action to take
        """
        system_prompt = ""
        user_prompt = ""
        try:
            # Use kernel-specific prompts when kernel challenges are available
            if observation.kernel_available_challenges:
                system_prompt, user_prompt = build_kernel_action_prompt(
                    persona=self.llm_config.persona,
                    observation=observation,
                    custom_system_prompt=self.llm_config.system_prompt,
                    memory=self.get_memory(limit=10),
                )
            else:
                system_prompt, user_prompt = build_action_prompt(
                    persona=self.llm_config.persona,
                    observation=observation,
                    custom_system_prompt=self.llm_config.system_prompt,
                    memory=self.get_memory(limit=10),
                )

            response, input_tokens, output_tokens = await self._call_llm_async(
                system_prompt, user_prompt
            )
            self._audit_llm_exchange(
                kind="act",
                observation=observation,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

            action_dict = self._parse_action_response(response)

            # Store reasoning in memory
            if action_dict.get("reasoning"):
                self.remember(
                    {
                        "type": "action_reasoning",
                        "action_type": action_dict.get("action_type"),
                        "reasoning": action_dict["reasoning"],
                    }
                )

            return self._action_dict_to_action(action_dict)

        except Exception as e:
            self._audit_llm_exchange(
                kind="act_error",
                observation=observation,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=None,
                input_tokens=None,
                output_tokens=None,
                error=str(e),
            )
            logger.error(f"LLM action failed: {e}, returning NOOP")
            return self.create_noop_action()

    def accept_interaction(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Decide whether to accept a proposed interaction.

        Args:
            proposal: The interaction proposal
            observation: Current observation

        Returns:
            True to accept, False to reject
        """
        system_prompt = ""
        user_prompt = ""
        try:
            proposal_dict = {
                "proposal_id": proposal.proposal_id,
                "initiator_id": proposal.initiator_id,
                "interaction_type": proposal.interaction_type.value,
                "content": proposal.content,
                "offered_transfer": proposal.offered_transfer,
            }

            system_prompt, user_prompt = build_accept_prompt(
                persona=self.llm_config.persona,
                proposal=proposal_dict,
                observation=observation,
                custom_system_prompt=self.llm_config.system_prompt,
                memory=self.get_memory(limit=10),
            )

            response, input_tokens, output_tokens = self._call_llm_sync(
                system_prompt, user_prompt
            )
            self._audit_llm_exchange(
                kind="accept",
                observation=observation,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                extra={
                    "proposal_id": proposal.proposal_id,
                    "initiator_id": proposal.initiator_id,
                },
            )

            decision_dict = self._parse_action_response(response)

            # Store reasoning
            if decision_dict.get("reasoning"):
                self.remember(
                    {
                        "type": "accept_decision",
                        "proposal_id": proposal.proposal_id,
                        "initiator_id": proposal.initiator_id,
                        "accepted": decision_dict.get("accept", False),
                        "reasoning": decision_dict["reasoning"],
                    }
                )

            return bool(decision_dict.get("accept", False))

        except Exception as e:
            self._audit_llm_exchange(
                kind="accept_error",
                observation=observation,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=None,
                input_tokens=None,
                output_tokens=None,
                error=str(e),
                extra={
                    "proposal_id": proposal.proposal_id,
                    "initiator_id": proposal.initiator_id,
                },
            )
            logger.error(f"LLM accept_interaction failed: {e}, defaulting to False")
            return False

    async def accept_interaction_async(
        self,
        proposal: InteractionProposal,
        observation: Observation,
    ) -> bool:
        """
        Async version of accept_interaction().

        Args:
            proposal: The interaction proposal
            observation: Current observation

        Returns:
            True to accept, False to reject
        """
        system_prompt = ""
        user_prompt = ""
        try:
            proposal_dict = {
                "proposal_id": proposal.proposal_id,
                "initiator_id": proposal.initiator_id,
                "interaction_type": proposal.interaction_type.value,
                "content": proposal.content,
                "offered_transfer": proposal.offered_transfer,
            }

            system_prompt, user_prompt = build_accept_prompt(
                persona=self.llm_config.persona,
                proposal=proposal_dict,
                observation=observation,
                custom_system_prompt=self.llm_config.system_prompt,
                memory=self.get_memory(limit=10),
            )

            response, input_tokens, output_tokens = await self._call_llm_async(
                system_prompt, user_prompt
            )
            self._audit_llm_exchange(
                kind="accept",
                observation=observation,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=response,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                extra={
                    "proposal_id": proposal.proposal_id,
                    "initiator_id": proposal.initiator_id,
                },
            )

            decision_dict = self._parse_action_response(response)

            # Store reasoning
            if decision_dict.get("reasoning"):
                self.remember(
                    {
                        "type": "accept_decision",
                        "proposal_id": proposal.proposal_id,
                        "initiator_id": proposal.initiator_id,
                        "accepted": decision_dict.get("accept", False),
                        "reasoning": decision_dict["reasoning"],
                    }
                )

            return bool(decision_dict.get("accept", False))

        except Exception as e:
            self._audit_llm_exchange(
                kind="accept_error",
                observation=observation,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_text=None,
                input_tokens=None,
                output_tokens=None,
                error=str(e),
                extra={
                    "proposal_id": proposal.proposal_id,
                    "initiator_id": proposal.initiator_id,
                },
            )
            logger.error(f"LLM accept_interaction failed: {e}, defaulting to False")
            return False

    def propose_interaction(
        self,
        observation: Observation,
        counterparty_id: str,
    ) -> Optional[InteractionProposal]:
        """
        Create an interaction proposal for a counterparty.

        LLM agents propose interactions only via act() returning a
        PROPOSE_INTERACTION action; this method exists for interface
        compatibility and always returns None.

        Args:
            observation: Current observation
            counterparty_id: Target agent ID

        Returns:
            InteractionProposal or None if not proposing
        """
        # LLM agents propose interactions through the act() method
        # This is here for interface compatibility
        return None

    def apply_memory_decay(self, epoch: int) -> None:
        """Apply memory decay, including Memori session rotation."""
        super().apply_memory_decay(epoch)

        middleware = self._get_memori_middleware()
        if middleware is not None:
            middleware.on_epoch_boundary(
                epoch, self.memory_config.epistemic_persistence
            )

    def get_usage_stats(self) -> Dict[str, Any]:
        """Get LLM usage statistics."""
        result: Dict[str, Any] = self.usage_stats.to_dict()
        return result

    def __repr__(self) -> str:
        return (
            f"LLMAgent(id={self.agent_id}, "
            f"provider={self.llm_config.provider.value}, "
            f"model={self.llm_config.model}, "
            f"persona={self.llm_config.persona.value})"
        )
