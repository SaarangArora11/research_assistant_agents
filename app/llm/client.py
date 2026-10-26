"""Unified OpenAI-compatible asynchronous LLM client for llama.cpp and OpenRouter."""

from __future__ import annotations
import asyncio
import json
import re
from typing import Any, Dict, List, Optional, Type, TypeVar
from openai import AsyncOpenAI
from pydantic import BaseModel
from app.core.config import Settings, load_settings

T = TypeVar("T", bound=BaseModel)


class LLMResponse(BaseModel):
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    parsed: Optional[Any] = None


class LLMClient:
    """Async client that connects to either local llama-server or OpenRouter."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or load_settings()
        self._init_client()

    def _init_client(self) -> None:
        if self.settings.provider == "openrouter":
            base_url = self.settings.openrouter_base_url.rstrip("/")
            api_key = self.settings.openrouter_api_key or "placeholder_key"
            default_headers = {
                "HTTP-Referer": "https://github.com/research-assistant-agents",
                "X-Title": "Research Assistant Agents",
            }
        else:
            base_url = self.settings.llamacpp_base_url.rstrip("/")
            api_key = self.settings.llamacpp_api_key or "no-key-required"
            default_headers = {}

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            default_headers=default_headers,
            timeout=120.0,
            max_retries=2,
        )

    def get_model_for_role(self, role: str) -> str:
        """Resolve model name for an agent role."""
        if self.settings.provider == "llamacpp":
            # llama.cpp server usually serves one model, alias is fine
            return self.settings.llamacpp_model_alias or "local-model"

        # Check role-specific override
        override = self.settings.role_models.get(role)
        if override and override.strip():
            return override.strip()

        return self.settings.openrouter_default_model

    async def test_connection(self) -> Dict[str, Any]:
        """Test API connectivity and retrieve available models."""
        try:
            models_response = await self.client.models.list()
            model_ids = [m.id for m in models_response.data] if hasattr(models_response, "data") else []
            return {
                "success": True,
                "provider": self.settings.provider,
                "base_url": str(self.client.base_url),
                "model_count": len(model_ids),
                "models": model_ids[:30],
            }
        except Exception as e:
            return {
                "success": False,
                "provider": self.settings.provider,
                "base_url": str(self.client.base_url),
                "error": str(e),
            }

    async def list_available_models(self) -> List[str]:
        """List models supported by current provider."""
        try:
            resp = await self.client.models.list()
            if hasattr(resp, "data"):
                return [m.id for m in resp.data]
            return []
        except Exception:
            return []

    async def chat(
        self,
        messages: List[Dict[str, str]],
        role: str = "researcher",
        model_override: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        response_schema: Optional[Type[T]] = None,
        max_retries: int = 3,
    ) -> LLMResponse:
        """Execute chat completion with retry, token tracking, and structured output parsing."""
        model = model_override or self.get_model_for_role(role)
        temp = temperature if temperature is not None else self.settings.temperature

        # If schema is requested, enforce JSON
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
        }

        if response_schema is not None:
            # Add instruction if not already present
            kwargs["response_format"] = {"type": "json_object"}

        last_error: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = await self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content or ""

                prompt_tokens = response.usage.prompt_tokens if response.usage else 0
                completion_tokens = response.usage.completion_tokens if response.usage else 0
                total_tokens = response.usage.total_tokens if response.usage else 0

                llm_resp = LLMResponse(
                    content=content,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    model=model,
                )

                if response_schema is not None:
                    parsed_obj = self._extract_and_validate_json(content, response_schema)
                    if parsed_obj is not None:
                        llm_resp.parsed = parsed_obj
                        return llm_resp
                    else:
                        # Append feedback message to retry with correction
                        if attempt < max_retries - 1:
                            messages.append({"role": "assistant", "content": content})
                            messages.append({
                                "role": "user",
                                "content": (
                                    f"Your response could not be parsed according to the required JSON schema for {response_schema.__name__}. "
                                    f"Please return ONLY a valid, raw JSON object matching the schema without markdown tags or extraneous commentary."
                                ),
                            })
                            kwargs["messages"] = messages
                            await asyncio.sleep(1.0)
                            continue
                else:
                    return llm_resp

            except Exception as e:
                last_error = e
                # Exponential backoff for network / rate-limit errors
                wait_time = (2 ** attempt) * 1.5
                await asyncio.sleep(wait_time)

        if response_schema is not None and last_error is None:
            raise ValueError(f"Failed to obtain valid JSON schema response for {response_schema.__name__} after {max_retries} attempts.")
        raise last_error or RuntimeError("LLM request failed")

    @staticmethod
    def _extract_and_validate_json(text: str, schema: Type[T]) -> Optional[T]:
        """Extract JSON block from text and parse into schema."""
        # Clean markdown code blocks if wrapped
        cleaned = text.strip()
        json_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(json_pattern, cleaned)
        if match:
            cleaned = match.group(1).strip()

        # Try to find the outermost { ... }
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            cleaned = cleaned[start : end + 1]

        try:
            data = json.loads(cleaned)
            return schema.model_validate(data)
        except Exception:
            return None
