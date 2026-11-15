from __future__ import annotations

import json
from typing import Type, TypeVar

from pydantic import BaseModel

from src.core.exceptions import ProviderAPIError
from src.core.key_rotator import get_key_rotator
from src.providers.base import BaseLLM

T = TypeVar("T", bound=BaseModel)


class GeminiLLM(BaseLLM):
    """
    Kết nối trực tiếp dòng mô hình mới nhất của Google (gemini-4-hydrogen / gemini-3.8-flash).
    Hỗ trợ xoay vòng API Key tự động và ép khuôn JSON Schema (Structured Outputs) chuẩn 100%.
    """

    def __init__(
        self,
        model_name: str = "gemini-4-hydrogen",
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.rotator = get_key_rotator("gemini")

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        from google import genai
        from google.genai import types

        last_err: Exception | None = None
        for _ in range(max(1, self.rotator.total_keys)):
            api_key = self.rotator.get_next_key()
            try:
                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_prompt,
                )
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                return response.text or ""
            except Exception as exc:
                last_err = exc
                if "429" in str(exc) or "quota" in str(exc).lower() or "resource_exhausted" in str(exc).lower():
                    self.rotator.mark_rate_limited(api_key, cooldown_seconds=60.0)
                    continue
                break

        raise ProviderAPIError("gemini", str(last_err))

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_prompt: str | None = None,
    ) -> T:
        from google import genai
        from google.genai import types

        last_err: Exception | None = None
        for _ in range(max(1, self.rotator.total_keys)):
            api_key = self.rotator.get_next_key()
            try:
                client = genai.Client(api_key=api_key)
                config = types.GenerateContentConfig(
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                    response_schema=response_schema,
                )
                response = client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                raw_json = response.text or "{}"
                return response_schema.model_validate_json(raw_json)
            except Exception as exc:
                last_err = exc
                if "429" in str(exc) or "quota" in str(exc).lower() or "resource_exhausted" in str(exc).lower():
                    self.rotator.mark_rate_limited(api_key, cooldown_seconds=60.0)
                    continue
                break

        raise ProviderAPIError("gemini-structured", str(last_err))


class OpenAICompatibleLLM(BaseLLM):
    """
    Cổng kết nối đa năng dùng chung cho:
    - GroqCloud API (qwen/qwen3.8-27b, gpt-oss-120b) -> của TV5 & TV7
    - OpenRouter API -> của TV5
    - Ollama Local API (gemma4:12b, gemma-4-26B-A4B) -> của TV4
    """

    BASE_URLS = {
        "groq": "https://api.groq.com/openai/v1",
        "openrouter": "https://openrouter.ai/api/v1",
        "ollama": "http://localhost:11434/v1",
    }

    def __init__(
        self,
        provider: str,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens)
        self.provider = provider.lower().strip()
        self.base_url = self.BASE_URLS.get(self.provider, self.BASE_URLS["groq"])
        self.rotator = get_key_rotator(self.provider) if self.provider != "ollama" else None

    def _get_api_key(self) -> str:
        if self.provider == "ollama" or self.rotator is None:
            return "ollama-local"
        return self.rotator.get_next_key()

    def generate_text(self, prompt: str, system_prompt: str | None = None) -> str:
        from openai import OpenAI

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        attempts = 1 if self.rotator is None else max(1, self.rotator.total_keys)
        last_err: Exception | None = None

        for _ in range(attempts):
            api_key = self._get_api_key()
            try:
                client = OpenAI(api_key=api_key, base_url=self.base_url)
                completion = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return completion.choices[0].message.content or ""
            except Exception as exc:
                last_err = exc
                if self.rotator and ("429" in str(exc) or "rate" in str(exc).lower()):
                    self.rotator.mark_rate_limited(api_key, cooldown_seconds=60.0)
                    continue
                break

        raise ProviderAPIError(self.provider, str(last_err))

    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_prompt: str | None = None,
    ) -> T:
        schema_json = json.dumps(response_schema.model_json_schema(), ensure_ascii=False)
        augmented_system = (
            f"{system_prompt or ''}\n\n"
            f"BẮT BUỘC TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON KHỚP VỚI SCHEMA SAU (Không thêm markdown ```json):\n{schema_json}"
        ).strip()

        raw_text = self.generate_text(prompt=prompt, system_prompt=augmented_system)
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`").removeprefix("json").strip()
        return response_schema.model_validate_json(cleaned)