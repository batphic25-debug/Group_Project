from __future__ import annotations

from src.core.config import AppSettings, LLMModelConfig, get_settings
from src.core.key_rotator import get_key_rotator
from src.providers.base import BaseEmbedder, BaseLLM, BaseReranker
from src.providers.embedders.cloud_embedders import (
    DeterministicMockEmbedder,
    GeminiEmbedder,
    VoyageEmbedder,
)
from src.providers.llms.cloud_llms import GeminiLLM, OpenAICompatibleLLM
from src.providers.rerankers.cloud_rerankers import (
    CohereReranker,
    LexicalFallbackReranker,
    VoyageReranker,
)


def _has_real_key(provider: str) -> bool:
    """Kiểm tra xem nhà cung cấp đã được cấu hình API Key thật (không bắt đầu bằng 'test_') hay chưa."""
    try:
        rotator = get_key_rotator(provider)
        if rotator.total_keys == 0:
            return False
        sample = rotator.get_next_key()
        return not sample.startswith("test_")
    except Exception:
        return False


def create_embedder(settings: AppSettings | None = None, allow_mock_fallback: bool = True) -> BaseEmbedder:
    """Khởi tạo Embedder theo cấu hình trong settings.yaml."""
    cfg = (settings or get_settings()).models.embedding

    if allow_mock_fallback and not _has_real_key(cfg.provider):
        return DeterministicMockEmbedder(model_name=f"mock-{cfg.model_name}", dimensions=cfg.dimensions)

    if cfg.provider == "voyage":
        return VoyageEmbedder(model_name=cfg.model_name, dimensions=cfg.dimensions, batch_size=cfg.batch_size)
    if cfg.provider == "gemini":
        return GeminiEmbedder(model_name=cfg.model_name, dimensions=cfg.dimensions)

    return DeterministicMockEmbedder(model_name=cfg.model_name, dimensions=cfg.dimensions)


def create_reranker(settings: AppSettings | None = None, allow_fallback: bool = True) -> BaseReranker:
    """Khởi tạo Reranker theo cấu hình trong settings.yaml."""
    cfg = (settings or get_settings()).models.reranker

    if allow_fallback and not _has_real_key(cfg.provider):
        return LexicalFallbackReranker(model_name=f"fallback-{cfg.model_name}")

    if cfg.provider == "voyage":
        return VoyageReranker(model_name=cfg.model_name)
    if cfg.provider == "cohere":
        return CohereReranker(model_name=cfg.model_name)

    return LexicalFallbackReranker(model_name=cfg.model_name)


def create_llm(
    role: str = "primary",
    custom_config: LLMModelConfig | None = None,
    settings: AppSettings | None = None,
) -> BaseLLM:
    """
    Khởi tạo LLM theo vai trò:
    - role='primary'   -> Lấy llm_primary (ví dụ: gemini-4-hydrogen)
    - role='secondary' -> Lấy llm_secondary (ví dụ: qwen/qwen3.8-27b trên Groq)
    Hoặc truyền trực tiếp custom_config để chạy Benchmark nhiều model cùng lúc.
    """
    app_cfg = settings or get_settings()
    llm_cfg = custom_config or (
        app_cfg.models.llm_primary if role == "primary" else app_cfg.models.llm_secondary
    )

    if llm_cfg.provider == "gemini":
        return GeminiLLM(
            model_name=llm_cfg.model_name,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
        )

    return OpenAICompatibleLLM(
        provider=llm_cfg.provider,
        model_name=llm_cfg.model_name,
        temperature=llm_cfg.temperature,
        max_tokens=llm_cfg.max_tokens,
    )