from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Sequence, Type, TypeVar

from pydantic import BaseModel

from src.core.schemas import LegalChunk, RetrievedChunk

T = TypeVar("T", bound=BaseModel)


class BaseEmbedder(ABC):
    """Giao diện chuẩn cho tất cả các mô hình Embedding (Voyage, Cohere, Gemini)."""

    def __init__(self, model_name: str, dimensions: int = 1024) -> None:
        self.model_name = model_name
        self.dimensions = dimensions

    @abstractmethod
    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        """Vector hóa danh sách các đoạn văn bản Luật / Hợp đồng."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Vector hóa câu hỏi tìm kiếm của người dùng."""
        raise NotImplementedError


class BaseReranker(ABC):
    """Giao diện chuẩn cho các mô hình Reranker (Voyage Rerank 3, Cohere Rerank 5)."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int = 5,
    ) -> List[RetrievedChunk]:
        """Chấm điểm lại danh sách ứng viên và sắp xếp theo độ liên quan giảm dần."""
        raise NotImplementedError


class BaseLLM(ABC):
    """Giao diện chuẩn cho cả Cloud API LLM (TV5) và Open LLM (TV4)."""

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    @abstractmethod
    def generate_text(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> str:
        """Sinh câu trả lời dạng văn bản (phục vụ Hỏi đáp Pháp lý & Benchmark 50 câu)."""
        raise NotImplementedError

    @abstractmethod
    def generate_structured(
        self,
        prompt: str,
        response_schema: Type[T],
        system_prompt: str | None = None,
    ) -> T:
        """
        Sinh kết quả ép khuôn chặt chẽ theo Pydantic Schema (JSON).
        Phục vụ bóc tách và thẩm định Biên bản/Hợp đồng BĐS tự động.
        """
        raise NotImplementedError