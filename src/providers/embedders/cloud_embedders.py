from __future__ import annotations

import hashlib
from typing import List, Sequence

import numpy as np

from src.core.exceptions import ProviderAPIError
from src.core.key_rotator import get_key_rotator
from src.providers.base import BaseEmbedder


class VoyageEmbedder(BaseEmbedder):
    """Kết nối API Voyage AI (voyage-context-4 / voyage-3-large) kèm xoay vòng Key."""

    def __init__(self, model_name: str = "voyage-context-4", dimensions: int = 1024, batch_size: int = 32) -> None:
        super().__init__(model_name=model_name, dimensions=dimensions)
        self.batch_size = batch_size
        self.rotator = get_key_rotator("voyage")

    def _call_voyage(self, texts: List[str], input_type: str) -> List[List[float]]:
        import voyageai

        last_err: Exception | None = None
        # Thử tối đa qua các key hiện có nếu gặp lỗi Rate Limit
        for _ in range(max(1, self.rotator.total_keys)):
            api_key = self.rotator.get_next_key()
            try:
                client = voyageai.Client(api_key=api_key)
                # Hỗ trợ truyền output_dimension cho các dòng model hỗ trợ Matryoshka
                result = client.embed(
                    texts=texts,
                    model=self.model_name,
                    input_type=input_type,
                )
                return [list(vec) for vec in result.embeddings]
            except Exception as exc:
                last_err = exc
                if "429" in str(exc) or "rate" in str(exc).lower():
                    self.rotator.mark_rate_limited(api_key, cooldown_seconds=60.0)
                    continue
                break
        raise ProviderAPIError("voyage", str(last_err))

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        all_embeddings: List[List[float]] = []
        text_list = list(texts)
        for i in range(0, len(text_list), self.batch_size):
            batch = text_list[i : i + self.batch_size]
            all_embeddings.extend(self._call_voyage(batch, input_type="document"))
        return all_embeddings

    def embed_query(self, query: str) -> List[float]:
        return self._call_voyage([query], input_type="query")[0]


class GeminiEmbedder(BaseEmbedder):
    """Kết nối Google Gemini Embedding (gemini-embedding-2-preview) miễn phí qua AI Studio."""

    def __init__(self, model_name: str = "gemini-embedding-2-preview", dimensions: int = 768) -> None:
        super().__init__(model_name=model_name, dimensions=dimensions)
        self.rotator = get_key_rotator("gemini")

    def _embed_batch(self, texts: List[str], task_type: str) -> List[List[float]]:
        from google import genai
        from google.genai import types

        last_err: Exception | None = None
        for _ in range(max(1, self.rotator.total_keys)):
            api_key = self.rotator.get_next_key()
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.embed_content(
                    model=self.model_name,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.dimensions,
                    ),
                )
                return [list(item.values) for item in response.embeddings]
            except Exception as exc:
                last_err = exc
                if "429" in str(exc) or "quota" in str(exc).lower():
                    self.rotator.mark_rate_limited(api_key, cooldown_seconds=60.0)
                    continue
                break
        raise ProviderAPIError("gemini-embedding", str(last_err))

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return self._embed_batch(list(texts), task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> List[float]:
        return self._embed_batch([query], task_type="RETRIEVAL_QUERY")[0]


class DeterministicMockEmbedder(BaseEmbedder):
    """
    Bộ sinh Vector giả lập xác định (dùng để kiểm thử luồng Pipeline khi chưa điền API Key thật).
    Đảm bảo văn bản giống nhau luôn sinh ra vector giống hệt nhau.
    """

    def _text_to_vector(self, text: str) -> List[float]:
        seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.dimensions)
        norm = np.linalg.norm(vec)
        return (vec / norm if norm > 0 else vec).tolist()

    def embed_documents(self, texts: Sequence[str]) -> List[List[float]]:
        return [self._text_to_vector(t) for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self._text_to_vector(query)