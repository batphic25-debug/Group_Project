from __future__ import annotations

from typing import List, Sequence

from src.core.exceptions import ProviderAPIError
from src.core.key_rotator import get_key_rotator
from src.core.schemas import RetrievedChunk
from src.providers.base import BaseReranker


class VoyageReranker(BaseReranker):
    """Kết nối API Voyage Rerank 3 (cửa sổ 32K tokens, 200M tokens miễn phí)."""

    def __init__(self, model_name: str = "rerank-3") -> None:
        super().__init__(model_name=model_name)
        self.rotator = get_key_rotator("voyage")

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int = 5,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        import voyageai

        docs = [c.chunk.contextual_text for c in candidates]
        last_err: Exception | None = None

        for _ in range(max(1, self.rotator.total_keys)):
            api_key = self.rotator.get_next_key()
            try:
                client = voyageai.Client(api_key=api_key)
                reranked = client.rerank(
                    query=query,
                    documents=docs,
                    model=self.model_name,
                    top_k=min(top_n, len(candidates)),
                )
                results: List[RetrievedChunk] = []
                for rank_idx, item in enumerate(reranked.results, start=1):
                    orig = candidates[item.index].model_copy(deep=True)
                    orig.scores.rerank_score = float(item.relevance_score)
                    orig.rank = rank_idx
                    results.append(orig)
                return results
            except Exception as exc:
                last_err = exc
                if "429" in str(exc) or "rate" in str(exc).lower():
                    self.rotator.mark_rate_limited(api_key, cooldown_seconds=60.0)
                    continue
                break

        raise ProviderAPIError("voyage-rerank", str(last_err))


class CohereReranker(BaseReranker):
    """Kết nối API Cohere Rerank (rerank-v5.0-pro / rerank-v3.5)."""

    def __init__(self, model_name: str = "rerank-v5.0-pro") -> None:
        super().__init__(model_name=model_name)
        self.rotator = get_key_rotator("cohere")

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int = 5,
    ) -> List[RetrievedChunk]:
        if not candidates:
            return []

        import cohere

        docs = [c.chunk.contextual_text for c in candidates]
        last_err: Exception | None = None

        for _ in range(max(1, self.rotator.total_keys)):
            api_key = self.rotator.get_next_key()
            try:
                client = cohere.ClientV2(api_key=api_key)
                response = client.rerank(
                    model=self.model_name,
                    query=query,
                    documents=docs,
                    top_n=min(top_n, len(candidates)),
                )
                results: List[RetrievedChunk] = []
                for rank_idx, item in enumerate(response.results, start=1):
                    orig = candidates[item.index].model_copy(deep=True)
                    orig.scores.rerank_score = float(item.relevance_score)
                    orig.rank = rank_idx
                    results.append(orig)
                return results
            except Exception as exc:
                last_err = exc
                if "429" in str(exc) or "rate" in str(exc).lower():
                    self.rotator.mark_rate_limited(api_key, cooldown_seconds=60.0)
                    continue
                break

        raise ProviderAPIError("cohere-rerank", str(last_err))


class LexicalFallbackReranker(BaseReranker):
    """
    Reranker dự phòng an toàn (chấm điểm dựa trên độ phủ từ khóa + điểm RRF).
    Giúp hệ thống vẫn chạy mượt khi test offline hoặc chưa nhập key Reranker.
    """

    def rerank(
        self,
        query: str,
        candidates: Sequence[RetrievedChunk],
        top_n: int = 5,
    ) -> List[RetrievedChunk]:
        query_terms = set(query.lower().split())
        scored: List[RetrievedChunk] = []

        for item in candidates:
            copy_item = item.model_copy(deep=True)
            doc_terms = set(copy_item.chunk.contextual_text.lower().split())
            overlap = len(query_terms & doc_terms) / max(len(query_terms), 1)
            # Kết hợp điểm RRF sẵn có và độ trùng khớp từ khóa
            copy_item.scores.rerank_score = round(min(0.99, overlap * 0.7 + copy_item.scores.rrf_score * 5.0), 4)
            scored.append(copy_item)

        scored.sort(key=lambda x: x.scores.rerank_score, reverse=True)
        for idx, elem in enumerate(scored[:top_n], start=1):
            elem.rank = idx
        return scored[:top_n]