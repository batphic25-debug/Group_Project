from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Sequence

from src.core.config import AppSettings, get_settings
from src.core.schemas import LegalChunk, RetrievedChunk
from src.providers.base import BaseEmbedder, BaseReranker
from src.providers.factory import create_embedder, create_reranker
from src.providers.rerankers.cloud_rerankers import LexicalFallbackReranker
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.lexical_search import BM25LexicalEngine
from src.retrieval.seed_data import ensure_seed_data
from src.retrieval.vector_store import LocalDiskVectorStore


class HybridLegalRetriever:
    """
    Động cơ Truy xuất Pháp lý Kết hợp (Enhanced RAG Retriever):
    1. Lọc Metadata (nếu có) + Tìm kiếm song song BM25 & Vector Search.
    2. Hợp nhất ứng viên bằng Reciprocal Rank Fusion (RRF).
    3. Chấm điểm lại bằng Cloud Reranker (Voyage Rerank 3 / Cohere Rerank 5).
    4. Lưu Cache kết quả truy vấn xuống đĩa để chạy Benchmark 50 câu tức thì.
    """

    def __init__(
        self,
        chunks: Optional[Sequence[LegalChunk]] = None,
        embedder: Optional[BaseEmbedder] = None,
        reranker: Optional[BaseReranker] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.chunks: List[LegalChunk] = list(chunks) if chunks is not None else ensure_seed_data()
        self.embedder: BaseEmbedder = embedder or create_embedder(self.settings, allow_mock_fallback=True)
        self.reranker: BaseReranker = reranker or create_reranker(self.settings, allow_fallback=True)

        self.bm25_engine = BM25LexicalEngine(self.chunks)
        self.vector_store = LocalDiskVectorStore(
            chunks=self.chunks,
            embedder=self.embedder,
            index_dir=self.settings.paths.vector_index_dir,
        )
        self.cache_dir: Path = self.settings.paths.cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file_for(self, query: str, top_n: int, law_filter: Optional[str]) -> Path:
        raw_key = f"{self.embedder.model_name}|{self.reranker.model_name}|{top_n}|{law_filter or ''}|{query.strip().lower()}"
        digest = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"ret_{digest}.json"

    def retrieve(
        self,
        query: str,
        top_n: Optional[int] = None,
        law_filter: Optional[str] = None,
        use_cache: Optional[bool] = None,
    ) -> List[RetrievedChunk]:
        """Thực hiện toàn bộ chu trình Hybrid Search + Reranking cho một câu truy vấn."""
        final_top_n = top_n or self.settings.models.reranker.top_n
        should_cache = self.settings.retrieval.enable_cache if use_cache is None else use_cache

        cache_path = self._cache_file_for(query, final_top_n, law_filter)
        if should_cache and cache_path.exists():
            try:
                raw_cached = json.loads(cache_path.read_text(encoding="utf-8"))
                return [RetrievedChunk.model_validate(item) for item in raw_cached]
            except Exception:
                pass

        # Bước 1: Tìm kiếm song song BM25 và Vector
        bm25_hits = self.bm25_engine.search(
            query=query,
            top_k=self.settings.retrieval.bm25_top_k,
            law_filter=law_filter,
        )
        vec_hits = self.vector_store.search(
            query=query,
            top_k=self.settings.retrieval.vector_top_k,
            law_filter=law_filter,
        )

        # Bước 2: Gộp bằng RRF
        fused_candidates = reciprocal_rank_fusion(
            bm25_results=bm25_hits,
            vector_results=vec_hits,
            rrf_k=self.settings.retrieval.rrf_k,
            top_candidates=self.settings.retrieval.hybrid_candidates,
        )

        if not fused_candidates:
            return []

        # Bước 3: Chấm điểm lại bằng Reranker (có tự động fallback nếu lỗi API)
        try:
            reranked = self.reranker.rerank(
                query=query,
                candidates=fused_candidates,
                top_n=final_top_n,
            )
        except Exception:
            fallback = LexicalFallbackReranker(model_name="fallback-lexical")
            reranked = fallback.rerank(query=query, candidates=fused_candidates, top_n=final_top_n)

        if should_cache and reranked:
            cache_path.write_text(
                json.dumps([item.model_dump() for item in reranked], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return reranked