from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.core.schemas import LegalChunk
from src.providers.base import BaseEmbedder


class LocalDiskVectorStore:
    """
    Kho Vector gọn nhẹ dựa trên Numpy Cosine Similarity.
    Tự động lưu/nạp chỉ mục từ 'data/vector_index/' để tránh gọi lại API Embedding khi khởi động lại.
    """

    def __init__(
        self,
        chunks: Sequence[LegalChunk],
        embedder: BaseEmbedder,
        index_dir: Path,
    ) -> None:
        self.chunks: List[LegalChunk] = list(chunks)
        self.embedder = embedder
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._matrix: Optional[np.ndarray] = None
        self._build_or_load_index()

    def _compute_signature(self) -> str:
        """Tạo chữ ký băm dựa trên danh sách chunk_id và tên model embedding."""
        payload = f"{self.embedder.model_name}:{self.embedder.dimensions}:" + "|".join(
            f"{c.chunk_id}:{len(c.content)}" for c in self.chunks
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

    def _build_or_load_index(self) -> None:
        if not self.chunks:
            self._matrix = np.zeros((0, self.embedder.dimensions), dtype=np.float32)
            return

        sig = self._compute_signature()
        vec_file = self.index_dir / f"vectors_{sig}.npy"

        if vec_file.exists():
            self._matrix = np.load(vec_file)
            return

        texts = [c.contextual_text for c in self.chunks]
        embeddings = self.embedder.embed_documents(texts)
        mat = np.array(embeddings, dtype=np.float32)

        # Chuẩn hóa L2 từng hàng để tích vô hướng trở thành Cosine Similarity
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = mat / norms
        np.save(vec_file, self._matrix)

    def search(
        self,
        query: str,
        top_k: int = 20,
        law_filter: Optional[str] = None,
    ) -> List[Tuple[LegalChunk, float]]:
        """Tìm kiếm tương đồng ngữ nghĩa Cosine kết hợp lọc Metadata."""
        if self._matrix is None or len(self.chunks) == 0:
            return []

        q_vec = np.array(self.embedder.embed_query(query), dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        similarities = np.dot(self._matrix, q_vec)
        results: List[Tuple[LegalChunk, float]] = []

        for idx, sim in enumerate(similarities):
            chunk = self.chunks[idx]
            if law_filter and law_filter.lower() not in chunk.law_name.lower():
                continue
            results.append((chunk, float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]