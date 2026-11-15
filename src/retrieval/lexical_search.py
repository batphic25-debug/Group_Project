from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

from rank_bm25 import BM25Okapi

from src.core.schemas import LegalChunk

VIETNAMESE_STOPWORDS = {
    "là", "và", "của", "có", "được", "trong", "cho", "các", "với", "tại",
    "theo", "này", "đó", "những", "một", "về", "khi", "thì", "mà", "để",
}


def tokenize_vietnamese_legal(text: str) -> List[str]:
    """
    Chuẩn hóa văn bản pháp lý tiếng Việt thành danh sách token cho BM25.
    Giữ lại các con số phần trăm (05%, 30%, 70%, 95%) và số hiệu Điều/Khoản.
    """
    lowered = text.lower()
    # Chuẩn hóa ký hiệu phần trăm đi liền số (ví dụ: '05%' -> '05_phan_tram' và '5%')
    lowered = re.sub(r"0?(\d+)\s*%", r"\1% \1_phan_tram", lowered)
    tokens = re.findall(r"[a-zA-Z0-9_àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ%]+", lowered)
    return [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) > 1]


class BM25LexicalEngine:
    """Động cơ tìm kiếm từ khóa BM25 hỗ trợ lọc trước theo Metadata (Pre-filtering)."""

    def __init__(self, chunks: Sequence[LegalChunk]) -> None:
        self.chunks: List[LegalChunk] = list(chunks)
        self._corpus_tokens: List[List[str]] = [
            tokenize_vietnamese_legal(c.contextual_text) for c in self.chunks
        ]
        self._bm25: Optional[BM25Okapi] = (
            BM25Okapi(self._corpus_tokens) if self._corpus_tokens else None
        )

    def search(
        self,
        query: str,
        top_k: int = 20,
        law_filter: Optional[str] = None,
    ) -> List[Tuple[LegalChunk, float]]:
        """Trả về danh sách (LegalChunk, bm25_score) giảm dần."""
        if not self.chunks or self._bm25 is None:
            return []

        query_tokens = tokenize_vietnamese_legal(query)
        if not query_tokens:
            return []

        raw_scores = self._bm25.get_scores(query_tokens)
        scored_items: List[Tuple[LegalChunk, float]] = []

        for idx, score in enumerate(raw_scores):
            chunk = self.chunks[idx]
            if law_filter and law_filter.lower() not in chunk.law_name.lower():
                continue
            scored_items.append((chunk, float(score)))

        scored_items.sort(key=lambda x: x[1], reverse=True)
        return scored_items[:top_k]