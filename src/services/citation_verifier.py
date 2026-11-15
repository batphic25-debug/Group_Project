from __future__ import annotations

import re
from typing import List, Sequence, Set, Tuple

from src.core.schemas import RetrievedChunk


class CitationVerifier:
    """
    Chốt chặn kiểm tra tính xác thực của trích dẫn pháp lý (Deterministic Guardrail).
    Đối chiếu mọi số hiệu 'Điều X' xuất hiện trong câu trả lời của LLM với Metadata thật.
    """

    ARTICLE_PATTERN = re.compile(r"điều\s+(\d+[a-zA-Z]?)", re.IGNORECASE)

    @classmethod
    def extract_cited_articles(cls, text: str) -> List[str]:
        """Trích xuất danh sách các số hiệu Điều được nhắc tới trong văn bản (ví dụ: ['Điều 23', 'Điều 25'])."""
        matches = cls.ARTICLE_PATTERN.findall(text)
        unique_articles: List[str] = []
        for num in matches:
            label = f"Điều {num}"
            if label not in unique_articles:
                unique_articles.append(label)
        return unique_articles

    @classmethod
    def verify_answer(
        cls,
        answer_text: str,
        retrieved_chunks: Sequence[RetrievedChunk],
    ) -> Tuple[bool, List[str]]:
        """
        Kiểm tra xem tất cả các Điều luật được LLM trích dẫn có nằm trong retrieved_chunks hay không.
        Trả về: (is_verified: bool, unverified_citations: List[str])
        """
        cited_articles = cls.extract_cited_articles(answer_text)
        if not cited_articles:
            return True, []

        allowed_articles: Set[str] = {
            c.chunk.article_number.strip().lower() for c in retrieved_chunks
        }

        unverified: List[str] = []
        for cited in cited_articles:
            if cited.lower() not in allowed_articles:
                unverified.append(cited)

        return (len(unverified) == 0), unverified