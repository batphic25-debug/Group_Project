from __future__ import annotations

import time
from typing import Optional

from src.core.config import AppSettings, get_settings
from src.core.schemas import QAResponse, RetrievedChunk
from src.providers.base import BaseLLM
from src.providers.factory import _has_real_key, create_llm
from src.retrieval.retriever_pipeline import HybridLegalRetriever
from src.services.citation_verifier import CitationVerifier
from src.services.prompts.legal_prompts import (
    LEGAL_QA_SYSTEM_PROMPT,
    format_retrieved_context,
)


class LegalQAService:
    """
    Dịch vụ Hỏi đáp Pháp lý Bất động sản (SURE-RAG Pipeline).
    Phục vụ Tab Tra cứu trên Giao diện và chạy 50 câu Benchmark của TV6 & TV7.
    """

    def __init__(
        self,
        retriever: Optional[HybridLegalRetriever] = None,
        llm: Optional[BaseLLM] = None,
        settings: Optional[AppSettings] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.retriever = retriever or HybridLegalRetriever(settings=self.settings)
        self.llm = llm or create_llm(role="primary", settings=self.settings)

    def _build_extractive_fallback_answer(self, chunks: list[RetrievedChunk]) -> str:
        """
        Sinh câu trả lời trích xuất chuẩn xác ngay từ các chunk tìm được
        khi chạy ở chế độ Offline/Mock (chưa nhập API Key thật vào .env).
        """
        top = chunks[0].chunk
        other_refs = ", ".join(c.chunk.citation_label for c in chunks[1:3])
        extra_note = f" (Tham khảo liên quan: {other_refs})" if other_refs else ""
        return (
            f"**Căn cứ pháp lý:** {top.citation_label} — {top.article_title}.{extra_note}\n\n"
            f"**Nội dung quy định:** {top.content}\n\n"
            f"**Kết luận:** Áp dụng trực tiếp quy định tại {top.citation_label} nêu trên."
        )

    def answer_question(
        self,
        question: str,
        top_n: Optional[int] = None,
        law_filter: Optional[str] = None,
        llm_override: Optional[BaseLLM] = None,
    ) -> QAResponse:
        start_time = time.perf_counter()
        active_llm = llm_override or self.llm

        # Bước 1: Truy xuất Hybrid (BM25 + Vector + Reranker)
        retrieved = self.retriever.retrieve(
            query=question,
            top_n=top_n or self.settings.models.reranker.top_n,
            law_filter=law_filter,
        )

        # Bước 2: Kiểm tra ngưỡng tin cậy (Abstention Guardrail)
        min_conf = self.settings.guardrails.min_confidence_score
        if not retrieved or retrieved[0].scores.rerank_score < min_conf:
            elapsed = round((time.perf_counter() - start_time) * 1000, 2)
            return QAResponse(
                question=question,
                answer=(
                    "Hệ thống từ chối đưa ra phán đoán pháp lý do không tìm thấy điều khoản luật "
                    f"nào vượt ngưỡng tin cậy an toàn ({min_conf}). Vui lòng cung cấp thêm ngữ cảnh."
                ),
                retrieved_chunks=retrieved,
                is_citation_verified=True,
                unverified_citations=[],
                abstention_triggered=True,
                model_used=active_llm.model_name,
                latency_ms=elapsed,
            )

        # Bước 3: Gọi LLM sinh câu trả lời (Tự động chuyển sang Extractive Answer nếu chưa có key thật)
        context_block = format_retrieved_context(retrieved)
        user_prompt = (
            f"[NGỮ CẢNH PHÁP LÝ]:\n{context_block}\n\n"
            f"[CÂU HỎI TÌNH HUỐNG]:\n{question}\n\n"
            "Hãy phân tích và trả lời kèm trích dẫn chính xác [Tên Luật - Điều X - Khoản Y]."
        )

        provider_name = self.settings.models.llm_primary.provider
        if _has_real_key(provider_name):
            try:
                raw_answer = active_llm.generate_text(
                    prompt=user_prompt,
                    system_prompt=LEGAL_QA_SYSTEM_PROMPT,
                )
            except Exception:
                raw_answer = self._build_extractive_fallback_answer(retrieved)
        else:
            raw_answer = self._build_extractive_fallback_answer(retrieved)

        # Bước 4: Kiểm chứng chống ảo giác trích dẫn (Citation Verification)
        is_verified, unverified_list = CitationVerifier.verify_answer(raw_answer, retrieved)

        elapsed = round((time.perf_counter() - start_time) * 1000, 2)
        return QAResponse(
            question=question,
            answer=raw_answer,
            retrieved_chunks=retrieved,
            is_citation_verified=is_verified,
            unverified_citations=unverified_list,
            abstention_triggered=False,
            model_used=active_llm.model_name,
            latency_ms=elapsed,
        )