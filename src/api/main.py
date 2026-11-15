from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.core.schemas import ContractAuditReport, QAResponse
from src.services.contract_audit_service import ContractAuditService
from src.services.legal_qa_service import LegalQAService

settings = get_settings()
app = FastAPI(
    title=settings.project.name,
    version=settings.project.version,
    description="API Hệ thống Enhanced RAG Tra cứu Pháp lý & Kiểm toán Rủi ro Giao dịch Bất động sản",
)

qa_service = LegalQAService(settings=settings)
audit_service = ContractAuditService(retriever=qa_service.retriever, settings=settings)


class QARequestPayload(BaseModel):
    question: str = Field(..., min_length=5, description="Câu hỏi tình huống pháp lý BĐS")
    top_n: int = Field(default=5, ge=1, le=15)
    law_filter: Optional[str] = Field(default=None, description="Lọc theo tên bộ Luật (nếu cần)")


class ContractAuditPayload(BaseModel):
    contract_title: str = Field(default="Biên bản Đặt cọc & Mua bán BĐS")
    contract_text: str = Field(..., min_length=20, description="Toàn văn nội dung biên bản/hợp đồng cần thẩm định")


@app.get("/api/v1/health")
def health_check() -> dict:
    return {
        "status": "online",
        "project": settings.project.name,
        "version": settings.project.version,
        "knowledge_base_chunks": len(qa_service.retriever.chunks),
        "primary_llm": settings.models.llm_primary.model_name,
        "embedding_model": settings.models.embedding.model_name,
        "reranker_model": settings.models.reranker.model_name,
    }


@app.post("/api/v1/qa", response_model=QAResponse)
def ask_legal_question(payload: QARequestPayload) -> QAResponse:
    return qa_service.answer_question(
        question=payload.question,
        top_n=payload.top_n,
        law_filter=payload.law_filter,
    )


@app.post("/api/v1/audit-contract", response_model=ContractAuditReport)
def audit_real_estate_contract(payload: ContractAuditPayload) -> ContractAuditReport:
    return audit_service.audit_contract(
        contract_text=payload.contract_text,
        contract_title=payload.contract_title,
    )