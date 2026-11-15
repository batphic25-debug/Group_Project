from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Mức độ rủi ro pháp lý của từng điều khoản trong biên bản/hợp đồng BĐS."""

    HIGH = "HIGH"              # Vi phạm trực tiếp quy định bắt buộc của Luật / Vô hiệu
    MEDIUM = "MEDIUM"          # Điều khoản bất lợi, thiếu rõ ràng hoặc thiếu hồ sơ kèm theo
    SAFE = "SAFE"              # Phù hợp với quy định pháp luật hiện hành
    UNVERIFIED = "UNVERIFIED"  # Không đủ dữ liệu luật trong DB để kết luận


class LegalChunk(BaseModel):
    """
    Cấu trúc chuẩn của một đơn vị kiến thức pháp lý sau khi chia chunk.
    - TV2 chịu trách nhiệm tách cấp Điều/Khoản và điền 'article_title' (tên Điều cha).
    - TV3 chịu trách nhiệm chuẩn hóa metadata ('law_name', 'article_number', 'clause_number').
    """

    chunk_id: str = Field(..., description="Mã định danh duy nhất, ví dụ: LKD_BDS_2023_D23_K5")
    law_name: str = Field(..., description="Tên văn bản luật, ví dụ: Luật Kinh doanh bất động sản 2023")
    law_code: Optional[str] = Field(default=None, description="Số hiệu văn bản, ví dụ: 29/2023/QH15")
    chapter: Optional[str] = Field(default=None, description="Tên Chương, ví dụ: Chương III")
    article_number: str = Field(..., description="Số hiệu Điều, ví dụ: Điều 23")
    article_title: str = Field(..., description="Tiêu đề Điều cha giúp bảo toàn ngữ cảnh pháp lý")
    clause_number: Optional[str] = Field(default=None, description="Số hiệu Khoản, ví dụ: Khoản 5")
    content: str = Field(..., min_length=1, description="Nội dung nguyên văn của Khoản/Điều luật")
    effective_date: Optional[str] = Field(default="2024-08-01", description="Ngày có hiệu lực")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata mở rộng")

    @property
    def citation_label(self) -> str:
        """Tạo nhãn trích dẫn chuẩn hiển thị cho người dùng, ví dụ: [Luật Kinh doanh BĐS 2023 - Điều 23 - Khoản 5]."""
        parts = [self.law_name, self.article_number]
        if self.clause_number:
            parts.append(self.clause_number)
        return f"[{' - '.join(parts)}]"

    @property
    def contextual_text(self) -> str:
        """
        Ghép ngữ cảnh Điều cha vào nội dung Khoản con trước khi đưa vào Embedding.
        Giúp giải quyết triệt để hiện tượng mất ngữ cảnh khi cắt chunk nhỏ.
        """
        header = f"{self.citation_label} ({self.article_title})"
        return f"{header}: {self.content}"


class RetrievalScoreDetail(BaseModel):
    """Lưu chi tiết điểm số từng bước để hiển thị trên Tab 'Giải phẫu RAG' khi Demo."""

    bm25_score: float = Field(default=0.0, description="Điểm khớp từ khóa chính xác BM25")
    vector_score: float = Field(default=0.0, description="Điểm tương đồng ngữ nghĩa Cosine")
    rrf_score: float = Field(default=0.0, description="Điểm tổng hợp Reciprocal Rank Fusion")
    rerank_score: float = Field(default=0.0, description="Điểm đánh giá chéo từ mô hình Reranker")


class RetrievedChunk(BaseModel):
    """Một chunk luật sau khi đi qua tầng Hybrid Search & Reranking."""

    chunk: LegalChunk
    scores: RetrievalScoreDetail
    rank: int = Field(..., ge=1, description="Thứ hạng sau khi Rerank (1 là liên quan nhất)")


class QAResponse(BaseModel):
    """
    Kết quả đầu ra chuẩn của Luồng 1 (Hỏi đáp Pháp lý).
    Phục vụ trực tiếp cho TV6 (rà soát thủ công) và TV7 (chạy Benchmark 50 câu).
    """

    question: str
    answer: str
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    is_citation_verified: bool = Field(
        default=True,
        description="True nếu mọi Điều/Khoản LLM trích dẫn đều khớp với retrieved_chunks",
    )
    unverified_citations: list[str] = Field(
        default_factory=list,
        description="Danh sách các Điều/Khoản bị phát hiện trích dẫn sai/ngoài ngữ cảnh",
    )
    abstention_triggered: bool = Field(
        default=False,
        description="True nếu hệ thống chủ động từ chối trả lời do điểm Rerank dưới ngưỡng an toàn",
    )
    model_used: str
    latency_ms: float = Field(default=0.0, ge=0.0)


class ExtractedContractClause(BaseModel):
    """Cấu trúc một điều khoản được LLM bóc tách từ Biên bản/Hợp đồng BĐS của khách hàng."""

    clause_id: str = Field(..., description="Mã điều khoản trong hợp đồng, ví dụ: DIEU_03_DAT_COC")
    category: str = Field(
        ...,
        description="Phân loại: Chủ thể | Đối tượng BĐS | Giá & Thanh toán | Đặt cọc & Phạt vi phạm | Bàn giao & Sổ hồng",
    )
    original_text: str = Field(..., description="Đoạn văn bản trích nguyên văn từ hợp đồng khách hàng")
    legal_audit_query: str = Field(
        ...,
        description="Câu truy vấn pháp lý được sinh tự động để đối chiếu điều khoản này với kho Luật",
    )


class AuditFinding(BaseModel):
    """Kết quả kiểm toán rủi ro cho một điều khoản cụ thể trong hợp đồng."""

    clause_id: str
    category: str
    original_text: str
    risk_level: RiskLevel
    summary: str = Field(..., description="Tóm tắt ngắn gọn kết luận rủi ro (1 câu)")
    legal_basis: list[RetrievedChunk] = Field(
        default_factory=list,
        description="Các Điều/Khoản luật dùng làm căn cứ đối chiếu",
    )
    detailed_analysis: str = Field(..., description="Phân tích chi tiết hợp pháp hay vi phạm ở điểm nào")
    suggested_revision: Optional[str] = Field(
        default=None,
        description="Đề xuất câu chữ sửa lại điều khoản hợp đồng cho đúng luật",
    )


class ContractAuditReport(BaseModel):
    """
    Kết quả đầu ra chuẩn của Luồng 2 (Thẩm định Hồ sơ & Biên bản Mua bán BĐS thực tế của TV1).
    Hiển thị trực tiếp trên màn hình Split-Screen Auditor khi Demo.
    """

    contract_title: str
    total_clauses_checked: int
    high_risk_count: int
    medium_risk_count: int
    safe_count: int
    findings: list[AuditFinding] = Field(default_factory=list)
    model_used: str
    processing_time_ms: float = Field(default=0.0, ge=0.0)