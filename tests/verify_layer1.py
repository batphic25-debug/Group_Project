from __future__ import annotations

import sys
from pathlib import Path

# Đảm bảo Python luôn tìm thấy thư mục gốc của dự án dù chạy lệnh từ đâu
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.config import get_settings
from src.core.exceptions import APIKeyExhaustedError
from src.core.key_rotator import get_key_rotator
from src.core.schemas import (
    AuditFinding,
    ContractAuditReport,
    LegalChunk,
    QAResponse,
    RetrievalScoreDetail,
    RetrievedChunk,
    RiskLevel,
)


def run_diagnostics() -> None:
    print("=" * 65)
    print(" BẮT ĐẦU KIỂM TRA HỆ THỐNG NỀN TẢNG (LAYER 1 DIAGNOSTICS)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # TEST 1: Kiểm tra đọc cấu hình settings.yaml & Đường dẫn OS
    # ------------------------------------------------------------------
    print("\n[1/4] Đang kiểm tra Config & Đường dẫn (src/core/config.py)...")
    settings = get_settings()
    print(f"  + Tên dự án       : {settings.project.name} (v{settings.project.version})")
    print(f"  + Embedding Model : {settings.models.embedding.provider} / {settings.models.embedding.model_name}")
    print(f"  + Reranker Model  : {settings.models.reranker.provider} / {settings.models.reranker.model_name}")
    print(f"  + Primary LLM     : {settings.models.llm_primary.provider} / {settings.models.llm_primary.model_name}")
    print(f"  + Đường dẫn KB    : {settings.paths.knowledge_base_file}")
    assert settings.paths.knowledge_base_file.is_absolute(), "Lỗi: Đường dẫn chưa được chuyển thành tuyệt đối!"
    print("  => KẾT QUẢ TEST 1 : ĐẠT (OK)")

    # ------------------------------------------------------------------
    # TEST 2: Kiểm tra Schema LegalChunk (Hợp đồng dữ liệu với TV2 & TV3)
    # ------------------------------------------------------------------
    print("\n[2/4] Đang kiểm tra Schema LegalChunk (src/core/schemas.py)...")
    sample_chunk = LegalChunk(
        chunk_id="LKD_BDS_2023_D23_K5",
        law_name="Luật Kinh doanh bất động sản 2023",
        law_code="29/2023/QH15",
        chapter="Chương III",
        article_number="Điều 23",
        article_title="Nguyên tắc kinh doanh nhà ở, công trình xây dựng hình thành trong tương lai",
        clause_number="Khoản 5",
        content="Chủ đầu tư dự án bất động sản chỉ được thu tiền đặt cọc không quá 05% giá bán, cho thuê mua nhà ở...",
    )
    print(f"  + Nhãn trích dẫn  : {sample_chunk.citation_label}")
    print(f"  + Văn bản ngữ cảnh: {sample_chunk.contextual_text[:95]}...")
    assert "[Luật Kinh doanh bất động sản 2023 - Điều 23 - Khoản 5]" in sample_chunk.citation_label
    print("  => KẾT QUẢ TEST 2 : ĐẠT (OK)")

    # ------------------------------------------------------------------
    # TEST 3: Kiểm tra Schema QAResponse (TV6, TV7) & ContractAuditReport (TV1)
    # ------------------------------------------------------------------
    print("\n[3/4] Đang kiểm tra Schema Đầu ra QA & Audit Report...")
    retrieved = RetrievedChunk(
        chunk=sample_chunk,
        scores=RetrievalScoreDetail(bm25_score=12.4, vector_score=0.88, rrf_score=0.032, rerank_score=0.94),
        rank=1,
    )
    qa_sample = QAResponse(
        question="Đặt cọc mua nhà hình thành trong tương lai tối đa bao nhiêu phần trăm?",
        answer="Theo quy định tại [Luật Kinh doanh bất động sản 2023 - Điều 23 - Khoản 5], tối đa không quá 5%.",
        retrieved_chunks=[retrieved],
        model_used=settings.models.llm_primary.model_name,
        latency_ms=420.5,
    )
    audit_sample = ContractAuditReport(
        contract_title="Hợp đồng đặt cọc mua bán căn hộ mẫu (TV1)",
        total_clauses_checked=1,
        high_risk_count=1,
        medium_risk_count=0,
        safe_count=0,
        findings=[
            AuditFinding(
                clause_id="DIEU_2_DAT_COC",
                category="Đặt cọc & Phạt vi phạm",
                original_text="Bên B đặt cọc ngay 30% giá trị căn hộ hình thành trong tương lai khi ký biên bản này.",
                risk_level=RiskLevel.HIGH,
                summary="Mức thu tiền đặt cọc 30% vượt quá trần 5% theo luật định.",
                legal_basis=[retrieved],
                detailed_analysis="Vi phạm Khoản 5 Điều 23 Luật Kinh doanh BĐS 2023 (chỉ cho phép tối đa 5%).",
                suggested_revision="Điều chỉnh mức tiền đặt cọc xuống không quá 05% giá bán căn hộ.",
            )
        ],
        model_used=settings.models.llm_primary.model_name,
        processing_time_ms=1150.0,
    )
    print(f"  + QA Output JSON keys   : {list(qa_sample.model_dump().keys())}")
    print(f"  + Audit Report Risk     : {audit_sample.findings[0].risk_level.value} - {audit_sample.findings[0].summary}")
    print("  => KẾT QUẢ TEST 3 : ĐẠT (OK)")

    # ------------------------------------------------------------------
    # TEST 4: Kiểm tra Bộ xoay vòng API Key & Tự động Cooldown khi lỗi 429
    # ------------------------------------------------------------------
    print("\n[4/4] Đang kiểm tra Bộ xoay vòng API Keys (src/core/key_rotator.py)...")
    rotator = get_key_rotator("gemini")
    health = rotator.get_health_summary()
    print(f"  + Trạng thái ban đầu (Gemini): Tổng={health['total']}, Sẵn sàng={health['active']}")

    if health["total"] > 0:
        first_key = rotator.get_next_key()
        masked_key = f"{first_key[:6]}...{first_key[-4:]}" if len(first_key) > 10 else first_key
        print(f"  + Lấy Key lần 1 thành công   : {masked_key}")

        # Giả lập key đầu tiên bị dính Rate Limit 429 trong 5 giây
        rotator.mark_rate_limited(first_key, cooldown_seconds=5.0)
        health_after = rotator.get_health_summary()
        print(f"  + Sau khi giả lập lỗi 429    : Sẵn sàng={health_after['active']}, Đang nghỉ={health_after['cooling_down']}")

        if health["total"] >= 2:
            second_key = rotator.get_next_key()
            assert second_key != first_key, "Lỗi: Không tự động chuyển sang key thứ 2!"
            print("  + Tự động nhảy sang Key số 2 : THÀNH CÔNG")
        else:
            try:
                rotator.get_next_key()
            except APIKeyExhaustedError as exc:
                print(f"  + Bắt đúng ngoại lệ hết Key  : {exc.error_code}")
    else:
        print("  ! Cảnh báo: Bạn chưa điền GEMINI_API_KEYS trong file .env (nhưng code không bị crash).")

    print("  => KẾT QUẢ TEST 4 : ĐẠT (OK)")

    print("\n" + "=" * 65)
    print(" TẤT CẢ KIỂM TRA LỚP 1 ĐỀU VƯỢT QUA! SẴN SÀNG LÊN LỚP 2.")
    print("=" * 65)


if __name__ == "__main__":
    run_diagnostics()