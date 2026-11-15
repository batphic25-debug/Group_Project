from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.seed_data import SAMPLE_CONTRACT_DEMO_TEXT
from src.services.citation_verifier import CitationVerifier
from src.services.contract_audit_service import ContractAuditService
from src.services.legal_qa_service import LegalQAService


def test_layer4() -> None:
    print("=" * 72)
    print(" BẮT ĐẦU KIỂM TRA TẦNG NGHIỆP VỤ & GUARDRAILS (LAYER 4 DIAGNOSTICS)")
    print("=" * 72)

    # 1. Test Luồng 1: Hỏi đáp Pháp lý (LegalQAService)
    qa_service = LegalQAService()
    q = "Chủ đầu tư được thu tiền đặt cọc tối đa bao nhiêu phần trăm giá bán nhà ở hình thành trong tương lai?"
    qa_res = qa_service.answer_question(q)

    print(f"\n[1/3] Kiểm tra Luồng Hỏi đáp Pháp lý (Model: {qa_res.model_used}):")
    print(f"  + Câu hỏi             : {qa_res.question}")
    print(f"  + Trích dẫn Top 1     : {qa_res.retrieved_chunks[0].chunk.citation_label}")
    print(f"  + Xác thực trích dẫn  : Verified={qa_res.is_citation_verified} (Lỗi bịa={qa_res.unverified_citations})")
    print(f"  + Thời gian phản hồi  : {qa_res.latency_ms} ms")
    assert qa_res.is_citation_verified, "Lỗi: Trích dẫn không vượt qua bước xác thực!"

    # 2. Test Chốt chặn Chống Ảo giác (CitationVerifier)
    fake_answer = "Theo quy định tại Điều 23 và Điều 999 Luật Kinh doanh BĐS, mức phạt là 50%."
    is_ok, bad_list = CitationVerifier.verify_answer(fake_answer, qa_res.retrieved_chunks)
    print("\n[2/3] Kiểm tra Chốt chặn Chống Ảo giác (CitationVerifier):")
    print(f"  + Câu trả lời giả lập bịa 'Điều 999' -> Kết quả: Verified={is_ok}, Bắt được={bad_list}")
    assert not is_ok and "Điều 999" in bad_list, "Lỗi: Không bắt được Điều 999 giả mạo!"

    # 3. Test Luồng 2: Kiểm toán Hợp đồng BĐS tự động (ContractAuditService)
    auditor = ContractAuditService(retriever=qa_service.retriever)
    report = auditor.audit_contract(
        contract_text=SAMPLE_CONTRACT_DEMO_TEXT,
        contract_title="Biên bản Đặt cọc & Mua bán Căn hộ Hòa Lạc Skyview (Mẫu Demo TV1)",
    )
    print(f"\n[3/3] Kiểm tra Luồng Thẩm định Hợp đồng BĐS ({report.contract_title}):")
    print(
        f"  + Tổng số điều khoản quét: {report.total_clauses_checked} | "
        f"Rủi ro CAO (Đỏ): {report.high_risk_count} | "
        f"Trung bình (Vàng): {report.medium_risk_count} | "
        f"An toàn (Xanh): {report.safe_count}"
    )
    for f in report.findings:
        basis_label = f.legal_basis[0].chunk.citation_label if f.legal_basis else "N/A"
        print(f"    - [{f.clause_id} - {f.risk_level.value}] {f.summary}")
        print(f"      Căn cứ bắt lỗi : {basis_label}")

    assert report.high_risk_count >= 3, "Lỗi: Hệ thống không bắt đủ các lỗi pháp lý cài sẵn trong hợp đồng!"

    print("\n" + "=" * 72)
    print(" TẤT CẢ KIỂM TRA LỚP 4 ĐỀU VƯỢT QUA! SẴN SÀNG LÊN LỚP 5 (BENCHMARK & UI).")
    print("=" * 72)


if __name__ == "__main__":
    test_layer4()