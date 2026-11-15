from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever_pipeline import HybridLegalRetriever


def test_layer3() -> None:
    print("=" * 70)
    print(" BẮT ĐẦU KIỂM TRA ĐỘNG CƠ HYBRID RETRIEVAL (LAYER 3 DIAGNOSTICS)")
    print("=" * 70)

    retriever = HybridLegalRetriever()
    print(f"\n[1/3] Đã nạp Cơ sở tri thức Pháp lý : {len(retriever.chunks)} Điều/Khoản mẫu")
    print(f"      Embedder đang dùng            : {retriever.embedder.model_name}")
    print(f"      Reranker đang dùng            : {retriever.reranker.model_name}")

    # Test truy vấn 1: Hỏi về tỷ lệ đặt cọc tối đa
    q1 = "Chủ đầu tư được thu tiền đặt cọc tối đa bao nhiêu phần trăm giá bán nhà ở hình thành trong tương lai?"
    hits1 = retriever.retrieve(q1, top_n=3, use_cache=False)
    print(f"\n[2/3] Truy vấn thử nghiệm 1: '{q1}'")
    for h in hits1:
        print(
            f"  + Hạng #{h.rank}: {h.chunk.citation_label} | "
            f"BM25={h.scores.bm25_score} | Vec={h.scores.vector_score} | "
            f"RRF={h.scores.rrf_score} | Rerank={h.scores.rerank_score}"
        )
    assert any(h.chunk.article_number == "Điều 23" for h in hits1), "Lỗi: Không tìm thấy Điều 23 về đặt cọc!"

    # Test truy vấn 2: Hỏi về bảo lãnh ngân hàng và thanh toán 95%
    q2 = "Khi chưa được cấp giấy chứng nhận quyền sử dụng đất thì chủ đầu tư được thu tối đa bao nhiêu phần trăm?"
    hits2 = retriever.retrieve(q2, top_n=3, use_cache=False)
    print(f"\n[3/3] Truy vấn thử nghiệm 2: '{q2}'")
    for h in hits2:
        print(
            f"  + Hạng #{h.rank}: {h.chunk.citation_label} ({h.chunk.article_title[:45]}...) | "
            f"Rerank={h.scores.rerank_score}"
        )
    assert any(h.chunk.article_number == "Điều 25" for h in hits2), "Lỗi: Không tìm thấy Điều 25 về thanh toán!"

    print("\n" + "=" * 70)
    print(" TẤT CẢ KIỂM TRA LỚP 3 ĐỀU VƯỢT QUA! SẴN SÀNG LÊN LỚP 4 (SERVICES).")
    print("=" * 70)


if __name__ == "__main__":
    test_layer3()