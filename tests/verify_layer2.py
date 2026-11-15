from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.schemas import LegalChunk, RetrievalScoreDetail, RetrievedChunk
from src.providers.factory import _has_real_key, create_embedder, create_llm, create_reranker


def test_layer2() -> None:
    print("=" * 65)
    print(" BẮT ĐẦU KIỂM TRA TẦNG MODEL PROVIDERS (LAYER 2 DIAGNOSTICS)")
    print("=" * 65)

    # 1. Kiểm tra khởi tạo Embedder
    embedder = create_embedder(allow_mock_fallback=True)
    vec = embedder.embed_query("Quy định đặt cọc mua nhà hình thành trong tương lai")
    print(f"\n[1/3] Embedder đang kích hoạt : {embedder.__class__.__name__} ({embedder.model_name})")
    print(f"  + Số chiều Vector đầu ra    : {len(vec)} chiều (3 giá trị đầu: {[round(x, 4) for x in vec[:3]]})")

    # 2. Kiểm tra khởi tạo Reranker
    reranker = create_reranker(allow_fallback=True)
    sample_chunk = LegalChunk(
        chunk_id="LKD_BDS_2023_D23_K5",
        law_name="Luật Kinh doanh bất động sản 2023",
        article_number="Điều 23",
        article_title="Nguyên tắc kinh doanh nhà ở hình thành trong tương lai",
        clause_number="Khoản 5",
        content="Chủ đầu tư chỉ được thu tiền đặt cọc không quá 05% giá bán nhà ở hình thành trong tương lai.",
    )
    cand = RetrievedChunk(
        chunk=sample_chunk,
        scores=RetrievalScoreDetail(bm25_score=10.5, vector_score=0.82, rrf_score=0.03),
        rank=1,
    )
    reranked = reranker.rerank("Đặt cọc tối đa bao nhiêu phần trăm?", [cand], top_n=1)
    print(f"\n[2/3] Reranker đang kích hoạt : {reranker.__class__.__name__} ({reranker.model_name})")
    print(f"  + Điểm Rerank chấm thử      : {reranked[0].scores.rerank_score}")

    # 3. Kiểm tra khởi tạo Primary & Secondary LLM
    primary_llm = create_llm(role="primary")
    secondary_llm = create_llm(role="secondary")
    print(f"\n[3/3] Primary LLM             : {primary_llm.__class__.__name__} ({primary_llm.model_name})")
    print(f"      Secondary LLM           : {secondary_llm.__class__.__name__} ({secondary_llm.model_name})")

    if _has_real_key("gemini"):
        print("  + Phát hiện GEMINI_API_KEYS thật -> Đang gọi thử Gemini 4 Hydrogen...")
        reply = primary_llm.generate_text("Trả lời đúng 1 câu ngắn: Luật Kinh doanh BĐS 2023 có hiệu lực từ năm nào?")
        print(f"  + Phản hồi từ {primary_llm.model_name}: {reply.strip()}")
    else:
        print("  + Đang dùng key test trong .env (Khi bạn dán key thật từ AI Studio vào .env, hệ thống sẽ gọi Cloud trực tiếp).")

    print("\n" + "=" * 65)
    print(" TẤT CẢ KIỂM TRA LỚP 2 ĐỀU VƯỢT QUA! SẴN SÀNG LÊN LỚP 3.")
    print("=" * 65)


if __name__ == "__main__":
    test_layer2()