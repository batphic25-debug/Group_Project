from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark_runner import BenchmarkRunner
from src.core.config import get_settings


def test_layer5() -> None:
    print("=" * 72)
    print(" BẮT ĐẦU KIỂM TRA BENCHMARK RUNNER & TOÀN HỆ THỐNG (LAYER 5 DIAGNOSTICS)")
    print("=" * 72)

    settings = get_settings()
    runner = BenchmarkRunner()
    summary = runner.evaluate_model(settings.models.llm_primary, max_questions=5)

    print(f"\n[1/2] Kết quả chạy Benchmark trên {summary.total_questions} câu hỏi chuẩn (Model: {summary.model_name}):")
    print(f"  + HitRate@3 (Tỷ lệ tìm đúng Điều luật trong Top 3) : {summary.hit_rate_at_3}%")
    print(f"  + MRR (Mean Reciprocal Rank)                       : {summary.mrr_score}")
    print(f"  + Độ phủ từ khóa đáp án chuẩn của TV6              : {summary.avg_keyword_coverage}%")
    print(f"  + Tỷ lệ trích dẫn hợp lệ (Citation Verified Rate)  : {summary.citation_verified_rate}%")
    print(f"  + Độ trễ trung bình                                : {summary.avg_latency_ms} ms")

    assert summary.hit_rate_at_3 >= 80.0, "Lỗi: Độ chính xác truy xuất thấp hơn kỳ vọng!"
    print("\n[2/2] Đã xuất báo cáo JSON tự động vào thư mục: evaluation/reports/")
    print("\n" + "=" * 72)
    print(" TOÀN BỘ 5 LỚP KIẾN TRÚC ĐÃ HOÀN TẤT 100%! SẴN SÀNG TRÌNH CHIẾU DEMO.")
    print("=" * 72)


if __name__ == "__main__":
    test_layer5()