from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.config import LLMModelConfig, get_settings
from src.providers.factory import create_llm
from src.services.legal_qa_service import LegalQAService


class BenchmarkQuestionItem(BaseModel):
    """Cấu trúc 1 câu hỏi chuẩn do TV6 rà soát và đối chiếu Điều/Khoản luật."""

    question_id: str
    question: str
    expected_law: str
    expected_article: str
    expected_clause: Optional[str] = None
    expected_keywords: List[str] = Field(default_factory=list)
    reference_answer: str


class QuestionEvalDetail(BaseModel):
    """Kết quả chấm điểm chi tiết trên từng câu hỏi để TV6 kiểm tra câu thiếu/sai."""

    question_id: str
    question: str
    expected_article: str
    retrieved_top_articles: List[str]
    retrieval_hit: bool
    reciprocal_rank: float
    keyword_coverage: float
    citation_verified: bool
    latency_ms: float
    generated_answer: str


class ModelBenchmarkSummary(BaseModel):
    """Bảng tổng hợp điểm số của 1 mô hình để TV7 đưa vào báo cáo so sánh."""

    model_name: str
    provider: str
    total_questions: int
    hit_rate_at_3: float
    mrr_score: float
    avg_keyword_coverage: float
    citation_verified_rate: float
    avg_latency_ms: float
    details: List[QuestionEvalDetail] = Field(default_factory=list)


GOLDEN_SEED_QUESTIONS: List[Dict[str, Any]] = [
    {
        "question_id": "Q01",
        "question": "Chủ đầu tư dự án bất động sản được thu tiền đặt cọc tối đa bao nhiêu phần trăm giá bán nhà ở hình thành trong tương lai?",
        "expected_law": "Luật Kinh doanh bất động sản 2023",
        "expected_article": "Điều 23",
        "expected_clause": "Khoản 5",
        "expected_keywords": ["05%", "5%", "giá bán", "đủ điều kiện"],
        "reference_answer": "Theo Khoản 5 Điều 23 Luật Kinh doanh BĐS 2023, chủ đầu tư chỉ được thu tiền đặt cọc không quá 05% giá bán khi nhà ở đã đủ điều kiện đưa vào kinh doanh.",
    },
    {
        "question_id": "Q02",
        "question": "Điều kiện về xây dựng để nhà chung cư hình thành trong tương lai được phép đưa vào kinh doanh là gì?",
        "expected_law": "Luật Kinh doanh bất động sản 2023",
        "expected_article": "Điều 24",
        "expected_clause": "Khoản 1",
        "expected_keywords": ["nghiệm thu", "phần móng", "hạ tầng kỹ thuật"],
        "reference_answer": "Theo Khoản 1 Điều 24 Luật Kinh doanh BĐS 2023, đối với nhà chung cư, tòa nhà hỗn hợp có nhà ở phải có giấy tờ nghiệm thu đã hoàn thành xây dựng xong phần móng.",
    },
    {
        "question_id": "Q03",
        "question": "Việc thanh toán trong mua bán nhà ở hình thành trong tương lai lần đầu và trước khi bàn giao nhà tối đa là bao nhiêu phần trăm?",
        "expected_law": "Luật Kinh doanh bất động sản 2023",
        "expected_article": "Điều 25",
        "expected_clause": "Khoản 1",
        "expected_keywords": ["30%", "70%", "bàn giao"],
        "reference_answer": "Theo Khoản 1 Điều 25 Luật Kinh doanh BĐS 2023, thanh toán lần đầu không quá 30% giá trị hợp đồng (gồm cả tiền đặt cọc), tổng số trước khi bàn giao nhà không quá 70%.",
    },
    {
        "question_id": "Q04",
        "question": "Khi người mua chưa được cấp Giấy chứng nhận quyền sử dụng đất, quyền sở hữu tài sản gắn liền với đất thì bên bán được thu tối đa bao nhiêu phần trăm giá trị hợp đồng?",
        "expected_law": "Luật Kinh doanh bất động sản 2023",
        "expected_article": "Điều 25",
        "expected_clause": "Khoản 3",
        "expected_keywords": ["95%", "giấy chứng nhận"],
        "reference_answer": "Theo Khoản 3 Điều 25 Luật Kinh doanh BĐS 2023, bên bán không được thu quá 95% giá trị hợp đồng khi bên mua chưa được cấp Giấy chứng nhận.",
    },
    {
        "question_id": "Q05",
        "question": "Chủ đầu tư bán nhà ở hình thành trong tương lai có bắt buộc phải có bảo lãnh ngân hàng thương mại hay không?",
        "expected_law": "Luật Kinh doanh bất động sản 2023",
        "expected_article": "Điều 26",
        "expected_clause": "Khoản 1",
        "expected_keywords": ["bảo lãnh", "ngân hàng thương mại", "nghĩa vụ tài chính"],
        "reference_answer": "Theo Khoản 1 Điều 26 Luật Kinh doanh BĐS 2023, chủ đầu tư trước khi bán, cho thuê mua nhà ở hình thành trong tương lai phải được ngân hàng thương mại chấp thuận cấp bảo lãnh.",
    },
    {
        "question_id": "Q06",
        "question": "Hợp đồng mua bán, thuê mua nhà ở mà các bên tham gia giao dịch là cá nhân có bắt buộc phải công chứng hoặc chứng thực không?",
        "expected_law": "Luật Kinh doanh bất động sản 2023",
        "expected_article": "Điều 44",
        "expected_clause": "Khoản 4",
        "expected_keywords": ["cá nhân", "công chứng", "chứng thực"],
        "reference_answer": "Theo Khoản 4 Điều 44 Luật Kinh doanh BĐS 2023 và Khoản 1 Điều 164 Luật Nhà ở 2023, giao dịch mua bán nhà ở giữa các cá nhân bắt buộc phải công chứng hoặc chứng thực.",
    },
    {
        "question_id": "Q07",
        "question": "Chủ đầu tư dự án bất động sản nhận tiền thanh toán hợp đồng từ khách hàng theo hình thức nào?",
        "expected_law": "Luật Kinh doanh bất động sản 2023",
        "expected_article": "Điều 47",
        "expected_clause": "Khoản 1",
        "expected_keywords": ["tài khoản", "tổ chức tín dụng", "ngân hàng"],
        "reference_answer": "Theo Khoản 1 Điều 47 Luật Kinh doanh BĐS 2023, chủ đầu tư bắt buộc nhận tiền thanh toán từ khách hàng thông qua tài khoản mở tại tổ chức tín dụng hoạt động hợp pháp tại Việt Nam.",
    },
    {
        "question_id": "Q08",
        "question": "Thời điểm có hiệu lực của hợp đồng mua bán nhà ở thuộc trường hợp phải công chứng, chứng thực được xác định như thế nào?",
        "expected_law": "Luật Nhà ở 2023",
        "expected_article": "Điều 164",
        "expected_clause": "Khoản 1",
        "reference_answer": "Theo Khoản 1 Điều 164 Luật Nhà ở 2023, thời điểm có hiệu lực của hợp đồng là thời điểm hoàn thành việc công chứng hoặc chứng thực.",
        "expected_keywords": ["thời điểm", "công chứng", "chứng thực"],
    },
    {
        "question_id": "Q09",
        "question": "Người sử dụng đất cần đáp ứng những điều kiện nào để được thực hiện quyền chuyển nhượng quyền sử dụng đất?",
        "expected_law": "Luật Đất đai 2024",
        "expected_article": "Điều 45",
        "expected_clause": "Khoản 1",
        "expected_keywords": ["giấy chứng nhận", "tranh chấp", "kê biên", "thời hạn"],
        "reference_answer": "Theo Khoản 1 Điều 45 Luật Đất đai 2024, cần có Giấy chứng nhận, đất không có tranh chấp, không bị kê biên để bảo đảm thi hành án và còn trong thời hạn sử dụng đất.",
    },
    {
        "question_id": "Q10",
        "question": "Theo Bộ luật Dân sự, nếu bên nhận đặt cọc từ chối việc giao kết, thực hiện hợp đồng thì phải chịu trách nhiệm gì?",
        "expected_law": "Bộ luật Dân sự 2015",
        "expected_article": "Điều 328",
        "expected_clause": "Khoản 2",
        "expected_keywords": ["trả cho bên đặt cọc", "tương đương giá trị", "thỏa thuận khác"],
        "reference_answer": "Theo Khoản 2 Điều 328 Bộ luật Dân sự 2015, bên nhận đặt cọc phải trả cho bên đặt cọc tài sản đặt cọc và một khoản tiền tương đương giá trị tài sản đặt cọc, trừ trường hợp có thỏa thuận khác.",
    },
]


def ensure_benchmark_dataset() -> List[BenchmarkQuestionItem]:
    """Đảm bảo file dataset_50_qa.json của TV6 luôn sẵn sàng để TV7 chạy Benchmark."""
    settings = get_settings()
    ds_path: Path = settings.paths.benchmark_dataset_file
    ds_path.parent.mkdir(parents=True, exist_ok=True)

    if not ds_path.exists():
        with ds_path.open("w", encoding="utf-8") as f:
            json.dump(GOLDEN_SEED_QUESTIONS, f, ensure_ascii=False, indent=2)

    raw_data = json.loads(ds_path.read_text(encoding="utf-8"))
    return [BenchmarkQuestionItem.model_validate(item) for item in raw_data]


class BenchmarkRunner:
    """
    Bộ chạy Benchmark tự động (Phục vụ TV6 đối chiếu câu trả lời & TV7 thống kê điểm các Model).
    """

    def __init__(self, qa_service: Optional[LegalQAService] = None) -> None:
        self.settings = get_settings()
        self.qa_service = qa_service or LegalQAService(settings=self.settings)
        self.questions = ensure_benchmark_dataset()

    def evaluate_model(
        self,
        model_config: LLMModelConfig,
        max_questions: Optional[int] = None,
    ) -> ModelBenchmarkSummary:
        llm_instance = create_llm(custom_config=model_config, settings=self.settings)
        subset = self.questions[:max_questions] if max_questions else self.questions

        details: List[QuestionEvalDetail] = []
        hits = 0
        rr_sum = 0.0
        kw_sum = 0.0
        verified_cnt = 0
        latency_sum = 0.0

        for item in subset:
            qa_res = self.qa_service.answer_question(
                question=item.question,
                top_n=3,
                llm_override=llm_instance,
            )

            top_articles = [c.chunk.article_number for c in qa_res.retrieved_chunks]
            hit = False
            rr = 0.0
            for idx, art in enumerate(top_articles, start=1):
                if art.strip().lower() == item.expected_article.strip().lower():
                    hit = True
                    rr = 1.0 / idx
                    break

            ans_lower = qa_res.answer.lower()
            if item.expected_keywords:
                matched_kw = sum(1 for kw in item.expected_keywords if kw.lower() in ans_lower)
                kw_cov = matched_kw / len(item.expected_keywords)
            else:
                kw_cov = 1.0

            if hit:
                hits += 1
            rr_sum += rr
            kw_sum += kw_cov
            if qa_res.is_citation_verified:
                verified_cnt += 1
            latency_sum += qa_res.latency_ms

            details.append(
                QuestionEvalDetail(
                    question_id=item.question_id,
                    question=item.question,
                    expected_article=f"{item.expected_law} - {item.expected_article}",
                    retrieved_top_articles=top_articles,
                    retrieval_hit=hit,
                    reciprocal_rank=round(rr, 4),
                    keyword_coverage=round(kw_cov * 100, 1),
                    citation_verified=qa_res.is_citation_verified,
                    latency_ms=qa_res.latency_ms,
                    generated_answer=qa_res.answer,
                )
            )

        total = max(len(subset), 1)
        summary = ModelBenchmarkSummary(
            model_name=model_config.model_name,
            provider=model_config.provider,
            total_questions=len(subset),
            hit_rate_at_3=round((hits / total) * 100, 2),
            mrr_score=round(rr_sum / total, 4),
            avg_keyword_coverage=round((kw_sum / total) * 100, 2),
            citation_verified_rate=round((verified_cnt / total) * 100, 2),
            avg_latency_ms=round(latency_sum / total, 2),
            details=details,
        )

        # Lưu kết quả ra thư mục evaluation/reports/ cho TV6 và TV7
        reports_dir = self.settings.paths.benchmark_dataset_file.parent / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        safe_name = model_config.model_name.replace("/", "_").replace(":", "_")
        out_file = reports_dir / f"benchmark_{safe_name}.json"
        out_file.write_text(
            json.dumps(summary.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary