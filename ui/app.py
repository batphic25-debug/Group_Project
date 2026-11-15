from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.benchmark_runner import BenchmarkRunner
from src.core.config import LLMModelConfig, get_settings
from src.core.key_rotator import get_key_rotator
from src.core.schemas import RiskLevel
from src.providers.factory import _has_real_key, create_llm
from src.retrieval.seed_data import SAMPLE_CONTRACT_DEMO_TEXT
from src.services.contract_audit_service import ContractAuditService
from src.services.legal_qa_service import LegalQAService

st.set_page_config(
    page_title="BĐS Legal Auditor & Enhanced RAG Workstation",
    page_icon="⚖️",
    layout="wide",
)


@st.cache_resource
def init_core_services():
    cfg = get_settings()
    qa_svc = LegalQAService(settings=cfg)
    audit_svc = ContractAuditService(retriever=qa_svc.retriever, settings=cfg)
    bench_runner = BenchmarkRunner(qa_service=qa_svc)
    return cfg, qa_svc, audit_svc, bench_runner


settings, qa_service, audit_service, benchmark_runner = init_core_services()

# =====================================================================
# SIDEBAR: BẢNG ĐIỀU KHIỂN KIẾN TRÚC & CHUYỂN ĐỔI MÔ HÌNH (TV4 vs TV5)
# =====================================================================
with st.sidebar:
    st.markdown("## ⚖️ BĐS Legal Workstation")
    st.caption(f"Phiên bản: `{settings.project.version}` | Kiến trúc: **Enhanced Hybrid RAG**")
    st.divider()

    st.markdown("### 🧠 Tầng Mô hình AI (TV4 & TV5)")
    selected_llm_choice = st.selectbox(
        "Chọn Mô hình Sinh (LLM):",
        options=[
            "gemini | gemini-4-hydrogen (Primary - TV5)",
            "gemini | gemini-3.8-flash (Cloud API - TV5)",
            "groq | qwen/qwen3.8-27b (High-Speed LPU - TV5)",
            "ollama | gemma-4-26B-A4B-it (Open LLM - TV4)",
            "ollama | gemma4-12b-vi:latest (Open LLM - TV4)",
        ],
        index=0,
    )
    provider_str, model_label = [x.strip() for x in selected_llm_choice.split("|")]
    active_model_name = model_label.split(" ")[0]

    st.markdown(
        f"- **Embedding:** `{settings.models.embedding.model_name}` ({settings.models.embedding.dimensions}d)\n"
        f"- **Reranker:** `{settings.models.reranker.model_name}` (Top-{settings.models.reranker.top_n})\n"
        f"- **Kho Luật (TV2 & TV3):** `{len(qa_service.retriever.chunks)}` Điều/Khoản chuẩn hóa"
    )
    st.divider()

    st.markdown("### 🔑 Trạng thái API Key Rotator")
    for prov in ["gemini", "voyage", "groq"]:
        health = get_key_rotator(prov).get_health_summary()
        real_mode = "Cloud Live" if _has_real_key(prov) else "Deterministic Sandbox"
        st.caption(f"• **{prov.upper()}**: `{health['active']}/{health['total']}` keys ({real_mode})")

    st.divider()
    st.info(
        "**Chốt chặn Chống Ảo giác (Guardrails):**\n"
        f"- Ngưỡng Rerank tối thiểu: `{settings.guardrails.min_confidence_score}`\n"
        f"- Xác thực số hiệu Điều/Khoản: `BẬT`"
    )

# Khởi tạo LLM theo lựa chọn trên thanh Sidebar
active_llm_config = LLMModelConfig(
    provider=provider_str,  # type: ignore[arg-type]
    model_name=active_model_name,
    temperature=0.1,
    max_tokens=4096,
)
active_llm = create_llm(custom_config=active_llm_config, settings=settings)

# =====================================================================
# HEADER CHÍNH & 3 PHÂN HỆ SẢN PHẨM
# =====================================================================
st.title("🏛️ Hệ thống Kiểm toán Hồ sơ Pháp lý & Tra cứu Luật Bất động sản")
st.markdown(
    "**Đồ án Nhóm (7 Thành viên):** Kết hợp **Structural Legal Chunking (TV2, TV3)** + "
    "**Hybrid Search BM25/Vector & Cloud Reranking (TV5)** + **Automated Contract Due Diligence (TV1)** + "
    "**Multi-Model Benchmark (TV4, TV6, TV7)**."
)

tab_auditor, tab_explorer, tab_benchmark = st.tabs(
    [
        "📑 Phân hệ 1: Thẩm định Hợp đồng & Biên bản BĐS (TV1 & TV5)",
        "🔍 Phân hệ 2: Tra cứu Pháp lý & Giải phẫu RAG (TV2, TV3 & TV5)",
        "📊 Phân hệ 3: Đấu trường Benchmark 50 Câu hỏi (TV4, TV6 & TV7)",
    ]
)

# ---------------------------------------------------------------------
# TAB 1: SPLIT-SCREEN CONTRACT AUDITOR (THẨM ĐỊNH HỢP ĐỒNG SONG SONG)
# ---------------------------------------------------------------------
with tab_auditor:
    st.subheader("Kiểm toán Rủi ro Biên bản Đặt cọc & Hợp đồng Mua bán BĐS Tự động")
    col_left, col_right = st.columns([1.05, 1.25], gap="large")

    with col_left:
        st.markdown("#### 📄 Văn bản Hợp đồng / Biên bản Giao dịch (Dữ liệu TV1)")
        contract_title_input = st.text_input(
            "Tên hồ sơ / Biên bản kiểm toán:",
            value="Biên bản Đặt cọc & Mua bán Căn hộ Hòa Lạc Skyview (Mẫu thực tế TV1)",
        )
        contract_text_input = st.text_area(
            "Nội dung toàn văn Biên bản / Hợp đồng (Có thể chỉnh sửa trực tiếp để test):",
            value=SAMPLE_CONTRACT_DEMO_TEXT,
            height=410,
        )
        run_audit_btn = st.button("🚨 Chạy Kiểm toán Pháp lý Toàn diện", type="primary", use_container_width=True)

    with col_right:
        st.markdown("#### 🛡️ Báo cáo Phát hiện Rủi ro & Đối chiếu Điều/Khoản Luật")
        if run_audit_btn or "last_audit_report" not in st.session_state:
            with st.spinner(f"Đang bóc tách mệnh đề và đối chiếu kho Luật bằng {active_model_name}..."):
                st.session_state["last_audit_report"] = audit_service.audit_contract(
                    contract_text=contract_text_input,
                    contract_title=contract_title_input,
                    llm_override=active_llm,
                )

        report = st.session_state["last_audit_report"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Tổng điều khoản", report.total_clauses_checked)
        m2.metric("🔴 Vi phạm / Rủi ro Cao", report.high_risk_count)
        m3.metric("🟡 Cảnh báo Trung bình", report.medium_risk_count)
        m4.metric("🟢 Hợp lệ / An toàn", report.safe_count)
        st.caption(f"Model thực thi: `{report.model_used}` | Thời gian xử lý: `{report.processing_time_ms} ms`")

        for finding in report.findings:
            if finding.risk_level == RiskLevel.HIGH:
                badge = "🔴 [VI PHẠM PHÁP LUẬT - RỦI RO CAO]"
            elif finding.risk_level == RiskLevel.MEDIUM:
                badge = "🟡 [CẢNH BÁO RỦI RO TRUNG BÌNH]"
            else:
                badge = "🟢 [PHÙ HỢP QUY ĐỊNH]"

            with st.expander(f"{badge} {finding.clause_id} ({finding.category}): {finding.summary}", expanded=True):
                st.markdown(f"**1. Đoạn trích trong Hợp đồng khách hàng:**\n> _{finding.original_text}_")
                st.markdown(f"**2. Phân tích & Đối chiếu Pháp lý:**\n{finding.detailed_analysis}")
                if finding.legal_basis:
                    top_law = finding.legal_basis[0]
                    st.markdown(
                        f"**3. Căn cứ Điều/Khoản bắt lỗi (Điểm Rerank `{top_law.scores.rerank_score}`):**\n"
                        f"- **{top_law.chunk.citation_label}** *(Điều cha: {top_law.chunk.article_title})*:\n"
                        f"  `{top_law.chunk.content}`"
                    )
                if finding.suggested_revision:
                    st.success(f"💡 **Đề xuất sửa lại điều khoản:** {finding.suggested_revision}")

# ---------------------------------------------------------------------
# TAB 2: HIERARCHICAL LEGAL EXPLORER & RAG ANATOMY (GIẢI PHẪU RAG)
# ---------------------------------------------------------------------
with tab_explorer:
    st.subheader("Tra cứu Tình huống Pháp lý & Giải phẫu Luồng Điểm Hybrid RAG")
    sample_queries = [
        "Chủ đầu tư được thu tiền đặt cọc tối đa bao nhiêu phần trăm giá bán nhà ở hình thành trong tương lai?",
        "Khi người mua chưa được cấp Giấy chứng nhận (Sổ hồng) thì bên bán được thu tối đa bao nhiêu phần trăm?",
        "Mua bán nhà ở giữa các cá nhân viết tay không công chứng có hiệu lực pháp lý không?",
        "Điều kiện nghiệm thu để tòa nhà chung cư hình thành trong tương lai được phép mở bán là gì?",
    ]
    chosen_sample = st.selectbox("Chọn nhanh câu hỏi tình huống mẫu:", options=sample_queries)
    user_query = st.text_input("Hoặc nhập câu hỏi pháp lý bất kỳ:", value=chosen_sample)

    col_f1, col_f2, col_btn = st.columns([2, 1, 1])
    with col_f1:
        law_filter_opt = st.selectbox(
            "Lọc cứng theo Metadata Bộ Luật (TV3):",
            options=[
                "Tất cả văn bản Luật",
                "Luật Kinh doanh bất động sản 2023",
                "Luật Nhà ở 2023",
                "Luật Đất đai 2024",
                "Bộ luật Dân sự 2015",
            ],
        )
    with col_f2:
        top_n_slider = st.slider("Số lượng Chunks sau Rerank:", min_value=1, max_value=5, value=3)
    with col_btn:
        st.write("")
        st.write("")
        ask_btn = st.button("🔍 Truy xuất & Phân tích", type="primary", use_container_width=True)

    if ask_btn or "last_qa_response" not in st.session_state:
        actual_filter = None if law_filter_opt == "Tất cả văn bản Luật" else law_filter_opt
        with st.spinner("Đang chạy Hybrid Search (BM25 + Vector) -> RRF -> Reranker -> LLM..."):
            st.session_state["last_qa_response"] = qa_service.answer_question(
                question=user_query,
                top_n=top_n_slider,
                law_filter=actual_filter,
                llm_override=active_llm,
            )

    qa_res = st.session_state["last_qa_response"]
    c_ans, c_trace = st.columns([1.15, 1.15], gap="large")

    with c_ans:
        st.markdown("#### 💬 Câu trả lời Pháp lý & Trạng thái Kiểm chứng")
        if qa_res.is_citation_verified:
            st.success(
                f"✅ **Citation Verified:** Mọi trích dẫn Điều/Khoản đều khớp 100% với Metadata thật | "
                f"Model: `{qa_res.model_used}` | Độ trễ: `{qa_res.latency_ms} ms`"
            )
        else:
            st.error(f"⚠️ Phát hiện trích dẫn ngoài ngữ cảnh: {qa_res.unverified_citations}")

        st.markdown(qa_res.answer)

    with c_trace:
        st.markdown("#### 🔬 Giải phẫu Bên dưới Nắp Capo (BM25 + Vector + RRF + Rerank)")
        for item in qa_res.retrieved_chunks:
            with st.expander(
                f"Hạng #{item.rank}: {item.chunk.citation_label} — Rerank: {item.scores.rerank_score:.4f}",
                expanded=(item.rank == 1),
            ):
                s1, s2, s3, s4 = st.columns(4)
                s1.metric("BM25 (Từ khóa)", f"{item.scores.bm25_score:.2f}")
                s2.metric("Vector (Ngữ nghĩa)", f"{item.scores.vector_score:.3f}")
                s3.metric("RRF Fusion", f"{item.scores.rrf_score:.4f}")
                s4.metric("Rerank Score", f"{item.scores.rerank_score:.4f}")
                st.caption(f"**Ngữ cảnh Điều cha (TV2):** {item.chunk.article_title}")
                st.write(item.chunk.content)

# ---------------------------------------------------------------------
# TAB 3: BENCHMARK ARENA (ĐẤU TRƯỜNG ĐÁNH GIÁ 50 CÂU HỎI - TV4, TV6, TV7)
# ---------------------------------------------------------------------
with tab_benchmark:
    st.subheader("Hệ thống Chạy Thực nghiệm & Thống kê Điểm Benchmark Tự động (TV4, TV6 & TV7)")
    st.markdown(
        f"Bộ dữ liệu chuẩn hiện tại tại `evaluation/dataset_50_qa.json`: **`{len(benchmark_runner.questions)}` câu hỏi pháp lý chuẩn** "
        "(TV6 có thể bổ sung đủ 50 câu trực tiếp vào file JSON)."
    )

    b_col1, b_col2 = st.columns([1, 2])
    with b_col1:
        num_q_eval = st.slider(
            "Số lượng câu hỏi chạy kiểm thử ngay:",
            min_value=3,
            max_value=len(benchmark_runner.questions),
            value=min(10, len(benchmark_runner.questions)),
        )
        run_bench_btn = st.button("🚀 Chạy Benchmark trên các Model", type="primary", use_container_width=True)

    if run_bench_btn or "benchmark_summaries" not in st.session_state:
        models_to_compare = [
            LLMModelConfig(provider="gemini", model_name="gemini-4-hydrogen", temperature=0.1),
            LLMModelConfig(provider="groq", model_name="qwen/qwen3.8-27b", temperature=0.1),
            LLMModelConfig(provider="ollama", model_name="gemma-4-26B-A4B-it", temperature=0.1),
        ]
        summaries = []
        for m_cfg in models_to_compare:
            summaries.append(benchmark_runner.evaluate_model(m_cfg, max_questions=num_q_eval))
        st.session_state["benchmark_summaries"] = summaries

    summaries = st.session_state["benchmark_summaries"]

    st.markdown("#### 🏆 Bảng Tổng hợp Kết quả Thực nghiệm (Xuất tự động cho TV7)")
    leaderboard_rows = [
        {
            "Mô hình (Model)": f"{s.model_name} ({s.provider.upper()})",
            "Số câu test": s.total_questions,
            "HitRate@3 (%)": f"{s.hit_rate_at_3}%",
            "MRR (Mean Reciprocal Rank)": s.mrr_score,
            "Độ phủ Ý chuẩn TV6 (%)": f"{s.avg_keyword_coverage}%",
            "Tỷ lệ Chuẩn Trích dẫn (%)": f"{s.citation_verified_rate}%",
            "Độ trễ TB (ms)": f"{s.avg_latency_ms} ms",
        }
        for s in summaries
    ]
    st.table(leaderboard_rows)

    st.markdown("#### 📋 Chi tiết Đối chiếu Từng Câu hỏi (Phục vụ TV6 rà soát)")
    primary_summary = summaries[0]
    detail_rows = [
        {
            "Mã câu": d.question_id,
            "Câu hỏi pháp lý": d.question[:75] + "...",
            "Đáp án Điều luật Chuẩn": d.expected_article,
            "Top-3 Điều Truy xuất": ", ".join(d.retrieved_top_articles),
            "Tìm trúng (Hit@3)": "✅ Đạt" if d.retrieval_hit else "❌ Trượt",
            "Độ phủ từ khóa (%)": f"{d.keyword_coverage}%",
            "Xác thực Trích dẫn": "✅ Chuẩn" if d.citation_verified else "⚠️ Sai",
        }
        for d in primary_summary.details
    ]
    st.dataframe(detail_rows, use_container_width=True)