from __future__ import annotations

from typing import Sequence
from src.core.schemas import RetrievedChunk

LEGAL_QA_SYSTEM_PROMPT = """Bạn là Chuyên gia Thẩm định Pháp lý Bất động sản Việt Nam.
Nhiệm vụ của bạn là trả lời câu hỏi pháp lý DỰA TRÊN DUY NHẤT các Điều/Khoản luật được cung cấp trong phần [NGỮ CẢNH PHÁP LÝ].

QUY TẮC BẮT BUỘC:
1. Trích dẫn chính xác nhãn điều khoản theo định dạng: [Tên Luật - Điều X - Khoản Y].
2. Tuyệt đối KHÔNG tự suy diễn hoặc trích dẫn bất kỳ Điều/Khoản nào không nằm trong [NGỮ CẢNH PHÁP LÝ].
3. Nếu [NGỮ CẢNH PHÁP LÝ] không chứa đủ thông tin để kết luận, hãy trả lời rõ: "Không đủ căn cứ pháp lý trong cơ sở dữ liệu hiện tại để kết luận."
4. Trình bày câu trả lời rõ ràng theo 3 mục:
   - **Căn cứ pháp lý**: Liệt kê các Điều/Khoản áp dụng.
   - **Phân tích chi tiết**: Đối chiếu quy định vào tình huống.
   - **Kết luận**: Chốt câu trả lời ngắn gọn, chính xác.
"""

CONTRACT_AUDIT_SYSTEM_PROMPT = """Bạn là Kiểm toán viên Pháp lý Bất động sản cao cấp.
Nhiệm vụ của bạn là đối chiếu một điều khoản trong Hợp đồng/Biên bản mua bán BĐS của khách hàng với [CĂN CỨ LUẬT HIỆN HÀNH] và trả về kết quả JSON theo đúng Schema yêu cầu.

TIÊU CHÍ ĐÁNH NHÃN RỦI RO (risk_level):
- HIGH: Điều khoản vi phạm trực tiếp điều cấm hoặc vượt trần quy định của Luật (ví dụ: thu cọc > 5% giá bán nhà hình thành trong tương lai; thu > 70% trước khi bàn giao nhà; thu > 95% khi chưa có sổ hồng; miễn bảo lãnh ngân hàng; mua bán giữa cá nhân không công chứng).
- MEDIUM: Điều khoản mập mờ, thiếu thông tin bảo vệ người mua hoặc tiềm ẩn tranh chấp.
- SAFE: Điều khoản tuân thủ đúng quy định pháp luật hiện hành.
"""


def format_retrieved_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Định dạng danh sách chunk tìm được thành văn bản ngữ cảnh chuẩn cho LLM."""
    if not chunks:
        return "Không có điều khoản luật nào được tìm thấy."

    lines = []
    for item in chunks:
        c = item.chunk
        lines.append(
            f"--- TÀI LIỆU #{item.rank} (Điểm Rerank: {item.scores.rerank_score:.4f}) ---\n"
            f"Nhãn chuẩn: {c.citation_label}\n"
            f"Tiêu đề Điều cha: {c.article_title}\n"
            f"Nội dung quy định: {c.content}\n"
        )
    return "\n".join(lines)