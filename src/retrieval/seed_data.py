from __future__ import annotations

import json
from pathlib import Path
from typing import List

from src.core.config import get_settings
from src.core.schemas import LegalChunk

GOLDEN_LEGAL_CHUNKS: List[dict] = [
    {
        "chunk_id": "LKDBDS2023_D23_K5",
        "law_name": "Luật Kinh doanh bất động sản 2023",
        "law_code": "29/2023/QH15",
        "chapter": "Chương III",
        "article_number": "Điều 23",
        "article_title": "Nguyên tắc kinh doanh nhà ở, công trình xây dựng hình thành trong tương lai",
        "clause_number": "Khoản 5",
        "content": (
            "Chủ đầu tư dự án bất động sản chỉ được thu tiền đặt cọc không quá 05% giá bán, cho thuê mua "
            "nhà ở, công trình xây dựng, phần diện tích sàn xây dựng trong công trình xây dựng từ bên đặt cọc "
            "để mua, thuê mua khi nhà ở, công trình xây dựng đã có đủ các điều kiện đưa vào kinh doanh theo "
            "quy định của Luật này. Thỏa thuận đặt cọc phải ghi rõ giá bán, cho thuê mua nhà ở, công trình xây dựng."
        ),
        "effective_date": "2024-08-01",
    },
    {
        "chunk_id": "LKDBDS2023_D24_K1",
        "law_name": "Luật Kinh doanh bất động sản 2023",
        "law_code": "29/2023/QH15",
        "chapter": "Chương III",
        "article_number": "Điều 24",
        "article_title": "Điều kiện của nhà ở, công trình xây dựng hình thành trong tương lai được đưa vào kinh doanh",
        "clause_number": "Khoản 1",
        "content": (
            "Nhà ở, công trình xây dựng hình thành trong tương lai phải đã được khởi công xây dựng theo quy định "
            "của pháp luật về xây dựng và có giấy tờ nghiệm thu việc đã hoàn thành xây dựng cơ sở hạ tầng kỹ thuật "
            "tương ứng theo tiến độ dự án; trường hợp là nhà chung cư, tòa nhà hỗn hợp có nhà ở phải có giấy tờ "
            "nghiệm thu đã hoàn thành xây dựng xong phần móng."
        ),
        "effective_date": "2024-08-01",
    },
    {
        "chunk_id": "LKDBDS2023_D25_K1",
        "law_name": "Luật Kinh doanh bất động sản 2023",
        "law_code": "29/2023/QH15",
        "chapter": "Chương III",
        "article_number": "Điều 25",
        "article_title": "Thanh toán trong mua bán, thuê mua nhà ở, công trình xây dựng hình thành trong tương lai",
        "clause_number": "Khoản 1",
        "content": (
            "Việc thanh toán trong mua bán nhà ở, công trình xây dựng hình thành trong tương lai được thực hiện nhiều lần, "
            "lần đầu không quá 30% giá trị hợp đồng bao gồm cả tiền đặt cọc, những lần tiếp theo phải phù hợp với tiến độ "
            "xây dựng nhưng tổng số không quá 70% giá trị hợp đồng khi chưa bàn giao nhà ở, công trình xây dựng cho bên mua."
        ),
        "effective_date": "2024-08-01",
    },
    {
        "chunk_id": "LKDBDS2023_D25_K3",
        "law_name": "Luật Kinh doanh bất động sản 2023",
        "law_code": "29/2023/QH15",
        "chapter": "Chương III",
        "article_number": "Điều 25",
        "article_title": "Thanh toán trong mua bán, thuê mua nhà ở, công trình xây dựng hình thành trong tương lai",
        "clause_number": "Khoản 3",
        "content": (
            "Nếu bên mua, thuê mua chưa được cấp giấy chứng nhận về quyền sử dụng đất, quyền sở hữu tài sản gắn liền "
            "với đất theo quy định của pháp luật về đất đai thì bên bán, cho thuê mua không được thu quá 95% giá trị "
            "hợp đồng; giá trị còn lại của hợp đồng được thanh toán khi cơ quan nhà nước có thẩm quyền đã cấp giấy chứng nhận."
        ),
        "effective_date": "2024-08-01",
    },
    {
        "chunk_id": "LKDBDS2023_D26_K1",
        "law_name": "Luật Kinh doanh bất động sản 2023",
        "law_code": "29/2023/QH15",
        "chapter": "Chương III",
        "article_number": "Điều 26",
        "article_title": "Bảo lãnh trong bán, cho thuê mua nhà ở hình thành trong tương lai",
        "clause_number": "Khoản 1",
        "content": (
            "Chủ đầu tư dự án bất động sản trước khi bán, cho thuê mua nhà ở hình thành trong tương lai phải được "
            "ngân hàng thương mại trong nước hoặc chi nhánh ngân hàng nước ngoài đang hoạt động hợp pháp tại Việt Nam "
            "chấp thuận cấp bảo lãnh cho nghĩa vụ tài chính của chủ đầu tư đối với bên mua, thuê mua nhà ở khi chủ đầu tư "
            "không bàn giao nhà ở theo cam kết trong hợp đồng."
        ),
        "effective_date": "2024-08-01",
    },
    {
        "chunk_id": "LKDBDS2023_D44_K4",
        "law_name": "Luật Kinh doanh bất động sản 2023",
        "law_code": "29/2023/QH15",
        "chapter": "Chương V",
        "article_number": "Điều 44",
        "article_title": "Hợp đồng trong kinh doanh bất động sản",
        "clause_number": "Khoản 4",
        "content": (
            "Hợp đồng mua bán, thuê mua nhà ở, công trình xây dựng, phần diện tích sàn xây dựng trong công trình xây dựng "
            "mà các bên tham gia giao dịch là cá nhân phải công chứng hoặc chứng thực."
        ),
        "effective_date": "2024-08-01",
    },
    {
        "chunk_id": "LKDBDS2023_D47_K1",
        "law_name": "Luật Kinh doanh bất động sản 2023",
        "law_code": "29/2023/QH15",
        "chapter": "Chương V",
        "article_number": "Điều 47",
        "article_title": "Thanh toán trong kinh doanh bất động sản",
        "clause_number": "Khoản 1",
        "content": (
            "Chủ đầu tư dự án, doanh nghiệp kinh doanh bất động sản, doanh nghiệp kinh doanh dịch vụ bất động sản "
            "nhận tiền thanh toán theo hợp đồng kinh doanh bất động sản, hợp đồng kinh doanh dịch vụ bất động sản "
            "từ khách hàng thông qua tài khoản mở tại tổ chức tín dụng trong nước hoặc chi nhánh ngân hàng nước ngoài "
            "đang hoạt động hợp pháp tại Việt Nam."
        ),
        "effective_date": "2024-08-01",
    },
    {
        "chunk_id": "LNO2023_D164_K1",
        "law_name": "Luật Nhà ở 2023",
        "law_code": "27/2023/QH15",
        "chapter": "Chương IX",
        "article_number": "Điều 164",
        "article_title": "Công chứng, chứng thực hợp đồng và thời điểm có hiệu lực của hợp đồng về nhà ở",
        "clause_number": "Khoản 1",
        "content": (
            "Trường hợp mua bán, thuê mua, tặng cho, đổi, góp vốn, thế chấp nhà ở thì phải thực hiện công chứng hoặc "
            "chứng thực hợp đồng, trừ trường hợp quy định tại khoản 2 Điều này. Thời điểm có hiệu lực của hợp đồng là "
            "thời điểm hoàn thành việc công chứng hoặc chứng thực theo quy định của pháp luật về công chứng, chứng thực."
        ),
        "effective_date": "2024-08-01",
    },
    {
        "chunk_id": "LDD2024_D45_K1",
        "law_name": "Luật Đất đai 2024",
        "law_code": "31/2024/QH15",
        "chapter": "Chương III",
        "article_number": "Điều 45",
        "article_title": "Điều kiện thực hiện các quyền chuyển đổi, chuyển nhượng, cho thuê, cho thuê lại, thừa kế, tặng cho quyền sử dụng đất",
        "clause_number": "Khoản 1",
        "content": (
            "Người sử dụng đất được thực hiện các quyền chuyển nhượng, tặng cho, thế chấp quyền sử dụng đất khi có đủ "
            "các điều kiện sau đây: a) Có Giấy chứng nhận quyền sử dụng đất; b) Đất không có tranh chấp hoặc tranh chấp "
            "đã được giải quyết bởi cơ quan nhà nước có thẩm quyền; c) Quyền sử dụng đất không bị kê biên, áp dụng biện pháp "
            "khẩn cấp tạm thời để bảo đảm thi hành án; d) Trong thời hạn sử dụng đất."
        ),
        "effective_date": "2024-08-01",
    },
    {
        "chunk_id": "BLDS2015_D328_K1",
        "law_name": "Bộ luật Dân sự 2015",
        "law_code": "91/2015/QH13",
        "chapter": "Chương XV",
        "article_number": "Điều 328",
        "article_title": "Đặt cọc",
        "clause_number": "Khoản 1",
        "content": (
            "Đặt cọc là việc một bên (sau đây gọi là bên đặt cọc) giao cho bên kia (sau đây gọi là bên nhận đặt cọc) "
            "một khoản tiền hoặc kim khí quý, đá quý hoặc vật có giá trị khác trong một thời hạn để bảo đảm giao kết "
            "hoặc thực hiện hợp đồng."
        ),
        "effective_date": "2017-01-01",
    },
    {
        "chunk_id": "BLDS2015_D328_K2",
        "law_name": "Bộ luật Dân sự 2015",
        "law_code": "91/2015/QH13",
        "chapter": "Chương XV",
        "article_number": "Điều 328",
        "article_title": "Đặt cọc",
        "clause_number": "Khoản 2",
        "content": (
            "Trường hợp hợp đồng được giao kết, thực hiện thì tài sản đặt cọc được trả lại cho bên đặt cọc hoặc được trừ "
            "để thực hiện nghĩa vụ trả tiền; nếu bên đặt cọc từ chối việc giao kết, thực hiện hợp đồng thì tài sản đặt cọc "
            "thuộc về bên nhận đặt cọc; nếu bên nhận đặt cọc từ chối việc giao kết, thực hiện hợp đồng thì phải trả cho "
            "bên đặt cọc tài sản đặt cọc và một khoản tiền tương đương giá trị tài sản đặt cọc, trừ trường hợp có thỏa thuận khác."
        ),
        "effective_date": "2017-01-01",
    },
    {
        "chunk_id": "BLDS2015_D418_K1",
        "law_name": "Bộ luật Dân sự 2015",
        "law_code": "91/2015/QH13",
        "chapter": "Chương XV",
        "article_number": "Điều 418",
        "article_title": "Thỏa thuận phạt vi phạm",
        "clause_number": "Khoản 1",
        "content": (
            "Phạt vi phạm là sự thỏa thuận giữa các bên trong hợp đồng, theo đó bên vi phạm nghĩa vụ phải nộp một khoản tiền "
            "cho bên bị vi phạm. Mức phạt vi phạm do các bên thỏa thuận, trừ trường hợp luật liên quan có quy định khác."
        ),
        "effective_date": "2017-01-01",
    },
]


SAMPLE_CONTRACT_DEMO_TEXT = """BIÊN BẢN ĐẶT CỌC VÀ THỎA THUẬN MUA BÁN CĂN HỘ HÌNH THÀNH TRONG TƯƠNG LAI
Số: 15/2026/BBDC-BĐS

Điều 1. Đối tượng giao dịch và Hiện trạng pháp lý
Bên A (Chủ đầu tư Công ty Cổ phần BĐS Hòa Lạc) đồng ý bán cho Bên B (Ông Nguyễn Văn An) căn hộ chung cư mã A12-08 thuộc Dự án Chung cư Cao cấp Hòa Lạc Skyview (loại hình nhà ở hình thành trong tương lai). Hiện tại dự án đang trong giai đoạn san lấp mặt bằng, chuẩn bị ép cọc thử tĩnh và chưa có biên bản nghiệm thu hoàn thành phần móng của tòa nhà.

Điều 2. Giá bán và Khoản tiền Đặt cọc
Tổng giá bán căn hộ tạm tính là 4.000.000.000 VNĐ (Bốn tỷ đồng). Ngay tại thời điểm ký biên bản đặt cọc này, Bên B phải nộp ngay khoản tiền đặt cọc tương đương 25% tổng giá bán căn hộ (tức 1.000.000.000 VNĐ) bằng tiền mặt trực tiếp tại phòng kế toán của Chủ đầu tư.

Điều 3. Tiến độ thanh toán trước khi bàn giao nhà
Bên B có nghĩa vụ thanh toán 40% giá trị hợp đồng ngay trong đợt đầu tiên (bao gồm tiền cọc), và thanh toán lũy kế lên tới 85% tổng giá trị hợp đồng trước thời điểm Bên A thông báo bàn giao căn hộ. Toàn bộ 100% giá trị hợp đồng phải được tất toán ngay khi nhận bàn giao chìa khóa, kể cả khi cơ quan nhà nước chưa cấp Giấy chứng nhận quyền sở hữu nhà ở (Sổ hồng).

Điều 4. Bảo lãnh Ngân hàng và Hiệu lực Văn bản
Để tiết kiệm chi phí tài chính cho dự án, hai bên thống nhất miễn trừ thủ tục bảo lãnh ngân hàng thương mại đối với nghĩa vụ bàn giao nhà ở của Chủ đầu tư. Hợp đồng mua bán chính thức sau này giữa các cá nhân hoặc phụ lục chuyển nhượng chỉ cần ký viết tay có chữ ký của hai bên là phát sinh hiệu lực pháp lý đầy đủ, không cần qua thủ tục công chứng hoặc chứng thực.
"""


def ensure_seed_data() -> List[LegalChunk]:
    """
    Đảm bảo file dữ liệu Luật (knowledge_base) và Hợp đồng mẫu (sample_contracts) luôn sẵn sàng.
    Nếu file chưa tồn tại, tự động ghi bộ dữ liệu mồi chuẩn vào ổ cứng.
    """
    settings = get_settings()
    kb_path: Path = settings.paths.knowledge_base_file
    kb_path.parent.mkdir(parents=True, exist_ok=True)

    if not kb_path.exists():
        with kb_path.open("w", encoding="utf-8") as f:
            json.dump(GOLDEN_LEGAL_CHUNKS, f, ensure_ascii=False, indent=2)

    contract_dir: Path = settings.paths.sample_contracts_dir
    contract_dir.mkdir(parents=True, exist_ok=True)
    demo_contract_path = contract_dir / "hop_dong_dat_coc_demo_tv1.txt"
    if not demo_contract_path.exists():
        demo_contract_path.write_text(SAMPLE_CONTRACT_DEMO_TEXT, encoding="utf-8")

    with kb_path.open("r", encoding="utf-8") as f:
        raw_list = json.load(f)

    return [LegalChunk.model_validate(item) for item in raw_list]