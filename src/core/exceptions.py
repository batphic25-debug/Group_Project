from __future__ import annotations
from typing import Any, Optional


class BDSLegalRAGError(Exception):
    """Lớp ngoại lệ gốc (Base Exception) cho toàn bộ hệ thống RAG Pháp lý BĐS."""

    def __init__(
        self,
        message: str,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(f"[{self.error_code}] {self.message}")


class ConfigurationError(BDSLegalRAGError):
    """Lỗi thiếu file cấu hình, sai cú pháp YAML hoặc thiếu tham số bắt buộc."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="CONFIG_ERROR", details=details)


class APIKeyExhaustedError(BDSLegalRAGError):
    """Lỗi toàn bộ API Keys của một nhà cung cấp đều bị thiếu hoặc đang chạm giới hạn Rate Limit."""

    def __init__(self, provider: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Tất cả API Keys cho nhà cung cấp '{provider}' đều không khả dụng hoặc đang bị giới hạn tốc độ.",
            error_code="API_KEY_EXHAUSTED",
            details=details,
        )


class ProviderAPIError(BDSLegalRAGError):
    """Lỗi phát sinh từ phía API của Google, Voyage, Cohere hoặc Groq."""

    def __init__(self, provider: str, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            f"Lỗi gọi API [{provider}]: {message}",
            error_code="PROVIDER_API_ERROR",
            details=details,
        )


class DataContractError(BDSLegalRAGError):
    """Lỗi dữ liệu đầu vào từ TV2/TV3 không khớp với cấu trúc LegalChunk chuẩn."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="DATA_CONTRACT_ERROR", details=details)


class CitationHallucinationError(BDSLegalRAGError):
    """Lỗi phát hiện LLM trích dẫn Điều/Khoản luật không tồn tại trong ngữ cảnh truy xuất."""

    def __init__(self, invalid_citations: list[str]) -> None:
        super().__init__(
            f"Phát hiện trích dẫn pháp lý không có căn cứ: {', '.join(invalid_citations)}",
            error_code="CITATION_HALLUCINATION",
            details={"invalid_citations": invalid_citations},
        )