from __future__ import annotations

import os
import threading
import time
from typing import Dict, List

from src.core.exceptions import APIKeyExhaustedError


class APIKeyRotator:
    """
    Quản lý danh sách API Keys của nhiều thành viên theo cơ chế Round-Robin (Thread-safe).
    Hỗ trợ tự động tạm khóa (cooldown) các key bị dính Rate Limit (HTTP 429).
    """

    def __init__(self, provider_name: str, env_var_name: str) -> None:
        self.provider_name = provider_name
        self.env_var_name = env_var_name
        self._lock = threading.Lock()

        raw_keys = os.getenv(env_var_name, "")
        self._keys: List[str] = [
            k.strip() for k in raw_keys.split(",") if k.strip() and not k.strip().startswith("your_")
        ]
        self._index: int = 0
        # Lưu mốc thời gian (timestamp) mà một key được phép hoạt động trở lại
        self._cooldown_until: Dict[str, float] = {}

    @property
    def total_keys(self) -> int:
        return len(self._keys)

    def get_next_key(self) -> str:
        """Lấy API Key khả dụng tiếp theo. Ném APIKeyExhaustedError nếu không còn key nào dùng được."""
        with self._lock:
            if not self._keys:
                raise APIKeyExhaustedError(
                    provider=self.provider_name,
                    details={"env_var": self.env_var_name, "reason": "Chưa cấu hình key trong file .env"},
                )

            now = time.monotonic()
            num_keys = len(self._keys)

            # Duyệt tối đa 1 vòng qua tất cả các key để tìm key không bị cooldown
            for _ in range(num_keys):
                candidate = self._keys[self._index]
                self._index = (self._index + 1) % num_keys

                unblock_time = self._cooldown_until.get(candidate, 0.0)
                if now >= unblock_time:
                    # Xóa trạng thái cooldown nếu đã hết hạn
                    self._cooldown_until.pop(candidate, None)
                    return candidate

            # Tính thời gian chờ ngắn nhất nếu tất cả key đều đang bị rate limit
            min_wait = min(self._cooldown_until.values()) - now
            raise APIKeyExhaustedError(
                provider=self.provider_name,
                details={
                    "total_keys": num_keys,
                    "retry_after_seconds": round(max(min_wait, 1.0), 1),
                },
            )

    def mark_rate_limited(self, api_key: str, cooldown_seconds: float = 60.0) -> None:
        """Đánh dấu một API Key vừa bị lỗi 429 để tạm tránh sử dụng trong `cooldown_seconds` giây."""
        with self._lock:
            if api_key in self._keys:
                self._cooldown_until[api_key] = time.monotonic() + cooldown_seconds

    def get_health_summary(self) -> dict[str, int]:
        """Trả về thống kê số lượng key đang sẵn sàng và số key đang tạm nghỉ."""
        with self._lock:
            now = time.monotonic()
            cooling = sum(1 for t in self._cooldown_until.values() if t > now)
            return {
                "total": len(self._keys),
                "active": len(self._keys) - cooling,
                "cooling_down": cooling,
            }


# Registry lưu trữ các bộ xoay vòng dùng chung toàn ứng dụng
_ROTATORS: Dict[str, APIKeyRotator] = {}
_REGISTRY_LOCK = threading.Lock()

PROVIDER_ENV_MAP: Dict[str, str] = {
    "gemini": "GEMINI_API_KEYS",
    "voyage": "VOYAGE_API_KEYS",
    "cohere": "COHERE_API_KEYS",
    "groq": "GROQ_API_KEYS",
    "openrouter": "OPENROUTER_API_KEYS",
}


def get_key_rotator(provider: str) -> APIKeyRotator:
    """Lấy (hoặc khởi tạo) bộ xoay vòng API Key cho nhà cung cấp tương ứng."""
    normalized = provider.lower().strip()
    if normalized not in PROVIDER_ENV_MAP:
        raise ValueError(f"Nhà cung cấp '{provider}' chưa được hỗ trợ xoay vòng key.")

    with _REGISTRY_LOCK:
        if normalized not in _ROTATORS:
            _ROTATORS[normalized] = APIKeyRotator(
                provider_name=normalized,
                env_var_name=PROVIDER_ENV_MAP[normalized],
            )
        return _ROTATORS[normalized]