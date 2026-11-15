from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError

from src.core.exceptions import ConfigurationError

# Xác định thư mục gốc dự án (tính từ src/core/config.py ngược lên 2 cấp)
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


class ProjectConfig(BaseModel):
    name: str
    version: str
    environment: Literal["development", "demo", "benchmark"] = "development"


class PathsConfig(BaseModel):
    raw_laws_dir: Path
    sample_contracts_dir: Path
    knowledge_base_file: Path
    vector_index_dir: Path
    cache_dir: Path
    benchmark_dataset_file: Path

    def resolve_all(self, root: Path) -> "PathsConfig":
        """Chuyển đổi toàn bộ đường dẫn tương đối thành đường dẫn tuyệt đối chuẩn OS."""
        return PathsConfig(
            raw_laws_dir=(root / self.raw_laws_dir).resolve(),
            sample_contracts_dir=(root / self.sample_contracts_dir).resolve(),
            knowledge_base_file=(root / self.knowledge_base_file).resolve(),
            vector_index_dir=(root / self.vector_index_dir).resolve(),
            cache_dir=(root / self.cache_dir).resolve(),
            benchmark_dataset_file=(root / self.benchmark_dataset_file).resolve(),
        )


class EmbeddingModelConfig(BaseModel):
    provider: Literal["voyage", "cohere", "gemini"]
    model_name: str
    dimensions: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=32, gt=0)


class RerankerModelConfig(BaseModel):
    provider: Literal["voyage", "cohere"]
    model_name: str
    top_n: int = Field(default=5, gt=0)
    score_threshold: float = Field(default=0.35, ge=0.0, le=1.0)


class LLMModelConfig(BaseModel):
    provider: Literal["gemini", "groq", "openrouter", "ollama"]
    model_name: str
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    max_tokens: int = Field(default=4096, gt=0)


class ModelsConfig(BaseModel):
    embedding: EmbeddingModelConfig
    reranker: RerankerModelConfig
    llm_primary: LLMModelConfig
    llm_secondary: LLMModelConfig


class RetrievalConfig(BaseModel):
    vector_top_k: int = Field(default=20, gt=0)
    bm25_top_k: int = Field(default=20, gt=0)
    rrf_k: int = Field(default=60, gt=0)
    hybrid_candidates: int = Field(default=25, gt=0)
    enable_cache: bool = True


class GuardrailsConfig(BaseModel):
    strict_citation_check: bool = True
    min_confidence_score: float = Field(default=0.40, ge=0.0, le=1.0)


class AppSettings(BaseModel):
    project: ProjectConfig
    paths: PathsConfig
    models: ModelsConfig
    retrieval: RetrievalConfig
    guardrails: GuardrailsConfig


@lru_cache(maxsize=1)
def get_settings(config_path: Path | None = None) -> AppSettings:
    """Nạp biến môi trường (.env) và file cấu hình YAML (có cache singleton)."""
    # Nạp file .env tại thư mục gốc hoặc configs/.env
    load_dotenv(PROJECT_ROOT / ".env")
    load_dotenv(PROJECT_ROOT / "configs" / ".env")

    target_yaml = config_path or (PROJECT_ROOT / "configs" / "settings.yaml")
    if not target_yaml.exists():
        raise ConfigurationError(f"Không tìm thấy file cấu hình tại: {target_yaml}")

    try:
        with target_yaml.open("r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f) or {}

        settings = AppSettings(**raw_data)
        settings.paths = settings.paths.resolve_all(PROJECT_ROOT)
        return settings
    except ValidationError as exc:
        raise ConfigurationError(
            "Cấu trúc file settings.yaml không hợp lệ.",
            details={"validation_errors": exc.errors()},
        ) from exc
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Lỗi cú pháp YAML trong {target_yaml}: {exc}") from exc