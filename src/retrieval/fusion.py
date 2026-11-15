from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from src.core.schemas import LegalChunk, RetrievalScoreDetail, RetrievedChunk


def reciprocal_rank_fusion(
    bm25_results: Sequence[Tuple[LegalChunk, float]],
    vector_results: Sequence[Tuple[LegalChunk, float]],
    rrf_k: int = 60,
    top_candidates: int = 25,
) -> List[RetrievedChunk]:
    """
    Hợp nhất kết quả từ BM25 và Vector Search bằng công thức Reciprocal Rank Fusion (RRF):
    RRF_Score(d) = 1 / (k + rank_bm25(d)) + 1 / (k + rank_vector(d))
    """
    chunk_map: Dict[str, LegalChunk] = {}
    bm25_scores: Dict[str, float] = {}
    vec_scores: Dict[str, float] = {}
    rrf_scores: Dict[str, float] = {}

    for rank_idx, (chunk, score) in enumerate(bm25_results, start=1):
        cid = chunk.chunk_id
        chunk_map[cid] = chunk
        bm25_scores[cid] = score
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank_idx))

    for rank_idx, (chunk, score) in enumerate(vector_results, start=1):
        cid = chunk.chunk_id
        chunk_map[cid] = chunk
        vec_scores[cid] = score
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (rrf_k + rank_idx))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)
    fused_list: List[RetrievedChunk] = []

    for idx, cid in enumerate(sorted_ids[:top_candidates], start=1):
        detail = RetrievalScoreDetail(
            bm25_score=round(bm25_scores.get(cid, 0.0), 4),
            vector_score=round(vec_scores.get(cid, 0.0), 4),
            rrf_score=round(rrf_scores[cid], 6),
            rerank_score=0.0,
        )
        fused_list.append(
            RetrievedChunk(
                chunk=chunk_map[cid],
                scores=detail,
                rank=idx,
            )
        )

    return fused_list