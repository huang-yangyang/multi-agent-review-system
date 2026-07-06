"""混合检索器：RRF 融合 + CrossEncoder 精排。"""

import logging
from typing import List

logger = logging.getLogger(__name__)

RRF_K = 60
RERANK_TOP_K = 20
RERANK_FINAL_K = 10


def rrf_fuse(
    dense_candidates: List[dict],
    bm25_candidates: List[dict],
    k: int = RRF_K,
) -> List[dict]:
    """RRF 融合两路检索结果，去重后返回统一排名。

    每路候选格式: [{"id": ..., "chunk": ..., "score": ...}, ...]
    """
    def _make_key(item: dict) -> str:
        return item.get("id", "") or item.get("chunk", "")

    rrf_scores = {}

    for rank, item in enumerate(dense_candidates):
        key = _make_key(item)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)

    for rank, item in enumerate(bm25_candidates):
        key = _make_key(item)
        rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank + 1)

    seen = set()
    merged_candidates = []
    for item in dense_candidates + bm25_candidates:
        key = _make_key(item)
        if key not in seen:
            seen.add(key)
            merged_candidates.append(item)

    for item in merged_candidates:
        key = _make_key(item)
        item["rrf_score"] = rrf_scores.get(key, 0)

    merged_candidates.sort(key=lambda x: x["rrf_score"], reverse=True)
    return merged_candidates
