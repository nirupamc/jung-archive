"""
Reciprocal Rank Fusion (M3).

RRF_score(d) = sum over retrievers of 1 / (k + rank_i(d))

Ranks only — raw cosine and BM25 scores are never added or mixed.
Ties are broken deterministically by chunk_id.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from jung_archive.retrieval.dense import DenseRetriever  # noqa: F401 (re-export convenience)
from jung_archive.retrieval.results import RetrievalResult


@dataclass
class FusionCandidate:
    """Accumulates per-leg rank/score info for one chunk."""
    result: RetrievalResult
    contributions: List[Tuple[str, int]] = field(default_factory=list)

    def rrf_score(self, k: int) -> float:
        return sum(1.0 / (k + rank) for _, rank in self.contributions)


def fuse_rankings(
    ranked_lists: List[List[RetrievalResult]],
    rrf_k: int,
) -> List[RetrievalResult]:
    """Fuse multiple ranked result lists into one RRF-ordered list.

    - a chunk present in several lists merges into ONE result carrying
      every leg's rank/score
    - chunks present in only one list keep None for the other leg
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be >= 1")

    by_id: Dict[str, FusionCandidate] = {}

    for leg_name, ranking in zip(["dense", "bm25"], ranked_lists):
        for position, res in enumerate(ranking, start=1):
            existing = by_id.get(res.chunk_id)
            if existing is None:
                merged = _merge_into(res, None)
                by_id[res.chunk_id] = FusionCandidate(result=merged,
                                                      contributions=[(leg_name, position)])
            else:
                _merge_into(res, existing.result)
                existing.contributions.append((leg_name, position))

    fused = []
    for cand in by_id.values():
        res = cand.result
        score = cand.rrf_score(rrf_k)
        res.fusion_score = round(score, 6)
        for leg, pos in cand.contributions:
            if leg == "dense":
                res.dense_rank = pos
            elif leg == "bm25":
                res.bm25_rank = pos
        fused.append(res)

    # Deterministic order: fusion score desc, then chunk_id asc
    fused.sort(key=lambda r: (-r.fusion_score, r.chunk_id))
    for i, res in enumerate(fused, start=1):
        res.fusion_rank = i
    return fused


def _merge_into(incoming: RetrievalResult, target: Optional[RetrievalResult]) -> RetrievalResult:
    """Copy incoming leg scores/provenance into target (or return copy)."""
    t = target if target is not None else incoming.model_copy()
    # Provenance must agree; prefer the richer text/fields from whichever
    # leg provided them first.
    if target is not None:
        if incoming.dense_score is not None:
            t.dense_score = incoming.dense_score
            t.dense_rank = incoming.dense_rank
        if incoming.bm25_score is not None:
            t.bm25_score = incoming.bm25_score
            t.bm25_rank = incoming.bm25_rank
        # provenance sanity: identical chunk ids imply same source
        assert t.document_id == incoming.document_id
    return t
