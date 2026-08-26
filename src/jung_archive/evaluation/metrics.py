"""IR metrics for M6 evaluation.

Hand-implemented (no opaque metric libraries). Binary relevance unless
graded labels are explicitly provided in the dataset.

Documented convention decisions:
  - Precision@K uses denominator min(K, number of results actually
    returned), so systems returning fewer than K results are not
    silently rewarded or punished beyond their returned set.
  - NDCG@K uses binary gains with DCG = sum rel_i / log2(i + 2),
    IDCG computed from min(|relevant|, K) ideal hits.
"""


def _norm(ranked, relevant):
    rel = set(relevant)
    return list(ranked), rel


def hit_at_k(ranked, relevant, k: int) -> float:
    """1.0 if any relevant item appears in the top k, else 0.0."""
    if k < 1:
        raise ValueError("k must be >= 1")
    ranked, rel = _norm(ranked, relevant)
    if not rel:
        return 0.0
    return 1.0 if rel & set(ranked[:k]) else 0.0


def recall_at_k(ranked, relevant, k: int) -> float:
    """|relevant ∩ top-k| / |relevant|."""
    if k < 1:
        raise ValueError("k must be >= 1")
    ranked, rel = _norm(ranked, relevant)
    if not rel:
        return 0.0
    found = rel & set(ranked[:k])
    return len(found) / len(rel)


def precision_at_k(ranked, relevant, k: int) -> float:
    """|relevant ∩ top-k| / min(k, len(ranked))  (>=1 denominator)."""
    if k < 1:
        raise ValueError("k must be >= 1")
    ranked, rel = _norm(ranked, relevant)
    if not ranked or not rel:
        return 0.0
    found = rel & set(ranked[:k])
    denom = max(1, min(k, len(ranked)))
    return len(found) / denom


def reciprocal_rank(ranked, relevant) -> float:
    """1 / rank of first relevant result; 0.0 when none retrieved."""
    ranked, rel = _norm(ranked, relevant)
    if not rel:
        return 0.0
    for i, item in enumerate(ranked, start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked, relevant, k: int) -> float:
    """Binary-relevance NDCG@K."""
    if k < 1:
        raise ValueError("k must be >= 1")
    ranked, rel = _norm(ranked, relevant)
    if not rel:
        return 0.0

    def dcg(items):
        return sum(
            1.0 / _log2(i + 2)
            for i, item in enumerate(items[:k])
            if item in rel
        )

    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / _log2(i + 2) for i in range(ideal_hits))
    if idcg == 0.0:
        return 0.0
    return dcg(ranked) / idcg


def _log2(n: int) -> float:
    from math import log2

    return log2(n)


def first_relevant_rank(ranked, relevant):
    """1-based rank of the first relevant item, or None."""
    ranked, rel = _norm(ranked, relevant)
    for i, item in enumerate(ranked, start=1):
        if item in rel:
            return i
    return None
