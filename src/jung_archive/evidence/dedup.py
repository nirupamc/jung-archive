"""Deterministic duplicate detection for evidence candidates (M4).

Signals (either is sufficient, thresholds configurable):
  - shared source-block provenance overlap
  - normalized token overlap (containment) between texts

No embedding similarity. Ties are resolved by the caller's incoming
order (reranked relevance first).
"""
from typing import Dict, List, Set, Tuple

from jung_archive.retrieval.lexical import preprocess


def _token_set(text: str) -> Set[str]:
    return set(preprocess(text))


def block_overlap(a_blocks: List[str], b_blocks: List[str]) -> float:
    """Jaccard-style overlap over source block IDs."""
    if not a_blocks or not b_blocks:
        return 0.0
    a, b = set(a_blocks), set(b_blocks)
    inter = a & b
    if not inter:
        return 0.0
    return len(inter) / min(len(a), len(b))


def text_containment(a_text: str, b_text: str) -> float:
    """|A∩B| / min(|A|,|B|) over normalized lexical tokens."""
    ta, tb = _token_set(a_text), _token_set(b_text)
    if not ta or not tb:
        return 0.0
    inter = ta & tb
    return len(inter) / min(len(ta), len(tb))


def is_duplicate(
    cand,
    kept,
    provenance_threshold: float = 0.5,
    text_threshold: float = 0.8,
) -> Tuple[bool, str]:
    """Decide whether `cand` duplicates an already-kept candidate."""
    prov = block_overlap(cand.source_block_ids, kept.source_block_ids)
    if prov >= provenance_threshold:
        return True, f"shared_provenance:{prov:.2f}"
    cont = text_containment(cand.text, kept.text)
    if cont >= text_threshold:
        return True, f"text_overlap:{cont:.2f}"
    # Overlapping pages alone are NOT sufficient (adjacent chunks share
    # a page legitimately); require corroborating textual overlap.
    if set(cand.page_numbers) & set(kept.page_numbers):
        weaker_prov = block_overlap(cand.source_block_ids, kept.source_block_ids)
        weaker_text = text_containment(cand.text, kept.text)
        if weaker_prov >= provenance_threshold * 0.6 and \
                weaker_text >= text_threshold * 0.6:
            return True, f"page_overlap_corroborated:{weaker_prov:.2f}/{weaker_text:.2f}"
    return False, ""


def find_duplicates(
    ranked_candidates,
    provenance_threshold: float = 0.5,
    text_threshold: float = 0.8,
) -> Tuple[List[dict], List[Tuple[object, str]]]:
    """Greedy dedup in given (relevance) order.

    Returns (kept_records, suppressed) where kept_records are dicts:
      {candidate, duplicate_group} and suppressed are
      (candidate, reason) pairs.
    """
    kept: List[dict] = []
    suppressed: List[Tuple[object, str]] = []
    group_of: Dict[str, int] = {}
    next_group = 1

    for cand in ranked_candidates:
        dup_reason = None
        for record in kept:
            is_dup, why = is_duplicate(
                cand, record["candidate"],
                provenance_threshold, text_threshold)
            if is_dup:
                dup_reason = f"duplicate_of:{record['candidate'].chunk_id}:{why}"
                break
        if dup_reason:
            suppressed.append((cand, dup_reason))
            continue
        kept.append({"candidate": cand, "duplicate_group": next_group})
        next_group += 1
    return kept, suppressed
