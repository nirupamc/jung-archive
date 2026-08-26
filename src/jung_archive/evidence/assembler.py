"""Deterministic evidence assembly (M4).

Ranked reranked candidates -> cleanup -> dedup -> diversity caps ->
token budget -> stable S1..Sn identifiers -> EvidencePack.

Ordering is fully deterministic: reranker score desc, then fusion rank
asc, then chunk_id asc. The canonical chunks are never mutated; only
derived clean_text is produced.
"""
from dataclasses import dataclass
from typing import List, Optional

from jung_archive.chunking.tokenizer import count_tokens
from jung_archive.evidence.cleanup import clean_evidence_text
from jung_archive.evidence.dedup import find_duplicates
from jung_archive.evidence.models import (
    EvidenceItem,
    EvidencePack,
    ScorePath,
    SuppressedItem,
)
from jung_archive.models.document import SourceType


@dataclass
class EvidenceConfig:
    max_evidence_tokens: int = 2500
    max_evidence_items: int = 8
    dedup_provenance_threshold: float = 0.5
    dedup_text_threshold: float = 0.8
    # Conservative diversity defaults; multi-document safe.
    max_chunks_per_document: int = 8
    max_chunks_per_section: int = 4
    max_chunks_per_page_region: int = 3
    page_region_size: int = 8   # pages grouped into regions of this size

    def __post_init__(self):
        if self.max_evidence_tokens < 1:
            raise ValueError("max_evidence_tokens must be >= 1")
        if self.max_evidence_items < 1:
            raise ValueError("max_evidence_items must be >= 1")
        if self.page_region_size < 1:
            raise ValueError("page_region_size must be >= 1")
        for name in ("max_chunks_per_document", "max_chunks_per_section",
                     "max_chunks_per_page_region"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")


def _relevance_key(res):
    """Deterministic relevance order: reranker, then fusion, then id."""
    rr = res.reranker_score if res.reranker_score is not None else float("-inf")
    fr = res.fusion_rank if res.fusion_rank is not None else 10**9
    return (-rr, fr, res.chunk_id)


def _page_region(page_numbers: List[int], size: int) -> int:
    return min(page_numbers) // size


class EvidenceAssembler:
    def __init__(self, config: Optional[EvidenceConfig] = None):
        self.config = config or EvidenceConfig()

    # ------------------------------------------------------------------
    def assemble(self, question: str,
                 candidates: List) -> EvidencePack:
        cfg = self.config
        pack = EvidencePack(
            question=question,
            max_evidence_tokens=cfg.max_evidence_tokens,
            max_evidence_items=cfg.max_evidence_items,
            candidates_considered=len(candidates),
        )
        if not candidates:
            pack.warnings.append("no candidates provided")
            return pack

        ranked = sorted(candidates, key=_relevance_key)

        kept_records, dup_suppressed = find_duplicates(
            ranked,
            provenance_threshold=cfg.dedup_provenance_threshold,
            text_threshold=cfg.dedup_text_threshold,
        )
        group_by_chunk = {r["candidate"].chunk_id: r["duplicate_group"]
                          for r in kept_records}
        pack.suppressed_duplicates = [
            SuppressedItem(chunk_id=c.chunk_id, reason=reason)
            for c, reason in dup_suppressed
        ]

        # Diversity pass (conservative: counts per region/section/doc).
        doc_counts, sec_counts, reg_counts = {}, {}, {}
        diverse: List[dict] = []
        for record in kept_records:
            cand = record["candidate"]
            d = cand.document_id
            s = cand.section_id or "none"
            r = _page_region(cand.page_numbers, cfg.page_region_size)
            if doc_counts.get(d, 0) >= cfg.max_chunks_per_document:
                pack.suppressed_diversity.append(SuppressedItem(
                    chunk_id=cand.chunk_id, reason="max_chunks_per_document"))
                continue
            if sec_counts.get((d, s), 0) >= cfg.max_chunks_per_section:
                pack.suppressed_diversity.append(SuppressedItem(
                    chunk_id=cand.chunk_id, reason="max_chunks_per_section"))
                continue
            if reg_counts.get((d, r), 0) >= cfg.max_chunks_per_page_region:
                pack.suppressed_diversity.append(SuppressedItem(
                    chunk_id=cand.chunk_id, reason="max_chunks_per_page_region"))
                continue
            diverse.append(record)
            doc_counts[d] = doc_counts.get(d, 0) + 1
            sec_counts[(d, s)] = sec_counts.get((d, s), 0) + 1
            reg_counts[(d, r)] = reg_counts.get((d, r), 0) + 1

        # Token-budget selection following reranked relevance.
        tokens_used = 0
        selected: List[EvidenceItem] = []
        oversized_seen = False
        for i, record in enumerate(diverse):
            cand = record["candidate"]
            if len(selected) >= cfg.max_evidence_items:
                break
            cleanup = clean_evidence_text(
                cand.text,
                title=getattr(cand, "title", None) or "",
                heading_path=cand.heading_path,
            )
            token_count = count_tokens(cleanup.clean_text)
            if token_count > cfg.max_evidence_tokens:
                # Explicit handling: never silently exceed the budget.
                pack.skipped_oversized.append(SuppressedItem(
                    chunk_id=cand.chunk_id,
                    reason=f"oversized:{token_count}>{cfg.max_evidence_tokens}"))
                oversized_seen = True
                continue
            if tokens_used + token_count > cfg.max_evidence_tokens:
                # Budget exhausted; stop scanning (later items are less
                # relevant by construction).
                pack.warnings.append(
                    f"evidence token budget reached after {len(selected)} "
                    f"item(s); {len(diverse) - i - 1} candidate(s) left out")
                break
            tokens_used += token_count
            selected.append(EvidenceItem(
                evidence_id=f"S{len(selected) + 1}",
                chunk_id=cand.chunk_id,
                document_id=cand.document_id,
                text=cand.text,
                clean_text=cleanup.clean_text,
                page_numbers=list(cand.page_numbers),
                source_block_ids=list(cand.source_block_ids),
                heading_path=list(cand.heading_path),
                source_type=SourceType(cand.source_type.value),
                author=getattr(cand, "author", None),
                title=getattr(cand, "title", None),
                section_id=getattr(cand, "section_id", None),
                scores=ScorePath(
                    dense_rank=cand.dense_rank,
                    dense_score=cand.dense_score,
                    bm25_rank=cand.bm25_rank,
                    bm25_score=cand.bm25_score,
                    fusion_rank=cand.fusion_rank,
                    fusion_score=cand.fusion_score,
                    reranker_rank=cand.reranker_rank,
                    reranker_score=cand.reranker_score,
                ),
                token_count=token_count,
                was_cleaned=cleanup.was_cleaned,
                cleanup_operations=cleanup.operations,
                duplicate_group=group_by_chunk.get(cand.chunk_id),
                selection_reason="reranked_relevance",
            ))
        if oversized_seen and not selected:
            pack.warnings.append(
                "no evidence item fits within the configured token budget")

        pack.items = selected
        pack.tokens_used = tokens_used
        return pack
