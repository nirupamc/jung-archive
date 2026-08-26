"""
Hybrid retriever orchestrator (M3).

query -> [dense leg] + [bm25 leg] -> RRF -> top_k

Modes: dense | bm25 | hybrid. Filters apply identically to both legs.
EXCLUDE/REVIEW sources are never served; only INCLUDE documents are
eligible unless the index contains nothing else.
"""
import time
from typing import Dict, List, Optional

from jung_archive.embedding.provider import EmbeddingProvider
from jung_archive.indexing.vector_index import VectorIndex
from jung_archive.models.document import IndexStatus, SourceType
from jung_archive.retrieval.dense import (
    DenseRetriever,
    IndexCompatibilityError,
)
from jung_archive.retrieval.fusion import fuse_rankings
from jung_archive.retrieval.lexical import BM25Retriever
from jung_archive.retrieval.results import RetrievalResponse, RetrievalResult

VALID_MODES = ("dense", "bm25", "hybrid")


class HybridRetrieverConfig:
    def __init__(
        self,
        dense_candidate_k: int = 20,
        bm25_candidate_k: int = 20,
        rrf_k: int = 60,
        final_top_k: int = 10,
        mode: str = "hybrid",
        allow_mode_fallback: bool = False,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"invalid retrieval mode {mode!r}; expected one of {VALID_MODES}")
        if final_top_k < 1:
            raise ValueError("final_top_k must be >= 1")
        for name, v in (("dense_candidate_k", dense_candidate_k),
                        ("bm25_candidate_k", bm25_candidate_k)):
            if v < 1:
                raise ValueError(f"{name} must be >= 1")
        self.dense_candidate_k = dense_candidate_k
        self.bm25_candidate_k = bm25_candidate_k
        self.rrf_k = rrf_k
        self.final_top_k = final_top_k
        self.mode = mode
        # Explicit opt-in only; when a fallback happens it is reported.
        self.allow_mode_fallback = allow_mode_fallback


class HybridRetriever:
    def __init__(
        self,
        vector_index: VectorIndex,
        bm25: BM25Retriever,
        config: Optional[HybridRetrieverConfig] = None,
    ):
        self.vi = vector_index
        self.bm25 = bm25
        self.config = config or HybridRetrieverConfig()
        self.dense = DenseRetriever(vector_index)

    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filters: Optional[dict] = None,
        mode: Optional[str] = None,
    ) -> RetrievalResponse:
        started = time.perf_counter()
        cfg = self.config
        mode = (mode or cfg.mode).lower()
        if mode not in VALID_MODES:
            raise ValueError(f"invalid retrieval mode {mode!r}; expected one of {VALID_MODES}")
        k = cfg.final_top_k if top_k is None else top_k
        if k < 1:
            raise ValueError("top_k must be >= 1")

        filters = filters or {}
        allowed_docs, allowed_types, author_title, warnings = \
            self._normalize_filters(filters)

        # Single source of truth for eligibility: chunk artifacts registered
        # in the lexical layer. Only INCLUDE documents are ever served;
        # this constraint is applied identically to BOTH retrieval legs.
        eligible = self._eligible_document_ids()
        if allowed_docs is not None:
            eligible = [d for d in eligible if d in set(allowed_docs)]
        # M4: author/title filters are ENFORCED against the document
        # registry built from chunk artifacts (never silently ignored).
        if author_title and eligible:
            eligible = self._apply_author_title(eligible, author_title)
            if not eligible:
                warnings.append(
                    "no indexed documents match the requested "
                    f"author/title filters {author_title}"
                )
        allowed_docs = eligible
        if not allowed_docs:
            warnings.append("no eligible documents match the current filters")
            latency_ms = (time.perf_counter() - started) * 1000.0
            return RetrievalResponse(
                query=query,
                mode=mode,
                top_k=k,
                filters=filters,
                results=[],
                warnings=warnings,
                latency_ms=round(latency_ms, 2),
            )

        warnings.extend(self._check_eligibility())

         results: List[RetrievalResult] = []
        try:
            if mode == "dense":
                ranked, dense_err = self._dense_leg(
                    query, max(k, cfg.dense_candidate_k),
                    allowed_docs, allowed_types,
                )
                if dense_err:
                    raise RuntimeError(f"dense search failed: {dense_err}")
                results = self._finalize_single(ranked[:k], "dense")
            elif mode == "bm25":
                ranked, bm25_err = self._bm25_leg(
                    query, max(k, cfg.bm25_candidate_k),
                    allowed_docs, allowed_types,
                )
                if bm25_err:
                    raise RuntimeError(f"lexical search failed: {bm25_err}")
                results = self._finalize_single(ranked[:k], "bm25")
            else:
                dense_ranked, dense_err = self._dense_leg(
                    query, cfg.dense_candidate_k, allowed_docs, allowed_types)
                bm25_ranked, bm25_err = self._bm25_leg(
                    query, cfg.bm25_candidate_k, allowed_docs, allowed_types)

                if dense_err and bm25_err:
                    raise RuntimeError(
                        f"both retrieval legs failed: dense={dense_err}; "
                        f"bm25={bm25_err}"
                    )
                if dense_err:
                    if not cfg.allow_mode_fallback:
                        raise RuntimeError(
                            f"dense leg unavailable ({dense_err}); refusing "
                            f"silent fallback to lexical-only"
                        )
                    warnings.append(f"dense leg failed ({dense_err}); fell back to bm25-only")
                    results = self._finalize_single(bm25_ranked[:k], "bm25")
                elif bm25_err:
                    if not cfg.allow_mode_fallback:
                        raise RuntimeError(
                            f"lexical leg unavailable ({bm25_err}); refusing "
                            f"silent fallback to dense-only"
                        )
                    warnings.append(f"bm25 leg failed ({bm25_err}); fell back to dense-only")
                    results = self._finalize_single(dense_ranked[:k], "dense")
                else:
                    fused = fuse_rankings([dense_ranked, bm25_ranked], cfg.rrf_k)
                    results = fused[:k]
        finally:
            latency_ms = (time.perf_counter() - started) * 1000.0

        return RetrievalResponse(
            query=query,
            mode=mode,
            top_k=k,
            filters=filters,
            results=results,
            warnings=warnings,
            latency_ms=round(latency_ms, 2),
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_filters(filters: dict):
        """Return (allowed_document_ids|None, allowed_source_types|None,
        author_title_constraints|None, warnings)."""
        allowed_docs = None
        allowed_types = None
        author_title = {}
        warnings = []
        for key, value in filters.items():
            if key == "document_id":
                ids = value if isinstance(value, list) else [value]
                if not all(isinstance(i, str) and i for i in ids):
                    raise ValueError("document_id filter must be non-empty string(s)")
                allowed_docs = ids
            elif key == "source_type":
                types = value if isinstance(value, list) else [value]
                normalized = []
                for t in types:
                    try:
                        normalized.append(SourceType(t).value)
                    except ValueError:
                        raise ValueError(f"unknown source_type filter {t!r}")
                allowed_types = normalized
            elif key in ("author", "title"):
                values = value if isinstance(value, list) else [value]
                if not all(isinstance(v, str) and v.strip() for v in values):
                    raise ValueError(f"{key} filter must be non-empty string(s)")
                author_title[key] = [v.strip() for v in values]
            else:
                raise ValueError(f"unsupported filter {key!r}")
        return allowed_docs, allowed_types, author_title or None, warnings

    def _document_registry(self) -> Dict[str, Dict[str, Optional[str]]]:
        """document_id -> {author,title} from the chunk-artifact registry."""
        if not self.bm25._docs:
            self.bm25.build_or_load()
        registry: Dict[str, Dict[str, Optional[str]]] = {}
        for doc in self.bm25._docs:
            registry.setdefault(doc.document_id, {
                "author": doc.author, "title": doc.title})
        return registry

    def _apply_author_title(self, eligible: List[str],
                            constraints: Dict[str, List[str]]) -> List[str]:
        """Keep only documents whose registered metadata matches.

        Matching is case-insensitive exact against the registered
        value; lists within a field are OR, fields are AND.
        """
        registry = self._document_registry()
        kept = []
        for doc_id in eligible:
            meta = registry.get(doc_id, {})
            ok = True
            for field, wanted in constraints.items():
                actual = (meta.get(field) or "").strip().lower()
                if actual not in {w.strip().lower() for w in wanted}:
                    ok = False
                    break
            if ok:
                kept.append(doc_id)
        return sorted(kept)

    def _eligible_document_ids(self) -> List[str]:
        """INCLUDE-only document IDs from the chunk-artifact registry."""
        if self.bm25._docs:
            docs = self.bm25._docs
        else:
            self.bm25.build_or_load()
            docs = self.bm25._docs
        return sorted({d.document_id for d in docs
                       if d.index_status == IndexStatus.INCLUDE})

    def _check_eligibility(self) -> List[str]:
        """Warn when REVIEW documents exist in indexed artifacts."""
        warnings = []
        state = self.vi.load_state().get("documents", {})
        review = [
            doc_id for doc_id, meta in state.items()
            if meta.get("index_status") == IndexStatus.REVIEW.value
        ]
        if review:
            warnings.append(
                f"{len(review)} REVIEW document(s) present but never served: {review}"
            )
        return warnings

    def _dense_leg(self, query, k, allowed_docs, allowed_types):
        try:
            pairs = self.dense.search(query, k, allowed_docs, allowed_types)
        except IndexCompatibilityError as e:
            return [], str(e)
        except Exception as e:  # missing index etc.
            msg = str(e)
            if "does not exist" in msg.lower() or "no such" in msg.lower() or \
               "empty" in type(e).__name__.lower():
                return [], f"vector index unavailable: {msg}"
            return [], f"dense search failed: {msg}"
        out = []
        registry = None
        for cand, sim in pairs:
            cand = dict(cand)
            source_type = cand.pop("source_type", "UNKNOWN")
            if registry is None:
                registry = self._document_registry()
            doc_meta = registry.get(cand.get("document_id", ""), {})
            out.append(RetrievalResult(
                **cand,
                source_type=source_type,
                dense_score=round(sim, 6),
                author=doc_meta.get("author"),
                title=doc_meta.get("title"),
            ))
        return out, None

    def _bm25_leg(self, query, k, allowed_docs, allowed_types):
        try:
            pairs = self.bm25.search(query, k, allowed_docs, allowed_types)
        except FileNotFoundError as e:
            return [], f"chunk artifacts unavailable: {e}"
        except ValueError as e:
            if "empty query" in str(e).lower():
                raise
            return [], f"lexical corpus unavailable: {e}"
        out = []
        for doc, score in pairs:
            out.append(RetrievalResult(
                chunk_id=doc.chunk_id,
                document_id=doc.document_id,
                text=self.bm25.raw_texts.get(doc.chunk_id, ""),
                page_numbers=doc.page_numbers,
                source_block_ids=doc.source_block_ids,
                heading_path=doc.heading_path,
                source_type=doc.source_type,
                bm25_score=round(score, 6),
                author=doc.author,
                title=doc.title,
                section_id=doc.section_id,
            ))
        return out, None

    @staticmethod
    def _finalize_single(results: List[RetrievalResult], leg: str) -> List[RetrievalResult]:
        """Assign fusion rank by the single active leg's ordering."""
        for i, res in enumerate(results, start=1):
            res.fusion_rank = i
            res.fusion_score = None  # no cross-leg fusion happened
        return results
