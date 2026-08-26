"""Reranker abstraction and sequence-limit safety (M4).

A Reranker scores (query, passage) pairs. Implementations may be local
cross-encoders or any other scoring backend; the retrieval stack depends
only on this interface, never on one concrete model.

Sequence-limit policy (explicit, never silent):
  1. Inspect the reranker's tokenizer limit (`model_max_length`).
  2. Build pairs as (query, document) where the query is NEVER truncated
     first; the document is deterministically prefix-truncated to fit.
  3. Only if the query alone cannot fit is it truncated, and that fact
     is reported.
  4. Every truncation is counted in a PairConstructionReport so callers
     can surface it in warnings/telemetry.

Truncation uses the reranker's own tokenizer when available, falling
back to the deterministic project tokenizer otherwise, so behavior is
reproducible either way.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

from jung_archive.chunking.tokenizer import count_tokens, truncate_to_tokens


class RerankError(Exception):
    """Base class for explicit reranking failures."""


class RerankerUnavailableError(RerankError):
    """The configured reranker backend could not be initialized/used.

    Callers must NOT silently fall back to unreranked ordering unless an
    explicit allow_fallback flag was set by the operator.
    """


@dataclass
class PairConstructionReport:
    """Deterministic telemetry about query/document pair construction."""
    total_pairs: int = 0
    truncated_documents: int = 0
    truncated_queries: int = 0
    model_max_length: int = 0
    notes: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"pairs={self.total_pairs} "
            f"truncated_docs={self.truncated_documents} "
            f"truncated_queries={self.truncated_queries} "
            f"max_length={self.model_max_length}"
        )


def _token_counter(tokenizer=None):
    """Prefer the reranker's own tokenizer; fall back to tiktoken."""
    if tokenizer is not None:
        def count(text: str) -> int:
            try:
                return len(tokenizer.encode(text, add_special_tokens=False))
            except TypeError:
                return len(tokenizer.encode(text))
        return count
    return count_tokens


def _prefix_truncate(text: str, max_tokens: int, tokenizer=None) -> str:
    if max_tokens <= 0:
        return ""
    if tokenizer is not None:
        try:
            ids = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            ids = tokenizer.encode(text)
        if len(ids) <= max_tokens:
            return text
        decoded = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
        return decoded
    return truncate_to_tokens(text, max_tokens)


def build_safe_pairs(
    query: str,
    documents: Sequence[str],
    model_max_length: int,
    tokenizer=None,
    special_token_budget: int = 3,
    min_document_budget: int = 32,
) -> Tuple[List[Tuple[str, str]], PairConstructionReport]:
    """Construct (query, document) pairs that respect the model limit.

    Deterministic rules:
      - usable = model_max_length - special_token_budget
      - the document side always keeps at least `min_document_budget`
        tokens of headroom (bounded by half the window on tiny models);
        an oversized query is prefix-truncated to preserve it.
      - each document longer than (usable - query_tokens) is
        deterministically prefix-truncated; every truncation reported.
    """
    if model_max_length < 8:
        raise RerankError(
            f"implausible reranker sequence limit {model_max_length}")
    if not query.strip():
        raise ValueError("empty query for reranking")

    count = _token_counter(tokenizer)
    usable = model_max_length - max(1, special_token_budget)
    floor = min(min_document_budget, max(1, usable // 2))
    report = PairConstructionReport(
        total_pairs=len(documents), model_max_length=model_max_length)

    q_tokens = count(query)
    final_query = query
    max_query_tokens = usable - floor
    if q_tokens > max_query_tokens:
        final_query = _prefix_truncate(query, max_query_tokens, tokenizer)
        report.truncated_queries += 1
        report.notes.append(
            f"query truncated from {q_tokens} to <= {max_query_tokens} tokens")
        q_tokens = count(final_query)
    doc_budget = max(floor, usable - q_tokens)

    pairs: List[Tuple[str, str]] = []
    for doc in documents:
        d_tokens = count(doc)
        if d_tokens > doc_budget:
            truncated = _prefix_truncate(doc, doc_budget, tokenizer)
            pairs.append((final_query, truncated))
            report.truncated_documents += 1
        else:
            pairs.append((final_query, doc))
    return pairs, report


class Reranker(ABC):
    """Provider-neutral reranking interface.

    Concrete backends declare their identity (model name, sequence
    limit, device, batch size) and score (query, passage) pairs.
    """
    model_name: str
    model_max_length: int
    device: str
    batch_size: int

    @abstractmethod
    def score_pairs(self, query: str, documents: List[str]) -> \
            Tuple[List[float], PairConstructionReport]:
        """Score each (query, document) pair; higher = more relevant.

        Returns (scores aligned with input order, construction report).
        Must raise RerankError subclasses on failure — never fabricate
        scores.
        """

    def rank_results(self, query: str, results):
        from jung_archive.retrieval.results import RetrievalResult  # noqa: F401

        if not results:
            return [], PairConstructionReport()
        texts = [r.text for r in results]
        scores, report = self.score_pairs(query, texts)
        decorated = list(zip(results, scores))
        decorated.sort(key=lambda t: (-t[1], t[0].chunk_id))
        out = []
        for i, (res, score) in enumerate(decorated, start=1):
            res.reranker_score = round(float(score), 6)
            res.reranker_rank = i
            out.append(res)
        return out, report
