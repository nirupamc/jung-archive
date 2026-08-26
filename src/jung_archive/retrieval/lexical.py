"""
Deterministic lexical preprocessing + BM25 index (M3).

Preprocessing strategy (documented, conservative):
  - NFKC Unicode normalization
  - lowercase
  - tokenize on word characters, keeping intra-word hyphens so terms like
    "mass-mindedness" and "self-knowledge" stay intact
  - NO stemming, NO stopword removal: philosophical vocabulary
    (self, shadow, persona, anima, animus, individuation, ...) must remain
    searchable, and short words carry meaning in this corpus

LEXICAL_PREPROCESSING_VERSION must be bumped whenever tokenization changes,
so stale BM25 state is detected instead of silently serving old corpora.
"""
import hashlib
import json
import pickle
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from jung_archive.chunking.artifacts import load_chunk_artifact
from jung_archive.models.document import IndexStatus, SourceType

LEXICAL_PREPROCESSING_VERSION = "lexical-v1"
BM25_SCHEMA_VERSION = "bm25-schema-1"

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-'][a-z0-9]+)*", re.UNICODE)


def preprocess(text: str) -> List[str]:
    """Normalize and tokenize text into lexical terms."""
    import unicodedata

    normalized = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_RE.findall(normalized)


def _corpus_fingerprint(chunks) -> str:
    """Stable fingerprint over chunk ids + normalized texts."""
    h = hashlib.sha256()
    for c in sorted(chunks, key=lambda x: x.chunk_id):
        h.update(b"\x00")
        h.update(c.chunk_id.encode("utf-8"))
        h.update(b"\x01")
        h.update(" ".join(c.text.split()).encode("utf-8"))
    return h.hexdigest()


@dataclass
class BM25State:
    """Compatibility state for one persisted BM25 index."""
    corpus_fingerprint: str
    chunk_count: int
    preprocessing_version: str
    schema_version: str


@dataclass
class LexicalDoc:
    """Per-chunk lexical record kept beside the BM25 index."""
    chunk_id: str
    document_id: str
    source_type: SourceType
    index_status: IndexStatus
    page_numbers: List[int]
    source_block_ids: List[str]
    heading_path: List[str]
    title: Optional[str] = None
    author: Optional[str] = None
    section_id: Optional[str] = None


class BM25Retriever:
    """Local BM25 over the chunk corpus, built from chunk artifacts.

    State is persisted; a retriever rebuilds only when artifacts changed,
    the preprocessing version changed, or no state exists yet.
    """

    def __init__(self, chunks_dir: str = "data/chunks",
                 state_dir: str = "data/bm25"):
        self.chunks_dir = Path(chunks_dir)
        self.state_dir = Path(state_dir)
        self._index = None  # BM25Okapi
        self._docs: List[LexicalDoc] = []
        self.raw_texts: Dict[str, str] = {}  # chunk_id -> original text
        self._state: Optional[BM25State] = None
        self.rebuild_reason: Optional[str] = None

    # ------------------------------------------------------------------
    def _load_all_chunks(self):
        from jung_archive.models.chunk import Chunk

        if not self.chunks_dir.exists():
            raise FileNotFoundError(f"chunk artifacts not found: {self.chunks_dir}")
        chunks: List[Chunk] = []
        doc_meta_by_id: Dict[str, dict] = {}
        for path in sorted(self.chunks_dir.glob("*.json")):
            doc_meta, _, art_chunks = load_chunk_artifact(str(path))
            doc_meta_by_id[doc_meta["document_id"]] = doc_meta
            chunks.extend(art_chunks)
        if not chunks:
            raise ValueError("no chunk artifacts found")
        return chunks, doc_meta_by_id

    def build_or_load(self) -> "BM25Retriever":
        """Load persisted index if compatible; else rebuild deterministically."""
        chunks, doc_meta_by_id = self._load_all_chunks()
        fingerprint = _corpus_fingerprint(chunks)
        expected = BM25State(
            corpus_fingerprint=fingerprint,
            chunk_count=len(chunks),
            preprocessing_version=LEXICAL_PREPROCESSING_VERSION,
            schema_version=BM25_SCHEMA_VERSION,
        )

        persisted = self._read_state()
        if persisted is not None and persisted == expected:
            if self._try_load_index(expected):
                self._state = expected
                return self
            self.rebuild_reason = "persisted index unreadable"
        elif persisted is not None:
            if persisted.corpus_fingerprint != expected.corpus_fingerprint:
                self.rebuild_reason = "chunk corpus changed"
            elif persisted.preprocessing_version != expected.preprocessing_version:
                self.rebuild_reason = "lexical preprocessing version changed"
            elif persisted.schema_version != expected.schema_version:
                self.rebuild_reason = "bm25 schema version changed"
            elif persisted.chunk_count != expected.chunk_count:
                self.rebuild_reason = "chunk count changed"
        else:
            self.rebuild_reason = "no persisted bm25 state"

        self._build(chunks, doc_meta_by_id, expected)
        return self

    def _build(self, chunks, doc_meta_by_id, state: BM25State):
        from rank_bm25 import BM25Okapi

        ordered = sorted(chunks, key=lambda c: c.chunk_id)
        corpus_tokens = [preprocess(c.text) for c in ordered]
        self._index = BM25Okapi(corpus_tokens)
        self.raw_texts = {c.chunk_id: c.text for c in ordered}
        self._docs = [
            LexicalDoc(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                source_type=SourceType(c.source_type.value),
                index_status=IndexStatus(
                    doc_meta_by_id.get(c.document_id, {}).get(
                        "index_status", "REVIEW")),
                page_numbers=c.page_numbers,
                source_block_ids=c.source_block_ids,
                heading_path=c.heading_path,
                title=doc_meta_by_id.get(c.document_id, {}).get("title"),
                author=doc_meta_by_id.get(c.document_id, {}).get("author"),
                section_id=c.section_id,
            )
            for c in ordered
        ]
        self._state = state
        self._persist(state)

    # ------------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.state_dir / "bm25_state.json"

    @property
    def index_path(self) -> Path:
        return self.state_dir / "bm25_index.pkl"

    def _read_state(self) -> Optional[BM25State]:
        if not self.state_path.exists():
            return None
        try:
            with open(self.state_path, encoding="utf-8") as f:
                raw = json.load(f)
            return BM25State(**raw)
        except Exception:
            return None

    def _persist(self, state: BM25State):
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(vars(state), f, indent=2)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "index": self._index,
                "docs": self._docs,
                "raw_texts": self.raw_texts,
                "state": vars(state),
            }, f)

    def _try_load_index(self, state: BM25State) -> bool:
        if not self.index_path.exists():
            return False
        try:
            with open(self.index_path, "rb") as f:
                payload = pickle.load(f)
            self._index = payload["index"]
            self._docs = payload["docs"]
            self.raw_texts = payload.get("raw_texts", {})
            return len(self._docs) == state.chunk_count
        except Exception:
            return False

    # ------------------------------------------------------------------
    def search(self, query: str, top_k: int,
               allowed_document_ids: Optional[List[str]] = None,
               allowed_source_types: Optional[List[str]] = None) -> List[Tuple[LexicalDoc, float]]:
        """Rank eligible docs for query; returns [(doc, bm25_score)] top_k."""
        if not query.strip():
            raise ValueError("empty query")
        if self._index is None or self._state is None:
            self.build_or_load()

        tokens = preprocess(query)
        if not tokens:
            return []

        scores = self._index.get_scores(tokens)
        candidates = []
        for doc, score in zip(self._docs, scores):
            if doc.index_status != IndexStatus.INCLUDE:
                continue  # REVIEW/EXCLUDE never served
            if allowed_document_ids is not None and doc.document_id not in allowed_document_ids:
                continue
            if allowed_source_types is not None and doc.source_type.value not in allowed_source_types:
                continue
            if score > 0:
                candidates.append((doc, float(score)))
        candidates.sort(key=lambda t: (-t[1], t[0].chunk_id))
        return candidates[:top_k]
