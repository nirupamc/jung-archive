"""Shared fixtures for M3 retrieval tests."""
import pytest

from conftest import build_synthetic_document
from jung_archive.chunking.chunker import StructureAwareChunker
from jung_archive.chunking.validation import require_valid
from jung_archive.models.chunk import Chunk
from jung_archive.models.document import Document


SHADOW_TEXT = ("The shadow is the moral problem that challenges the whole "
               "ego-personality. Nobody can become conscious of the shadow "
               "without considerable moral effort. Individuation requires "
               "integrating the shadow into consciousness. ")
MASS_TEXT = ("Mass-mindedness threatens the individual with absorption into "
             "the crowd psyche where personal responsibility disappears. ")
ALCHEMY_TEXT = ("The alchemical opus transforms base matter through successive "
                "stages toward the philosopher's stone and the lapis. ")


@pytest.fixture(scope="module")
def synthetic_corpus():
    """Three small documents covering shadow / mass / alchemy themes."""
    from jung_archive.models.document import IndexStatus

    docs = []
    for doc_id, title, stype, status, paras in [
        ("docshadow01", "Shadow Studies", "PRIMARY", IndexStatus.INCLUDE,
         [SHADOW_TEXT * 4, SHADOW_TEXT.replace("moral", "ethical") * 4]),
        ("docmass0001", "Society Papers", "PRIMARY", IndexStatus.INCLUDE,
         [MASS_TEXT * 4, MASS_TEXT * 4]),
        ("docalchem02", "Alchemy Notes", "SECONDARY", IndexStatus.INCLUDE,
         [ALCHEMY_TEXT * 4]),
        ("docexcluded", "Excluded Bookey Guide", "SECONDARY", IndexStatus.EXCLUDE,
         [SHADOW_TEXT * 2]),
        ("docreview00", "Unverified Source", "UNKNOWN", IndexStatus.REVIEW,
         [MASS_TEXT * 2]),
    ]:
        document = build_synthetic_document(
            [[("TITLE", title)] + [("PARAGRAPH", p) for p in paras]],
            document_id=doc_id,
            title=title,
            source_type=stype,
        )
        document.index_status = status
        chunks = StructureAwareChunker().chunk_document(document)
        require_valid(chunks, document)
        docs.append((document, chunks))
    return docs


@pytest.fixture(scope="module")
def corpus_chunks(synthetic_corpus):
    out = []
    for _, chunks in synthetic_corpus:
        out.extend(chunks)
    return out


class FakeVectorIndex:
    """Minimal stand-in implementing the VectorIndex surface used by
    DenseRetriever/HybridRetriever, backed by exact cosine over a fixed
    embedding function (no model download in tests)."""

    def __init__(self, provider, rows):
        # rows: list of dicts with chunk fields + "embedding"
        self.provider = provider
        self.rows = rows

    def collection_metadata(self):
        return {
            "embedding_model": self.provider.model_name,
            "embedding_dimension": len(self.rows[0]["embedding"]) if self.rows else 0,
            "normalized": True,
            "index_schema_version": "index-schema-1",
            "chunking_config_version": "chunking-config-1",
        }

    def collection_schema_meta(self):
        from jung_archive.models.chunk import IndexSchemaMeta
        raw = self.collection_metadata()
        return IndexSchemaMeta.model_validate(raw)

    def load_state(self):
        return {"documents": {}}

    class _Col:
        def __init__(self, outer):
            self.outer = outer

        def count(self):
            return len(self.outer.rows)

        def query(self, query_embeddings=None, n_results=5, where=None):
            import numpy as np

            qv = np.asarray(query_embeddings[0], dtype=np.float32)
            hits = []
            for row in self.outer.rows:
                if not self._matches(row.get("metadata", {}), where):
                    continue
                v = np.asarray(row["embedding"], dtype=np.float32)
                sim = float(np.dot(qv, v))  # normalized vectors
                hits.append((sim, row))
            hits.sort(key=lambda t: (-t[0], t[1]["chunk_id"]))
            top = hits[:n_results]
            return {
                "ids": [[h[1]["chunk_id"] for h in top]],
                "distances": [[1.0 - h[0] for h in top]],
                "metadatas": [[self._meta(h[1]) for h in top]],
                "documents": [[h[1]["text"] for h in top]],
            }

        @staticmethod
        def _meta(row):
            import json

            m = dict(row["metadata"])
            m.setdefault("heading_path", json.dumps([]))
            m.setdefault("page_numbers", json.dumps([1]))
            m.setdefault("source_block_ids", json.dumps(["b0"]))
            return m

        @staticmethod
        def _matches(meta, where):
            if not where:
                return True
            if "$and" in where:
                return all(FakeVectorIndex._Col._matches(meta, c) for c in where["$and"])
            for key, cond in where.items():
                if isinstance(cond, dict) and "$in" in cond:
                    if meta.get(key) not in cond["$in"]:
                        return False
                elif meta.get(key) != cond:
                    return False
            return True

    def _ensure_collection(self):
        return FakeVectorIndex._Col(self)


class HashProvider:
    """Deterministic pseudo-embeddings via hashing bag-of-words.

    Similar texts share terms -> similar vectors; enough for ranking tests
    without downloading a model. Satisfies the EmbeddingProvider surface.
    """

    model_name = "hash-provider-test"
    dimension = 64
    normalized = True

    def _vec(self, tokens):
        import hashlib

        import numpy as np

        v = np.zeros(self.dimension, dtype=np.float32)
        for t in tokens:
            digest = hashlib.md5(t.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 else -1.0
            v[idx] += sign
        n = float(np.linalg.norm(v))
        return v / n if n > 0 else v

    def embed(self, texts):
        import numpy as np

        from jung_archive.retrieval.lexical import preprocess

        return np.stack([self._vec(preprocess(t)) for t in texts])

    def embed_one(self, text):
        return self.embed([text])[0]


def build_fake_index(corpus_chunks, provider):
    rows = []
    for c in corpus_chunks:
        rows.append({
            "chunk_id": c.chunk_id,
            "text": c.text,
            "embedding": provider.embed_one(c.text).tolist(),
            "metadata": {
                "document_id": c.document_id,
                "source_type": c.source_type.value,
            },
        })
    return FakeVectorIndex(provider, rows)


def make_bm25(tmp_path, corpus_chunks, statuses=None):
    from jung_archive.retrieval.lexical import BM25Retriever

    # Write chunk artifacts directly
    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir()
    by_doc = {}
    for c in corpus_chunks:
        by_doc.setdefault(c.document_id, []).append(c)
    statuses = statuses or {}
    titles = {}
    for c in corpus_chunks:
        titles[c.document_id] = titles.get(c.document_id, c.document_id)

    for doc_id, chs in by_doc.items():
        artifact = {
            "format_version": "chunk-artifact-1",
            "document": {
                "document_id": doc_id,
                "title": titles[doc_id],
                "author": None,
                "source_type": chs[0].source_type.value,
                "index_status": statuses.get(doc_id, "INCLUDE"),
                "source_path": f"x/{doc_id}.pdf",
                "page_count": 3,
                "source_sha256": None,
            },
            "chunking_config": {"target_tokens": 220, "max_tokens": 300,
                                "min_tokens": 50, "overlap_tokens": 30,
                                "strategy_name": "structure_aware_v1",
                                "config_version": "chunking-config-1"},
            "chunk_count": len(chs),
            "chunks": [c.to_dict() for c in sorted(chs, key=lambda x: x.chunk_id)],
        }
        import json
        with open(chunks_dir / f"{doc_id}.json", "w", encoding="utf-8") as f:
            json.dump(artifact, f)

    state_dir = tmp_path / "bm25"
    retriever = BM25Retriever(chunks_dir=str(chunks_dir), state_dir=str(state_dir))
    retriever.build_or_load()
    return retriever
