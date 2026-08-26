"""
Dense retriever over the existing Chroma index (M3).

Embeds the query with the SAME provider/model as the index and converts
Chroma output into canonical candidates with cosine similarity preserved.
"""
from typing import List, Optional, Tuple

from jung_archive.embedding.provider import EmbeddingProvider
from jung_archive.indexing.vector_index import VectorIndex
from jung_archive.models.document import IndexStatus


class IndexCompatibilityError(Exception):
    """Raised when embedding model/index metadata are incompatible."""


class DenseRetriever:
    def __init__(self, vector_index: VectorIndex):
        self.vi = vector_index

    def validate_compatibility(self) -> None:
        meta = self.vi.collection_schema_meta()
        if meta is None:
            raise IndexCompatibilityError(
                "vector index has no metadata or metadata is unreadable"
            )
        if meta.embedding_model != self.vi.provider.model_name:
            raise IndexCompatibilityError(
                f"index embedded with {meta.embedding_model!r} but query uses "
                f"{self.vi.provider.model_name!r}"
            )
        if self.vi.provider.dimension is not None and \
                meta.embedding_dimension and \
                meta.embedding_dimension != self.vi.provider.dimension:
            raise IndexCompatibilityError(
                f"index dimension {meta.embedding_dimension} != provider "
                f"dimension {self.vi.provider.dimension}"
            )

    def search(
        self,
        query: str,
        top_k: int,
        allowed_document_ids: Optional[List[str]] = None,
        allowed_source_types: Optional[List[str]] = None,
    ) -> List[Tuple[dict, float]]:
        """Return [(candidate_dict, cosine_similarity)] ranked by similarity.

        candidate_dict carries full provenance (chunk_id, document_id,
        page_numbers, source_block_ids, heading_path, source_type).
        """
        if not query.strip():
            raise ValueError("empty query")
        self.validate_compatibility()
        col = self.vi._ensure_collection()

        where = self._build_where(allowed_document_ids, allowed_source_types)
        n = max(top_k * 3, top_k)  # over-fetch so post-filtering can still fill K
        kwargs = dict(query_embeddings=[self.vi.provider.embed_one(query).tolist()],
                      n_results=min(n, max(col.count(), 1)))
        if where is not None:
            kwargs["where"] = where
        res = col.query(**kwargs)

        out: List[Tuple[dict, float]] = []
        ids = res.get("ids") or [[]]
        for cid, dist, meta, text in zip(
            ids[0],
            res.get("distances", [[], ])[0],
            (res.get("metadatas") or [[]])[0],
            (res.get("documents") or [[]])[0],
        ):
            candidate = {
                "chunk_id": cid,
                "document_id": meta.get("document_id", ""),
                "text": text,
                "page_numbers": _loads(meta.get("page_numbers"), []),
                "source_block_ids": _loads(meta.get("source_block_ids"), []),
                "heading_path": _loads(meta.get("heading_path"), []),
                "source_type": meta.get("source_type", "UNKNOWN"),
                "section_id": meta.get("section_id") or None,
            }
            # REVIEW policy: never serve unconfirmed sources from the index.
            if allowed_document_ids is not None and \
                    candidate["document_id"] not in allowed_document_ids:
                continue
            out.append((candidate, 1.0 - float(dist)))  # cosine distance -> similarity
        return out[:top_k]

    @staticmethod
    def _build_where(doc_ids, source_types) -> Optional[dict]:
        clauses = []
        if doc_ids is not None:
            clauses.append({"document_id": {"$in": list(doc_ids)}})
        if source_types is not None:
            clauses.append({"source_type": {"$in": list(source_types)}})
        if not clauses:
            return None
        return {"$and": clauses} if len(clauses) > 1 else clauses[0]


def _loads(raw, default):
    import json

    if raw is None:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default
