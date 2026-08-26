"""
Persistent vector index (ChromaDB) with idempotent, incremental indexing.

State tracking (data/index_state.json) records per document:
  - source SHA-256
  - chunking config version
  - embedding model name
  - index schema version

so the system can answer: has this document changed? does it need
re-indexing? Re-running an unchanged document is a no-op; changed documents
are replaced cleanly via upsert of deterministic chunk IDs.
"""
import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional

from jung_archive.embedding.provider import EmbeddingProvider
from jung_archive.models.chunk import Chunk, IndexSchemaMeta

STATE_FORMAT_VERSION = "index-state-1"
DEFAULT_COLLECTION = "jung_archive"


# ----------------------------------------------------------------------
# Chroma HNSW pickle repair (compatibility shim)
#
# After a machine restart / Chroma version change, the persisted HNSW
# index pickle (index_metadata.pickle) can deserialize as a plain dict
# instead of a chromadb PersistentData object.  Chroma's
# PersistentLocalHnswSegment.__init__ then does:
#     self._persist_data = PersistentData.load_from_file(...)
#     self._dimensionality = self._persist_data.dimensionality
# which raises AttributeError: 'dict' object has no attribute 'dimensionality'.
# ----------------------------------------------------------------------
def _recover_hnsw_dimensionality(segment_dir: Path) -> Optional[int]:
    """Recover the true embedding dimensionality from the on-disk HNSW
    index files.

    hnswlib reads the dimensionality from the persisted index header at
    load time, so loading with a dummy ``dim`` and reading ``idx.dim``
    yields the real value regardless of the constructor argument.
    """
    try:
        import hnswlib

        idx = hnswlib.Index(space="cosine", dim=1)
        idx.load_index(
            str(segment_dir),
            is_persistent_index=True,
            max_elements=100000,
        )
        dim = idx.dim
        idx.close_file_handles()
        return int(dim)
    except Exception:
        return None


def repair_corrupted_hnsw_pickles(persist_dir: Path) -> int:
    """Repair Chroma HNSW pickle files that deserialize as plain dicts.

    Returns the number of pickle files repaired.  Healthy pickles
    (proper ``PersistentData`` objects) are left untouched.
    """
    try:
        from chromadb.segment.impl.vector.local_persistent_hnsw import (
            PersistentData,
        )
    except ImportError:
        return 0

    repaired = 0
    for root, _dirs, files in os.walk(str(persist_dir)):
        if "index_metadata.pickle" not in files:
            continue
        path = os.path.join(root, "index_metadata.pickle")
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
        except Exception:
            continue

        if isinstance(obj, PersistentData):
            continue
        if not isinstance(obj, dict):
            continue

        dim = obj.get("dimensionality")
        if dim is None:
            dim = _recover_hnsw_dimensionality(Path(root))
        if dim is None:
            continue

        pd = PersistentData(
            dimensionality=dim,
            total_elements_added=obj.get("total_elements_added", 0),
            id_to_label=obj.get("id_to_label", {}),
            label_to_id=obj.get("label_to_id", {}),
            id_to_seq_id=obj.get("id_to_seq_id", {}),
        )
        if "max_seq_id" in obj:
            pd.max_seq_id = obj["max_seq_id"]

        tmp = path + ".repairing"
        with open(tmp, "wb") as f:
            pickle.dump(pd, f, pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
        repaired += 1

    return repaired


class VectorIndex:
    def __init__(
        self,
        provider: EmbeddingProvider,
        persist_dir: str = "data/chroma",
        collection_name: str = DEFAULT_COLLECTION,
    ):
        self.provider = provider
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client = None
        self._collection = None

    # ------------------------------------------------------------------
    def _ensure_collection(self):
        if self._collection is not None:
            return self._collection
        import chromadb

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        repair_corrupted_hnsw_pickles(self.persist_dir)
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        schema_meta = IndexSchemaMeta(
            embedding_model=self.provider.model_name,
            embedding_dimension=self.provider.dimension or 0,
            normalized=self.provider.normalized,
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                **schema_meta.model_dump(),
            },
        )
        return self._collection

    def collection_metadata(self) -> Dict:
        col = self._ensure_collection()
        return dict(col.metadata or {})

    def collection_schema_meta(self) -> Optional[IndexSchemaMeta]:
        """Return collection metadata as a typed IndexSchemaMeta, or None."""
        raw = self.collection_metadata()
        if not raw:
            return None
        try:
            return IndexSchemaMeta.model_validate(raw)
        except Exception:
            return None

    def count(self) -> int:
        col = self._ensure_collection()
        return int(col.count())

    def get_by_ids(self, ids: List[str]) -> List[Optional[dict]]:
        col = self._ensure_collection()
        res = col.get(ids=ids)
        found = {rid: met for rid, met in zip(res["ids"], res["metadatas"])}
        return [found.get(i) for i in ids]

    # ------------------------------------------------------------------
    def index_chunks(
        self,
        chunks: List[Chunk],
        source_sha256: str,
        chunking_config_version: str,
        force: bool = False,
    ) -> Dict:
        """Idempotently index one document's chunks.

        Returns a report: indexed count, skipped reason, duplicates=0 by
        construction (deterministic IDs + upsert).
        """
        if not chunks:
            return {"indexed": 0, "skipped": "no-chunks", "status": "noop"}

        document_id = chunks[0].document_id
        status = self.needs_reindex(
            document_id, source_sha256, chunking_config_version,
            len(chunks),
        )
        if status == "unchanged" and not force:
            return {
                "indexed": 0,
                "skipped": "unchanged",
                "status": "noop",
                "document_id": document_id,
            }

        # Embed FIRST: provider load determines the real dimension, which
        # must land in the collection metadata (never a placeholder 0).
        texts = [c.text for c in chunks]
        vectors = self.provider.embed(texts)
        col = self._ensure_collection()

        # Delete any previous chunks of this document that are no longer
        # present (stale chunk IDs from older chunking runs).
        existing = col.get(where={"document_id": document_id})
        stale = [rid for rid in existing["ids"] if rid not in {c.chunk_id for c in chunks}]
        if stale:
            col.delete(ids=stale)

        col.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors.tolist(),
            documents=texts,
            metadatas=[
                {
                    "document_id": c.document_id,
                    "source_type": c.source_type.value,
                    "page_numbers": json.dumps(c.page_numbers),
                    "source_block_ids": json.dumps(c.source_block_ids),
                    "heading_path": json.dumps(c.heading_path),
                    "token_count": c.token_count,
                    "chunk_index": c.chunk_index if c.chunk_index is not None else -1,
                    "section_id": c.section_id or "",
                }
                for c in chunks
            ],
        )

        self._record_state(document_id, source_sha256, chunking_config_version,
                           len(chunks))
        return {
            "indexed": len(chunks),
            "replaced_stale": len(stale),
            "status": "ok",
            "document_id": document_id,
        }

    # ------------------------------------------------------------------
    # State tracking
    # ------------------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return self.persist_dir / "index_state.json"

    def load_state(self) -> Dict:
        if not self.state_path.exists():
            return {"format_version": STATE_FORMAT_VERSION, "documents": {}}
        with open(self.state_path, encoding="utf-8") as f:
            return json.load(f)

    def _record_state(self, document_id, sha256, cfg_version, n_chunks):
        state = self.load_state()
        docs = state.setdefault("documents", {})
        docs[document_id] = {
            "source_sha256": sha256,
            "chunking_config_version": cfg_version,
            "embedding_model": self.provider.model_name,
            "embedding_dimension": self.provider.dimension,
            "normalized": self.provider.normalized,
            "index_schema_version": (
                self.collection_metadata().get("index_schema_version")
            ),
            "chunk_count": n_chunks,
        }
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

    def needs_reindex(
        self,
        document_id: str,
        source_sha256: str,
        chunking_config_version: str,
        n_chunks: Optional[int] = None,
    ) -> str:
        """Returns 'new' | 'source-changed' | 'config-changed' |
        'model-changed' | 'unchanged'."""
        state = self.load_state().get("documents", {}).get(document_id)
        if state is None:
            return "new"
        if state.get("source_sha256") != source_sha256:
            return "source-changed"
        if state.get("chunking_config_version") != chunking_config_version:
            return "config-changed"
        if state.get("embedding_model") != self.provider.model_name:
            return "model-changed"
        return "unchanged"
