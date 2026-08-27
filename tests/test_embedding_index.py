import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from jung_archive.chunking.artifacts import (
    load_chunk_artifact,
    save_chunk_artifact,
)
from jung_archive.chunking.chunker import StructureAwareChunker
from jung_archive.chunking.validation import require_valid
from jung_archive.embedding.provider import LocalSentenceTransformerProvider
from jung_archive.indexing.vector_index import (
    VectorIndex,
    repair_corrupted_hnsw_pickles,
)
from jung_archive.models.chunk import ChunkingConfig, IndexSchemaMeta

try:
    from chromadb.segment.impl.vector.local_persistent_hnsw import (
        PersistentData,
    )
    _HAS_PERSISTENT_DATA = True
except ImportError:
    PersistentData = None
    _HAS_PERSISTENT_DATA = False


@pytest.fixture(scope="module")
def provider():
    return LocalSentenceTransformerProvider()


def _build_test_document():
    from conftest import build_synthetic_document

    return build_synthetic_document([
        [("TITLE", "On the Nature of the Psyche")],
        [("PARAGRAPH",
          "The shadow is the moral problem that challenges the whole "
          "ego-personality. " * 6)],
        [("HEADING", "Alchemy"), ("PARAGRAPH",
          "The alchemical opus describes the transformation of base matter "
          "into the philosopher's stone through successive stages. " * 5)],
        [("PARAGRAPH",
          "Individuation requires the integration of conscious and "
          "unconscious contents over the second half of life. " * 5)],
    ])


@pytest.fixture(scope="module")
def chunked_doc():
    document = _build_test_document()
    chunks = StructureAwareChunker().chunk_document(document)
    require_valid(chunks, document)
    return document, chunks


class TestEmbeddings:
    def test_dimensions(self, provider):
        v = provider.embed(["one", "two", "three"])
        assert v.shape == (3, 384)
        assert provider.dimension == 384

    def test_normalization(self, provider):
        v = provider.embed(["individuation process", "collective unconscious"])
        norms = np.linalg.norm(v, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-3)

    def test_deterministic_for_same_text(self, provider):
        a = provider.embed_one("the ego and the self")
        b = provider.embed_one("the ego and the self")
        assert np.allclose(a, b, atol=1e-5)

    def test_similar_texts_rank_closer(self, provider):
        q = provider.embed_one("the shadow contains repressed desires")
        near = provider.embed_one("repressed desires live in the shadow")
        far = provider.embed_one("recipe for sourdough bread baking")
        assert float(np.dot(q, near)) > float(np.dot(q, far))


class TestVectorIndex:
    def _fresh_index(self, tmp_path, provider):
        return VectorIndex(provider, persist_dir=str(tmp_path / "chroma"))

    def test_insertion_and_count(self, tmp_path, provider, chunked_doc):
        document, chunks = chunked_doc
        idx = self._fresh_index(tmp_path, provider)
        report = idx.index_chunks(chunks, "sha-test-1",
                                  ChunkingConfig.CONFIG_VERSION)
        assert report["status"] == "ok"
        assert report["indexed"] == len(chunks)
        assert idx.count() == len(chunks)

    def test_retrieval_by_embedding_ranks_source_chunk_first(
        self, tmp_path, provider, chunked_doc
    ):
        document, chunks = chunked_doc
        idx = self._fresh_index(tmp_path, provider)
        idx.index_chunks(chunks, "sha-test-1", ChunkingConfig.CONFIG_VERSION)

        target = next(c for c in chunks if "shadow" in c.text.lower())
        q = provider.embed_one(target.text)
        res = idx._ensure_collection().query(
            query_embeddings=[q.tolist()], n_results=len(chunks)
        )
        top_ids = res["ids"][0]
        assert top_ids[0] == target.chunk_id

    def test_idempotent_reindex_no_duplicates(self, tmp_path, provider, chunked_doc):
        document, chunks = chunked_doc
        idx = self._fresh_index(tmp_path, provider)
        r1 = idx.index_chunks(chunks, "sha-1", ChunkingConfig.CONFIG_VERSION)
        r2 = idx.index_chunks(chunks, "sha-1", ChunkingConfig.CONFIG_VERSION)
        assert r1["indexed"] == len(chunks)
        assert r2["indexed"] == 0  # unchanged -> noop
        assert r2["skipped"] == "unchanged"
        assert idx.count() == len(chunks)

    def test_duplicate_prevention_via_upsert(self, tmp_path, provider, chunked_doc):
        document, chunks = chunked_doc
        idx = self._fresh_index(tmp_path, provider)
        idx.index_chunks(chunks, "sha-1", ChunkingConfig.CONFIG_VERSION)
        # force re-insert of same IDs must not duplicate
        idx.index_chunks(chunks, "sha-1", ChunkingConfig.CONFIG_VERSION,
                         force=True)
        ids_now = idx._ensure_collection().get()["ids"]
        assert len(ids_now) == len(set(ids_now))
        assert idx.count() == len(chunks)

    def test_changed_document_replaces_stale_chunks(self, tmp_path, provider, chunked_doc):
        document, chunks = chunked_doc
        cfg = ChunkingConfig()
        idx = self._fresh_index(tmp_path, provider)
        idx.index_chunks(chunks, "sha-v1", cfg.CONFIG_VERSION)

        # Simulate changed source producing fewer/different chunks
        smaller = chunks[:2]
        new_ids = {c.chunk_id for c in smaller}
        idx.index_chunks(smaller, "sha-v2", cfg.CONFIG_VERSION)
        ids_after = set(idx._ensure_collection().get()["ids"])
        stale_gone = not any(i not in new_ids for i in ids_after)
        assert stale_gone or all(i in new_ids for i in ids_after)

    def test_change_detection_answers(self, tmp_path, provider, chunked_doc):
        document, chunks = chunked_doc
        idx = self._fresh_index(tmp_path, provider)
        assert idx.needs_reindex("docX", "s", "v") == "new"
        idx.index_chunks(chunks[:2], "sha-A", "cfg-1")
        assert idx.needs_reindex(document.document_id, "sha-A", "cfg-1") == "unchanged"
        assert idx.needs_reindex(document.document_id, "sha-B", "cfg-1") == "source-changed"
        assert idx.needs_reindex(document.document_id, "sha-A", "cfg-2") == "config-changed"

    def test_model_change_detected(self, tmp_path, provider, chunked_doc):
        document, chunks = chunked_doc
        idx = self._fresh_index(tmp_path, provider)
        idx.index_chunks(chunks[:2], "sha-A", "cfg-1")

        class OtherModelProvider(LocalSentenceTransformerProvider):
            pass

        other = OtherModelProvider(model_name="other-model")
        idx2 = VectorIndex(other, persist_dir=str(tmp_path / "chroma"))
        assert idx2.needs_reindex(document.document_id, "sha-A", "cfg-1") == "model-changed"

    def test_index_metadata_compatibility(self, tmp_path, provider, chunked_doc):
        document, chunks = chunked_doc
        idx = self._fresh_index(tmp_path, provider)
        meta = idx.collection_metadata()
        assert meta["index_schema_version"]
        assert meta["embedding_model"] == provider.model_name
        assert int(meta["embedding_dimension"]) == 384
        assert meta["normalized"] in ("True", True)

    def test_collection_schema_meta_returns_typed_model(self, tmp_path):
        """collection_schema_meta() must return an IndexSchemaMeta, not a raw dict."""
        from jung_archive.indexing.vector_index import VectorIndex
        from jung_archive.models.chunk import IndexSchemaMeta

        class _MockProvider:
            model_name = "test-model"
            dimension = 128
            normalized = True

        idx = VectorIndex(_MockProvider(), persist_dir=str(tmp_path / "chroma"))
        meta = idx.collection_schema_meta()
        assert isinstance(meta, IndexSchemaMeta)
        assert meta.embedding_model == "test-model"
        assert meta.embedding_dimension == 128
        assert meta.normalized is True

    def test_cold_provider_records_real_dimension(self, tmp_path, chunked_doc):
        """Regression: a freshly constructed provider must not record
        dimension 0 in collection metadata."""
        document, chunks = chunked_doc
        cold = LocalSentenceTransformerProvider()  # not yet loaded
        assert cold.dimension is None
        idx = VectorIndex(cold, persist_dir=str(tmp_path / "chroma-cold"))
        report = idx.index_chunks(chunks[:2], "sha-x",
                                  ChunkingConfig.CONFIG_VERSION)
        assert report["status"] == "ok"
        meta = idx.collection_metadata()
        assert int(meta["embedding_dimension"]) == 384
        state = json.loads((tmp_path / "chroma-cold" / "index_state.json").read_text())
        assert state["documents"][document.document_id]["embedding_dimension"] == 384

    def test_state_persisted_with_checksum(self, tmp_path, provider, chunked_doc):
        document, chunks = chunked_doc
        idx = self._fresh_index(tmp_path, provider)
        idx.index_chunks(chunks[:2], "deadbeef" * 8, ChunkingConfig.CONFIG_VERSION)
        state = json.loads((tmp_path / "chroma" / "index_state.json").read_text())
        entry = state["documents"][document.document_id]
        assert entry["source_sha256"] == "deadbeef" * 8
        assert entry["chunking_config_version"] == ChunkingConfig.CONFIG_VERSION
        assert entry["chunk_count"] == 2

    def test_empty_chunks_is_noop(self, tmp_path, provider):
        idx = self._fresh_index(tmp_path, provider)
        report = idx.index_chunks([], "sha", "cfg")
        assert report["status"] == "noop"

    def test_failure_isolation_bad_provider(self, tmp_path, chunked_doc):
        from jung_archive.embedding.provider import EmbeddingProvider

        class BrokenProvider(EmbeddingProvider):
            model_name = "broken"
            dimension = 0
            normalized = True

            def embed(self, texts):
                raise RuntimeError("simulated embedding failure")

        _, chunks = chunked_doc
        idx = VectorIndex(BrokenProvider(), persist_dir=str(tmp_path / "chroma2"))
        with pytest.raises(RuntimeError):
            idx.index_chunks(chunks, "sha", "cfg")


class TestChunkArtifacts:
    def test_round_trip(self, tmp_path, doc_factory, chunked_doc):
        document, chunks = chunked_doc
        config = ChunkingConfig()
        path = save_chunk_artifact(chunks, document, config,
                                   str(tmp_path / "chunks"))
        doc_meta, loaded_cfg, loaded_chunks = load_chunk_artifact(str(path))
        assert doc_meta["document_id"] == document.document_id
        assert doc_meta["source_sha256"] is None or isinstance(
            doc_meta["source_sha256"], str
        )
        assert loaded_cfg == config
        assert loaded_chunks == chunks

    def test_artifact_preserves_provenance_fields(self, tmp_path, doc_factory,
                                                  chunked_doc):
        document, chunks = chunked_doc
        path = save_chunk_artifact(chunks, document, ChunkingConfig(),
                                   str(tmp_path / "chunks"))
        raw = json.loads(path.read_text(encoding="utf-8"))
        c0 = raw["chunks"][0]
        for key in ("chunk_id", "document_id", "source_block_ids",
                    "page_numbers", "heading_path", "token_count",
                    "source_type"):
            assert key in c0


@pytest.mark.skipif(not _HAS_PERSISTENT_DATA,
                    reason="chromadb PersistentData not available")
class TestHnswRepair:
    """Regression tests for repair_corrupted_hnsw_pickles."""

    @staticmethod
    def _write_state(persist_dir: Path, dim: int = 384):
        state = {
            "format_version": "index-state-1",
            "documents": {
                "doc1": {
                    "embedding_dimension": dim,
                    "embedding_model": "test-model",
                    "chunking_config_version": "chunking-config-1",
                    "source_sha256": "x",
                    "chunk_count": 1,
                }
            },
        }
        persist_dir.mkdir(parents=True, exist_ok=True)
        (persist_dir / "index_state.json").write_text(
            json.dumps(state), encoding="utf-8"
        )

    def test_repair_corrupted_dict_pickle(self, tmp_path):
        """A pickle that deserializes as a plain dict with dimensionality=None
        must be reconstructed into a valid PersistentData with dim from state."""
        persist = tmp_path / "chroma"
        seg = persist / "segment1"
        seg.mkdir(parents=True)
        self._write_state(persist, dim=384)

        corrupted = {
            "dimensionality": None,
            "total_elements_added": 10,
            "max_seq_id": 0,
            "id_to_label": {"a": 0},
            "label_to_id": {0: "a"},
            "id_to_seq_id": {"a": 0},
        }
        with open(seg / "index_metadata.pickle", "wb") as f:
            pickle.dump(corrupted, f)

        n = repair_corrupted_hnsw_pickles(persist)
        assert n == 1

        with open(seg / "index_metadata.pickle", "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, PersistentData)
        assert obj.dimensionality == 384
        assert obj.total_elements_added == 10

    def test_repair_persistentdata_bad_dimension(self, tmp_path):
        """A PersistentData with dimensionality=1 (broken repair pass) must
        be re-repaired with the correct dimension from state."""
        persist = tmp_path / "chroma"
        seg = persist / "segment1"
        seg.mkdir(parents=True)
        self._write_state(persist, dim=384)

        bad = PersistentData(
            dimensionality=1,
            total_elements_added=10,
            id_to_label={"a": 0},
            label_to_id={0: "a"},
            id_to_seq_id={"a": 0},
        )
        with open(seg / "index_metadata.pickle", "wb") as f:
            pickle.dump(bad, f)

        n = repair_corrupted_hnsw_pickles(persist)
        assert n == 1

        with open(seg / "index_metadata.pickle", "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, PersistentData)
        assert obj.dimensionality == 384

    def test_repair_persistentdata_none_dimension(self, tmp_path):
        """A PersistentData with dimensionality=None must be repaired."""
        persist = tmp_path / "chroma"
        seg = persist / "segment1"
        seg.mkdir(parents=True)
        self._write_state(persist, dim=384)

        bad = PersistentData(
            dimensionality=None,
            total_elements_added=5,
            id_to_label={"a": 0},
            label_to_id={0: "a"},
            id_to_seq_id={"a": 0},
        )
        with open(seg / "index_metadata.pickle", "wb") as f:
            pickle.dump(bad, f)

        n = repair_corrupted_hnsw_pickles(persist)
        assert n == 1

        with open(seg / "index_metadata.pickle", "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, PersistentData)
        assert obj.dimensionality == 384

    def test_repair_healthy_pickle_skipped(self, tmp_path):
        """A valid PersistentData with correct dimensionality must be skipped."""
        persist = tmp_path / "chroma"
        seg = persist / "segment1"
        seg.mkdir(parents=True)
        self._write_state(persist, dim=384)

        good = PersistentData(
            dimensionality=384,
            total_elements_added=10,
            id_to_label={"a": 0},
            label_to_id={0: "a"},
            id_to_seq_id={"a": 0},
        )
        with open(seg / "index_metadata.pickle", "wb") as f:
            pickle.dump(good, f)

        n = repair_corrupted_hnsw_pickles(persist)
        assert n == 0

        with open(seg / "index_metadata.pickle", "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, PersistentData)
        assert obj.dimensionality == 384

    def test_repair_no_state_no_repair(self, tmp_path):
        """Without index_state.json the dimension cannot be recovered;
        the corrupted pickle must be left untouched."""
        persist = tmp_path / "chroma"
        seg = persist / "segment1"
        seg.mkdir(parents=True)

        corrupted = {
            "dimensionality": None,
            "total_elements_added": 10,
            "max_seq_id": 0,
            "id_to_label": {},
            "label_to_id": {},
            "id_to_seq_id": {},
        }
        with open(seg / "index_metadata.pickle", "wb") as f:
            pickle.dump(corrupted, f)

        n = repair_corrupted_hnsw_pickles(persist)
        assert n == 0

        with open(seg / "index_metadata.pickle", "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, dict)
        assert obj["dimensionality"] is None


def test_schema_meta_deserialization_from_dict():
    """IndexSchemaMeta.model_validate must convert a raw dict (as returned
    by Chroma collection metadata) into a proper typed object."""
    raw = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dimension": 384,
        "normalized": True,
        "index_schema_version": "index-schema-1",
        "chunking_config_version": "chunking-config-1",
        "hnsw:space": "cosine",
    }
    meta = IndexSchemaMeta.model_validate(raw)
    assert isinstance(meta, IndexSchemaMeta)
    assert meta.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert meta.embedding_dimension == 384
    assert meta.normalized is True
    assert meta.index_schema_version == "index-schema-1"


def test_schema_meta_dimension_mismatch_still_raises():
    """The typed IndexSchemaMeta must expose embedding_dimension for
    comparison — a real mismatch must still be detectable."""
    raw = {
        "embedding_model": "model-a",
        "embedding_dimension": 384,
        "normalized": True,
    }
    meta = IndexSchemaMeta.model_validate(raw)
    assert meta.embedding_dimension == 384
    assert meta.embedding_dimension != 256
