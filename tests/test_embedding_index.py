import json

import numpy as np
import pytest

from jung_archive.chunking.artifacts import (
    load_chunk_artifact,
    save_chunk_artifact,
)
from jung_archive.chunking.chunker import StructureAwareChunker
from jung_archive.chunking.validation import require_valid
from jung_archive.embedding.provider import LocalSentenceTransformerProvider
from jung_archive.indexing.vector_index import VectorIndex
from jung_archive.models.chunk import ChunkingConfig


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
