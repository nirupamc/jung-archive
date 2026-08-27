import json

import pytest

from jung_archive.models.document import SourceType
from jung_archive.retrieval.dense import (
    DenseRetriever,
    IndexCompatibilityError,
)
from jung_archive.retrieval.fusion import fuse_rankings
from jung_archive.retrieval.hybrid import (
    HybridRetriever,
    HybridRetrieverConfig,
)
from jung_archive.retrieval.lexical import BM25Retriever, preprocess
from jung_archive.retrieval.results import RetrievalResult

from retrieval_fixtures import (
    ALCHEMY_TEXT,
    HashProvider,
    MASS_TEXT,
    SHADOW_TEXT,
    FakeVectorIndex,
    build_fake_index,
    corpus_chunks,      # noqa: F401  (fixture)
    make_bm25,
    synthetic_corpus,   # noqa: F401  (fixture)
)


class TestRetrievalResultModel:
    def test_serialization_round_trip(self):
        r = RetrievalResult(
            chunk_id="doc1-c00000", document_id="doc1", text="body text",
            page_numbers=[3, 4], source_block_ids=["p0003-b000"],
            heading_path=["Part I"], source_type="PRIMARY",
            dense_rank=2, dense_score=0.84, bm25_rank=1, bm25_score=11.7,
            fusion_rank=1, fusion_score=0.0325,
        )
        data = json.loads(r.model_dump_json())
        rebuilt = RetrievalResult(**data)
        assert rebuilt == r
        assert data["page_numbers"] == [3, 4]

    def test_nullable_leg_fields(self):
        r = RetrievalResult(
            chunk_id="x", document_id="d", text="t", page_numbers=[1],
            source_block_ids=["b"], source_type="PRIMARY",
        )
        assert r.dense_rank is None and r.bm25_rank is None
        assert r.fusion_score is None


class TestDenseRetriever:
    def test_returns_provenance_and_similarity(self, corpus_chunks):
        provider = HashProvider()
        vi = build_fake_index(corpus_chunks, provider)
        dr = DenseRetriever(vi)
        hits = dr.search("shadow moral effort", top_k=3)
        assert hits
        cand, sim = hits[0]
        assert cand["chunk_id"] and cand["document_id"]
        assert cand["source_block_ids"] and cand["page_numbers"]
        assert 0.0 <= sim <= 1.0 + 1e-6

    def test_stable_ranking_deterministic(self, corpus_chunks):
        vi = build_fake_index(corpus_chunks, HashProvider())
        a = DenseRetriever(vi).search("alchemy philosopher stone", top_k=5)
        b = DenseRetriever(vi).search("alchemy philosopher stone", top_k=5)
        assert [c["chunk_id"] for c, _ in a] == [c["chunk_id"] for c, _ in b]

    def test_source_type_filter_applies(self, corpus_chunks):
        vi = build_fake_index(corpus_chunks, HashProvider())
        hits = DenseRetriever(vi).search(
            "shadow", top_k=10, allowed_source_types=["SECONDARY"])
        assert hits, "expected secondary alchemy doc"
        assert all(c["source_type"] == "SECONDARY" for c, _ in hits)

    def test_document_filter_applies(self, corpus_chunks):
        vi = build_fake_index(corpus_chunks, HashProvider())
        hits = DenseRetriever(vi).search(
            "shadow mass alchemy", top_k=10,
            allowed_document_ids=["docmass0001"])
        assert all(c["document_id"] == "docmass0001" for c, _ in hits)

    def test_compatibility_mismatch_detected(self):
        provider = HashProvider()
        vi = build_fake_index([_dummy_chunk()], provider)
        # Freeze metadata to what the index was BUILT with, then swap the
        # query-time provider to a different model name.
        frozen = dict(vi.collection_metadata())
        vi.collection_metadata = lambda: frozen

        class WrongModel(HashProvider):
            model_name = "other-model"

        vi.provider = WrongModel()
        dr = DenseRetriever(vi)
        with pytest.raises(IndexCompatibilityError):
            dr.search("anything", top_k=1)

    def test_compatibility_passes_with_matching_metadata(self, corpus_chunks):
        """When model and dimension match, compatibility validation must not raise."""
        provider = HashProvider()
        vi = build_fake_index(corpus_chunks, provider)
        dr = DenseRetriever(vi)
        dr.validate_compatibility()  # should not raise

    def test_empty_query_rejected(self, corpus_chunks):
        vi = build_fake_index(corpus_chunks, HashProvider())
        with pytest.raises(ValueError):
            DenseRetriever(vi).search("   ", top_k=3)


def _dummy_chunk():
    from conftest import build_synthetic_document
    d = build_synthetic_document([[("PARAGRAPH", "hello world")]],
                                 document_id="dummy00001")
    return StructureAwareChunker().chunk_document(d)[0]


from jung_archive.chunking.chunker import StructureAwareChunker  # noqa: E402
from retrieval_fixtures import FakeVectorIndex  # noqa: E402


class TestBM25:
    def test_builds_from_artifacts(self, tmp_path, corpus_chunks):
        bm25 = make_bm25(tmp_path, corpus_chunks)
        assert bm25._index is not None
        assert bm25._state.chunk_count == len(corpus_chunks)
        assert bm25.state_path.exists()

    def test_lexical_ranking_exact_terms(self, tmp_path, corpus_chunks):
        bm25 = make_bm25(tmp_path, corpus_chunks)
        hits = bm25.search("mass-mindedness absorption crowd", top_k=3)
        assert hits, "mass query must return hits"
        top_doc, score = hits[0]
        assert top_doc.document_id == "docmass0001"
        assert score > 0

    def test_normalization_case_punctuation(self):
        toks = preprocess("The SELF-KNOWLEDGE! It's Jung's 'individuation' process.")
        joined = " ".join(toks)
        assert "self-knowledge" in joined
        assert "individuation" in joined
        assert "!" not in joined

    def test_jung_terms_searchable(self, tmp_path, corpus_chunks):
        bm25 = make_bm25(tmp_path, corpus_chunks)
        for term in ["shadow", "individuation"]:
            hits = bm25.search(term, top_k=3)
            assert hits, f"{term} must be searchable"

    def test_excluded_documents_never_served(self, tmp_path, corpus_chunks,
                                             synthetic_corpus):
        statuses = {d.document_id: d.index_status.value
                    for d, _ in synthetic_corpus}
        bm25 = make_bm25(tmp_path, corpus_chunks, statuses=statuses)
        hits = bm25.search("shadow moral effort", top_k=20)
        served_docs = {h[0].document_id for h in hits}
        assert "docexcluded" not in served_docs

    def test_review_documents_never_served(self, tmp_path, corpus_chunks,
                                           synthetic_corpus):
        statuses = {d.document_id: d.index_status.value
                    for d, _ in synthetic_corpus}
        bm25 = make_bm25(tmp_path, corpus_chunks, statuses=statuses)
        hits = bm25.search("absorption into the crowd psyche", top_k=50)
        served_docs = {h[0].document_id for h in hits}
        assert "docreview00" not in served_docs

    def test_source_type_filter(self, tmp_path, corpus_chunks):
        bm25 = make_bm25(tmp_path, corpus_chunks)
        hits = bm25.search("shadow alchemy opus", top_k=20,
                           allowed_source_types=["SECONDARY"])
        assert all(h[0].source_type.value == "SECONDARY" for h in hits)

    def test_stale_state_detected_and_rebuilt(self, tmp_path, corpus_chunks):
        bm25 = make_bm25(tmp_path, corpus_chunks)
        original_fingerprint = bm25._state.corpus_fingerprint

        # Simulate corpus change: add a chunk artifact file update
        art_path = tmp_path / "chunks" / "docmass0001.json"
        art = json.loads(art_path.read_text(encoding="utf-8"))
        extra = dict(art["chunks"][0])
        extra["chunk_id"] = "docmass0001-c99999"
        extra["text"] = "brand new sentence about synchronicity events."
        art["chunks"].append(extra)
        art["chunk_count"] += 1
        art_path.write_text(json.dumps(art), encoding="utf-8")

        bm25_new = BM25Retriever(chunks_dir=str(tmp_path / "chunks"),
                                 state_dir=str(tmp_path / "bm25"))
        result = bm25_new.build_or_load()
        assert result.rebuild_reason == "chunk corpus changed"
        assert result._state.corpus_fingerprint != original_fingerprint
        hits = bm25_new.search("synchronicity events", top_k=2)
        assert hits and hits[0][0].chunk_id == "docmass0001-c99999"

    def test_unchanged_corpus_skips_rebuild(self, tmp_path, corpus_chunks):
        bm25 = make_bm25(tmp_path, corpus_chunks)
        again = BM25Retriever(chunks_dir=str(tmp_path / "chunks"),
                              state_dir=str(tmp_path / "bm25")).build_or_load()
        assert again.rebuild_reason is None

    def test_empty_query_raises(self, tmp_path, corpus_chunks):
        bm25 = make_bm25(tmp_path, corpus_chunks)
        with pytest.raises(ValueError):
            bm25.search("  ", top_k=3)


class TestRRF:
    @staticmethod
    def _res(cid, doc="d1"):
        return RetrievalResult(
            chunk_id=cid, document_id=doc, text=f"text {cid}",
            page_numbers=[1], source_block_ids=[f"b-{cid}"],
            source_type="PRIMARY",
        )

    def test_formula_correctness(self):
        a = [self._res("x1"), self._res("x2")]
        b = [self._res("x2"), self._res("x3")]
        fused = fuse_rankings([a, b], rrf_k=60)
        by_id = {r.chunk_id: r for r in fused}
        expected_x2 = 1 / (60 + 2) + 1 / (60 + 1)
        expected_x1 = 1 / (60 + 1)
        expected_x3 = 1 / (60 + 2)
        assert abs(by_id["x2"].fusion_score - round(expected_x2, 6)) < 1e-9
        assert abs(by_id["x1"].fusion_score - round(expected_x1, 6)) < 1e-9
        assert abs(by_id["x3"].fusion_score - round(expected_x3, 6)) < 1e-9
        assert fused[0].chunk_id == "x2"

    def test_ties_broken_by_chunk_id(self):
        # Two disjoint single-item lists: both score 1/(k+1) -> exact tie.
        a = [self._res("bbb")]
        b = [self._res("aaa")]
        fused = fuse_rankings([a, b], rrf_k=60)
        scores = [r.fusion_score for r in fused]
        assert scores[0] == scores[1]
        assert [r.chunk_id for r in fused] == ["aaa", "bbb"]

    def test_merge_single_result_per_chunk(self):
        ra = self._res("both")
        rb = self._res("both")
        ra.dense_score = 0.9
        rb.bm25_score = 12.0
        fused = fuse_rankings([[ra], [rb]], rrf_k=60)
        assert len(fused) == 1
        merged = fused[0]
        # ranks are positions within each ranked list
        assert merged.dense_score == 0.9 and merged.dense_rank == 1
        assert merged.bm25_score == 12.0 and merged.bm25_rank == 1

    def test_dense_only_survives(self):
        a = [self._res("denseonly")]
        b = [self._res("other")]
        fused = fuse_rankings([a, b], rrf_k=60)
        ids = [r.chunk_id for r in fused]
        assert "denseonly" in ids
        do = next(r for r in fused if r.chunk_id == "denseonly")
        assert do.bm25_rank is None and do.bm25_score is None

    def test_bm25_only_survives(self):
        a = [self._res("other")]
        b = [self._res("lexonly")]
        fused = fuse_rankings([a, b], rrf_k=60)
        lo = next(r for r in fused if r.chunk_id == "lexonly")
        assert lo.dense_rank is None and lo.dense_score is None
        assert lo.bm25_rank == 1

    def test_raw_scores_not_added(self):
        # cosine ~1.0 and BM25 ~30 are wildly different scales; fusion must
        # use ranks only, so identical ranks yield identical fusion scores.
        ra = self._res("q")
        ra.dense_score = 0.99
        rb = self._res("w")
        rb.bm25_score = 30.0
        fused = fuse_rankings([[ra], [rb]], rrf_k=60)
        assert abs(fused[0].fusion_score - fused[1].fusion_score) < 1e-12


class TestHybridRetriever:
    def _retriever(self, tmp_path, corpus_chunks, synthetic_corpus, mode="hybrid",
                   **cfg_kwargs):
        statuses = {d.document_id: d.index_status.value
                    for d, _ in synthetic_corpus}
        provider = HashProvider()
        vi = build_fake_index(corpus_chunks, provider)
        bm25 = make_bm25(tmp_path, corpus_chunks, statuses=statuses)
        cfg = HybridRetrieverConfig(mode=mode, **cfg_kwargs)
        return HybridRetriever(vi, bm25, cfg)

    def test_hybrid_end_to_end(self, tmp_path, corpus_chunks, synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        resp = retr.search("the shadow requires moral effort", top_k=5)
        assert resp.mode == "hybrid"
        assert resp.results
        assert len(resp.results) <= 5
        first = resp.results[0]
        assert first.source_block_ids and first.page_numbers
        ids = [r.chunk_id for r in resp.results]
        assert len(ids) == len(set(ids))

    def test_modes_dense_bm25_only(self, tmp_path, corpus_chunks, synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        for mode in ("dense", "bm25"):
            resp = retr.search("shadow", top_k=3, mode=mode)
            assert resp.mode == mode
            assert resp.results

    def test_invalid_mode_rejected(self, tmp_path, corpus_chunks, synthetic_corpus):
        with pytest.raises(ValueError):
            HybridRetrieverConfig(mode="semantic")
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        with pytest.raises(ValueError):
            retr.search("shadow", mode="fuzzy")

    def test_invalid_top_k_rejected(self, tmp_path, corpus_chunks, synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        with pytest.raises(ValueError):
            retr.search("shadow", top_k=0)

    def test_empty_query_handled(self, tmp_path, corpus_chunks, synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        with pytest.raises(ValueError):
            retr.search("", top_k=3)

    def test_no_results_handled(self, tmp_path, corpus_chunks, synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        resp = retr.search("zzzzqqqq unrelatedtermxyz", top_k=3)
        assert isinstance(resp.results, list)

    def test_filters_apply_consistently(self, tmp_path, corpus_chunks,
                                        synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        resp = retr.search("shadow mass alchemy opus", top_k=10,
                           filters={"source_type": ["SECONDARY"]})
        assert all(r.source_type.value == "SECONDARY" for r in resp.results)
        resp2 = retr.search("shadow mass alchemy opus", top_k=10,
                            filters={"document_id": ["docshadow01"]})
        assert all(r.document_id == "docshadow01" for r in resp2.results)

    def test_excluded_review_never_in_results(self, tmp_path, corpus_chunks,
                                              synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        for q in ["shadow moral effort", "absorption crowd mass"]:
            resp = retr.search(q, top_k=50)
            docs = {r.document_id for r in resp.results}
            assert "docexcluded" not in docs
            assert "docreview00" not in docs

    def test_candidate_depth_respected(self, tmp_path, corpus_chunks,
                                       synthetic_corpus):
        retr = self._retriever(
            tmp_path, corpus_chunks, synthetic_corpus,
            dense_candidate_k=2, bm25_candidate_k=2, final_top_k=10)
        resp = retr.search("shadow mass alchemy", top_k=10)
        assert len(resp.results) <= 4  # at most 2+2 candidates fused

    def test_final_top_k_respected(self, tmp_path, corpus_chunks, synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        resp = retr.search("shadow mass alchemy opus stone individual",
                           top_k=3)
        assert len(resp.results) <= 3

    def test_provenance_survives_fusion(self, tmp_path, corpus_chunks,
                                        synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        resp = retr.search("shadow moral effort consciousness", top_k=5)
        for r in resp.results:
            assert r.chunk_id and r.document_id and r.text.strip()
            assert r.page_numbers and r.source_block_ids
            if r.fusion_score is not None:
                assert r.dense_rank is not None or r.bm25_rank is not None

    def test_malformed_filter_rejected(self, tmp_path, corpus_chunks,
                                       synthetic_corpus):
        retr = self._retriever(tmp_path, corpus_chunks, synthetic_corpus)
        with pytest.raises(ValueError):
            retr.search("shadow", filters={"bogus": 1})
        with pytest.raises(ValueError):
            retr.search("shadow", filters={"source_type": "NOTATYPE"})

    def test_compatibility_failure_reported_not_silent(
            self, tmp_path, corpus_chunks, synthetic_corpus):
        statuses = {d.document_id: d.index_status.value
                    for d, _ in synthetic_corpus}
        vi = build_fake_index(corpus_chunks, HashProvider())
        frozen = dict(vi.collection_metadata())
        vi.collection_metadata = lambda: frozen

        class WrongModel(HashProvider):
            model_name = "wrong-model"

        vi.provider = WrongModel()
        bm25 = make_bm25(tmp_path, corpus_chunks, statuses=statuses)
        retr = HybridRetriever(vi, bm25,
                               HybridRetrieverConfig(mode="hybrid"))
        with pytest.raises(RuntimeError):
            retr.search("shadow", top_k=3)

    def test_dense_mode_raises_not_silent_zero(
            self, tmp_path, corpus_chunks, synthetic_corpus):
        """DENSE mode must raise RuntimeError when the dense leg fails —
        it must NOT silently return 200 with 0 results and no warnings."""
        statuses = {d.document_id: d.index_status.value
                    for d, _ in synthetic_corpus}
        vi = build_fake_index(corpus_chunks, HashProvider())
        frozen = dict(vi.collection_metadata())
        vi.collection_metadata = lambda: frozen

        class WrongModel(HashProvider):
            model_name = "wrong-model"

        vi.provider = WrongModel()
        bm25 = make_bm25(tmp_path, corpus_chunks, statuses=statuses)
        retr = HybridRetriever(vi, bm25,
                               HybridRetrieverConfig(mode="dense"))
        with pytest.raises(RuntimeError, match="dense search failed"):
            retr.search("shadow", top_k=3)

    def test_hybrid_rerank_refuses_silent_fallback(
            self, tmp_path, corpus_chunks, synthetic_corpus):
        """hybrid_rerank mode must propagate dense-leg failure, not
        silently fall back to lexical-only."""
        from jung_archive.reranking.cross_encoder import (
            LocalCrossEncoderReranker,
        )
        from jung_archive.retrieval.pipeline import (
            RerankingPipeline,
            RerankingPipelineConfig,
        )

        statuses = {d.document_id: d.index_status.value
                    for d, _ in synthetic_corpus}
        vi = build_fake_index(corpus_chunks, HashProvider())
        frozen = dict(vi.collection_metadata())
        vi.collection_metadata = lambda: frozen

        class WrongModel(HashProvider):
            model_name = "wrong-model"

        vi.provider = WrongModel()
        bm25 = make_bm25(tmp_path, corpus_chunks, statuses=statuses)
        reranker = LocalCrossEncoderReranker()
        pipe = RerankingPipeline(
            vi, bm25, reranker,
            RerankingPipelineConfig(fusion_candidate_k=10, rerank_top_k=5),
        )
        # RerankingPipeline runs the "hybrid" leg path, whose dense-leg
        # failure is wrapped as "dense leg unavailable (...); refusing
        # silent fallback to lexical-only" (not the single-leg "dense
        # search failed" message). The key assertion: failure propagates
        # as a loud RuntimeError rather than silently falling back.
        with pytest.raises(RuntimeError, match="dense leg unavailable"):
            pipe.search("shadow", top_k=3)
