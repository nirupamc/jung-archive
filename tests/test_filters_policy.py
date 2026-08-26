"""M4 tests: metadata-filter enforcement and gating policy."""
import pytest

from retrieval_fixtures import (
    HashProvider,
    build_fake_index,
    make_bm25,
    synthetic_corpus,  # noqa: F401
)

from jung_archive.models.document import IndexStatus


@pytest.fixture(scope="module")
def filter_env(synthetic_corpus):
    provider = HashProvider()
    chunks = [c for _, chs in synthetic_corpus for c in chs]
    index = build_fake_index(chunks, provider)
    return provider, chunks, index


STATUS_MAP = {"docexcluded": "EXCLUDE", "docreview00": "REVIEW"}


def make_hybrid(filter_env, tmp_path):
    from jung_archive.retrieval.hybrid import HybridRetriever, \
        HybridRetrieverConfig

    provider, chunks, index = filter_env
    bm25 = make_bm25(tmp_path, chunks, statuses=STATUS_MAP)
    return HybridRetriever(index, bm25, HybridRetrieverConfig())


# ----------------------------------------------------------------------
# 29. author/title truly enforced

def test_title_filter_enforced(filter_env, tmp_path):
    hyb = make_hybrid(filter_env, tmp_path)
    resp = hyb.search("shadow moral effort", top_k=10,
                      filters={"title": ["docshadow01"]})
    assert resp.results
    assert {r.document_id for r in resp.results} == {"docshadow01"}


def test_author_filter_enforced_no_false_matches(filter_env, tmp_path):
    """Fixture registry has author=None, so any author query must yield
    zero results plus an honest warning - never silent acceptance."""
    hyb = make_hybrid(filter_env, tmp_path)
    resp = hyb.search("shadow", top_k=5, filters={"author": ["A. Author"]})
    assert resp.results == []
    assert any("no indexed documents match" in w for w in resp.warnings)


def test_unknown_title_yields_warning_and_empty(filter_env, tmp_path):
    hyb = make_hybrid(filter_env, tmp_path)
    resp = hyb.search("mass-mindedness", top_k=5,
                      filters={"title": ["No Such Book"]})
    assert resp.results == []
    assert any("no indexed documents match" in w for w in resp.warnings)


def test_document_and_source_type_filters_still_enforced(filter_env, tmp_path):
    hyb = make_hybrid(filter_env, tmp_path)
    resp = hyb.search("alchemy opus lapis", top_k=10,
                      filters={"document_id": ["docalchem02"]})
    assert resp.results
    assert {r.document_id for r in resp.results} == {"docalchem02"}

    resp2 = hyb.search("shadow", top_k=20, filters={"source_type": ["SECONDARY"]})
    assert {r.source_type.value for r in resp2.results} <= {"SECONDARY"}


def test_unsupported_filter_rejected(filter_env, tmp_path):
    hyb = make_hybrid(filter_env, tmp_path)
    with pytest.raises(ValueError):
        hyb.search("q", top_k=3, filters={"nonsense": 1})


def test_filters_flow_through_rerank_pipeline(filter_env, tmp_path):
    from jung_archive.retrieval.pipeline import RerankingPipeline, \
        RerankingPipelineConfig

    provider, chunks, index = filter_env
    bm25 = make_bm25(tmp_path, chunks, statuses=STATUS_MAP)

    from jung_archive.reranking.base import Reranker as _R

    class NoopReranker(_R):
        model_name = "noop"
        model_max_length = 512
        device = "cpu"
        batch_size = 1

        def score_pairs(self, query, documents):
            from jung_archive.reranking.base import build_safe_pairs

            pairs, report = build_safe_pairs(query, documents,
                                             self.model_max_length)
            return [0.0] * len(documents), report

    pipe = RerankingPipeline(
        index, bm25, NoopReranker(), RerankingPipelineConfig())
    resp = pipe.search("shadow moral", top_k=4,
                       filters={"title": ["docmass0001"]})
    if resp.results:
        assert {r.document_id for r in resp.results} == {"docmass0001"}
        assert all(r.reranker_rank is not None for r in resp.results)


# ----------------------------------------------------------------------
# 30. EXCLUDE/REVIEW policy still enforced everywhere

def test_excluded_review_never_served(filter_env, tmp_path):
    hyb = make_hybrid(filter_env, tmp_path)
    served_ids = set()
    for q in ["shadow moral", "mass-mindedness absorption",
              "alchemy opus lapis"]:
        for mode in ("dense", "bm25", "hybrid"):
            resp = hyb.search(q, top_k=15, mode=mode)
            served_ids |= {r.document_id for r in resp.results}
    assert "docexcluded" not in served_ids
    assert "docreview00" not in served_ids


def test_excluded_never_served_via_rerank(filter_env, tmp_path):
    from jung_archive.reranking.base import Reranker, build_safe_pairs
    from jung_archive.retrieval.pipeline import RerankingPipeline, \
        RerankingPipelineConfig

    provider, chunks, index = filter_env
    bm25 = make_bm25(tmp_path, chunks, statuses=STATUS_MAP)

    class Flat(Reranker):
        model_name = "flat"
        model_max_length = 512
        device = "cpu"
        batch_size = 1

        def score_pairs(self, query, documents):
            pairs, report = build_safe_pairs(query, documents,
                                             self.model_max_length)
            return [0.0] * len(documents), report

    pipe = RerankingPipeline(index, bm25, Flat(), RerankingPipelineConfig())
    resp = pipe.search("mass-mindedness shadow alchemy", top_k=20)
    assert "docexcluded" not in {r.document_id for r in resp.results}
    assert "docreview00" not in {r.document_id for r in resp.results}


# ----------------------------------------------------------------------
# 31-33. plain modes unchanged

def test_plain_modes_still_work(filter_env, tmp_path):
    hyb = make_hybrid(filter_env, tmp_path)
    for mode, expect_field in [("dense", "dense_score"),
                               ("bm25", "bm25_score"),
                               ("hybrid", "fusion_score")]:
        resp = hyb.search("individuation shadow integration", top_k=3,
                          mode=mode)
        assert resp.mode == mode
        assert len(resp.results) <= 3
        for res in resp.results:
            assert getattr(res, expect_field) is not None
            # M4 fields exist but stay unset on non-reranked paths
            assert res.reranker_rank is None
            assert res.reranker_score is None


def test_response_reports_no_rerank_telemetry(filter_env, tmp_path):
    hyb = make_hybrid(filter_env, tmp_path)
    resp = hyb.search("shadow", top_k=2, mode="hybrid")
    assert resp.candidates_retrieved is None
    assert resp.candidates_reranked is None
