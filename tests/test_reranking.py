"""M4 reranking tests: abstraction, pair safety, propagation, pipeline."""
import pytest

from conftest import build_synthetic_document
from jung_archive.chunking.chunker import StructureAwareChunker
from jung_archive.models.chunk import ChunkingConfig
from jung_archive.retrieval.results import RetrievalResult, RetrievalResponse
from retrieval_fixtures import (
    ALCHEMY_TEXT,
    HashProvider,
    MASS_TEXT,
    SHADOW_TEXT,
    build_fake_index,
    make_bm25,
    synthetic_corpus,  # noqa: F401  (module-scoped corpus)
)

from jung_archive.reranking.base import (
    PairConstructionReport,
    RerankError,
    Reranker,
    RerankerUnavailableError,
    build_safe_pairs,
)


# ----------------------------------------------------------------------
# Fixture rerankers (fully deterministic, no model downloads)

class KeywordReranker(Reranker):
    """Scores pairs by shared lexical tokens between query and doc."""
    model_name = "fixture-keyword"
    model_max_length = 512
    device = "cpu"
    batch_size = 8

    def __init__(self, tokenizer_limit=None):
        from jung_archive.retrieval.lexical import preprocess

        self._preprocess = preprocess
        self.model_max_length = tokenizer_limit or self.model_max_length

    def _tokens(self, text):
        return set(self._preprocess(text))

    def score_pairs(self, query, documents):
        pairs, report = build_safe_pairs(
            query, documents, self.model_max_length)
        q = self._tokens(query)
        scores = []
        for _, doc in pairs:
            scores.append(float(len(q & self._tokens(doc))))
        return scores, report


class ReverseReranker(Reranker):
    """Deliberately reverses input order to prove reranking reorders."""
    model_name = "fixture-reverse"
    model_max_length = 512
    device = "cpu"
    batch_size = 4

    def score_pairs(self, query, documents):
        pairs, report = build_safe_pairs(
            query, documents, self.model_max_length)
        # ascending input index -> desc sort reverses the order
        return [float(i) for i in range(len(documents))], report


class ExplodingReranker(Reranker):
    """Always fails: used to prove errors are explicit, never silent."""
    model_name = "fixture-exploding"
    model_max_length = 512
    device = "cpu"
    batch_size = 1

    def score_pairs(self, query, documents):
        raise RerankerUnavailableError("reranker backend offline")


# ----------------------------------------------------------------------
# Corpus helpers

@pytest.fixture(scope="module")
def rerank_env(synthetic_corpus):
    provider = HashProvider()
    chunks = [c for _, chs in synthetic_corpus for c in chs]
    index = build_fake_index(chunks, provider)
    return provider, chunks, index


def make_pipeline(index, reranker, tmp_path, chunks, **cfg_kwargs):
    """index is a FakeVectorIndex (already carries .provider)."""
    from jung_archive.retrieval.pipeline import (
        RerankingPipeline,
        RerankingPipelineConfig,
    )

    bm25 = make_bm25(tmp_path, chunks,
                     statuses={"docexcluded": "EXCLUDE",
                               "docreview00": "REVIEW"})
    config = RerankingPipelineConfig(**cfg_kwargs)
    return RerankingPipeline(index, bm25, reranker, config)


# ----------------------------------------------------------------------
# 1-2. Abstraction + determinism

def test_reranker_is_abstract():
    with pytest.raises(TypeError):
        Reranker()  # cannot instantiate bare ABC


def test_keyword_reranker_deterministic():
    r = KeywordReranker()
    docs = [MASS_TEXT, SHADOW_TEXT, ALCHEMY_TEXT]
    s1, rep1 = r.score_pairs("mass-mindedness", docs)
    s2, rep2 = r.score_pairs("mass-mindedness", docs)
    assert s1 == s2
    assert rep1.summary() == rep2.summary()


def test_pipeline_requires_reranker_interface(rerank_env, tmp_path):
    provider, chunks, index = rerank_env
    with pytest.raises(TypeError):
        make_pipeline(index, "not-a-reranker", tmp_path, chunks)


# ----------------------------------------------------------------------
# 3. Result model fields

def test_result_model_has_reranker_fields():
    res = RetrievalResult(
        chunk_id="c1", document_id="d1", text="t", page_numbers=[1],
        source_block_ids=["b1"], source_type="PRIMARY",
        reranker_rank=3, reranker_score=-2.5,
    )
    assert res.reranker_rank == 3
    assert res.reranker_score == -2.5
    fresh = RetrievalResult(
        chunk_id="c2", document_id="d1", text="t", page_numbers=[1],
        source_block_ids=["b1"], source_type="PRIMARY",
    )
    assert fresh.reranker_rank is None and fresh.reranker_score is None


# ----------------------------------------------------------------------
# 4-5. Pair construction + sequence limits

def test_pair_construction_basic():
    pairs, report = build_safe_pairs("q", ["a", "b"], 512)
    assert pairs == [("q", "a"), ("q", "b")]
    assert isinstance(report, PairConstructionReport)
    assert report.total_pairs == 2
    assert report.truncated_documents == 0
    assert report.truncated_queries == 0


def test_empty_query_rejected():
    with pytest.raises(ValueError):
        build_safe_pairs("   ", ["a"], 512)


def test_sequence_limit_document_truncation_reported():
    long_doc = " ".join(["word"] * 5000)
    pairs, report = build_safe_pairs(
        "short query", [long_doc], 128, special_token_budget=3)
    assert report.truncated_documents == 1
    assert report.truncated_queries == 0
    q_tokens = len(pairs[0][0].split())
    doc_words = len(pairs[0][1].split())
    assert doc_words < 5000
    # document fits within usable window minus the query
    assert q_tokens + doc_words <= 125


def test_sequence_limit_query_truncated_only_when_necessary():
    huge_query = " ".join(["querytoken"] * 300)
    short_doc = "tiny doc"
    pairs, report = build_safe_pairs(huge_query, [short_doc], 64)
    assert report.truncated_queries == 1
    assert pairs[0][1] == short_doc  # document untouched
    assert len(pairs[0][0].split()) <= 61


def test_implausible_limit_rejected():
    with pytest.raises(RerankError):
        build_safe_pairs("q", ["a"], 2)


def test_fixture_reranker_respects_custom_limit():
    r = KeywordReranker(tokenizer_limit=64)
    long_docs = [" ".join(["alpha"] * 900), "beta beta beta"]
    scores, report = r.score_pairs("alpha alpha alpha", long_docs)
    assert report.truncated_documents >= 1
    assert len(scores) == 2


# ----------------------------------------------------------------------
# 6-7. Score/rank propagation; earlier stages preserved

def test_rank_results_assigns_scores_and_ranks():
    r = KeywordReranker()
    res = [
        RetrievalResult(chunk_id=f"c{i}", document_id="d", text=t,
                        page_numbers=[1], source_block_ids=["b"],
                        source_type="PRIMARY")
        for i, t in enumerate([ALCHEMY_TEXT, MASS_TEXT, SHADOW_TEXT])
    ]
    ranked, report = r.rank_results("mass-mindedness", res)
    assert [x.reranker_rank for x in ranked] == [1, 2, 3]
    assert ranked[0].chunk_id.startswith("docmass") or \
        "mass-mindedness" in ranked[0].text.lower()
    assert all(x.reranker_score is not None for x in ranked)
    # descending scores
    sc = [x.reranker_score for x in ranked]
    assert sc == sorted(sc, reverse=True)


def test_pipeline_preserves_earlier_stage_metadata(rerank_env, tmp_path):
    provider, chunks, index = rerank_env
    pipe = make_pipeline(index, KeywordReranker(), tmp_path, chunks)
    resp = pipe.search("mass-mindedness absorption", top_k=5)
    assert resp.mode == "hybrid_rerank"
    assert resp.results
    for res in resp.results:
        assert res.fusion_rank is not None
        # at least one leg must have contributed
        assert res.dense_score is not None or res.bm25_score is not None
        assert res.reranker_rank is not None
        assert res.reranker_score is not None


# ----------------------------------------------------------------------
# 8-10. Ordering changes, pool + top-k limits

def _fusion_only_order(pipe, query):
    base = pipe.retriever.search(query, top_k=pipe.config.fusion_candidate_k,
                                 mode="hybrid")
    return [r.chunk_id for r in base.results]


def test_reranking_changes_ordering_when_appropriate(rerank_env, tmp_path):
    provider, chunks, index = rerank_env
    pipe = make_pipeline(index, ReverseReranker(), tmp_path, chunks,
                         fusion_candidate_k=6, rerank_top_k=4)
    query = "shadow moral effort individuation"
    before = _fusion_only_order(pipe, query)
    resp = pipe.search(query, top_k=4)
    assert len(before) >= 2
    assert [r.chunk_id for r in resp.results] == list(reversed(before))[:4]


def test_candidate_pool_and_topk_limits(rerank_env, tmp_path):
    provider, chunks, index = rerank_env
    pipe = make_pipeline(index, KeywordReranker(), tmp_path, chunks,
                         fusion_candidate_k=3, rerank_top_k=2)
    resp = pipe.search("shadow mass alchemy individuation opus", top_k=2)
    assert resp.candidates_retrieved <= 3
    assert resp.candidates_reranked <= 3
    assert len(resp.results) <= 2


def test_config_validation():
    from jung_archive.retrieval.pipeline import RerankingPipelineConfig

    with pytest.raises(ValueError):
        RerankingPipelineConfig(fusion_candidate_k=2, rerank_top_k=5)
    with pytest.raises(ValueError):
        RerankingPipelineConfig(mode="bogus")
    with pytest.raises(ValueError):
        RerankingPipelineConfig(dense_candidate_k=0)


# ----------------------------------------------------------------------
# 22. Deterministic tie behavior

def test_ties_break_on_chunk_id():
    class FlatReranker(Reranker):
        model_name = "flat"
        model_max_length = 512
        device = "cpu"
        batch_size = 1

        def score_pairs(self, query, documents):
            pairs, report = build_safe_pairs(query, documents,
                                             self.model_max_length)
            return [1.0] * len(documents), report

    res = [
        RetrievalResult(chunk_id=cid, document_id="d", text="shared words",
                        page_numbers=[1], source_block_ids=["b"],
                        source_type="PRIMARY")
        for cid in ["zzz", "aaa", "mmm"]
    ]
    ranked, _ = FlatReranker().rank_results("words", res)
    assert [r.chunk_id for r in ranked] == ["aaa", "mmm", "zzz"]


# ----------------------------------------------------------------------
# 34-36. Integration + explicit failures / fallback policy

def test_hybrid_rerank_integration_smoke(rerank_env, tmp_path):
    provider, chunks, index = rerank_env
    pipe = make_pipeline(index, KeywordReranker(), tmp_path, chunks)
    resp = pipe.search("individual responsibility mass society", top_k=4)
    assert isinstance(resp, RetrievalResponse)
    assert resp.mode == "hybrid_rerank"
    assert resp.pairs_truncated == 0
    assert resp.latency_ms is not None


def test_reranker_failure_is_explicit_no_silent_fallback(rerank_env, tmp_path):
    provider, chunks, index = rerank_env
    pipe = make_pipeline(index, ExplodingReranker(), tmp_path, chunks)
    with pytest.raises(RerankerUnavailableError):
        pipe.search("shadow", top_k=3)


def test_reranker_fallback_only_when_explicitly_configured(rerank_env, tmp_path):
    provider, chunks, index = rerank_env
    pipe = make_pipeline(index, ExplodingReranker(), tmp_path, chunks,
                         allow_reranker_fallback=True)
    resp = pipe.search("shadow", top_k=3)
    assert any("fell back" in w for w in resp.warnings)
    # unreranked fallback keeps fusion ordering
    assert all(r.reranker_rank is None for r in resp.results)


def test_local_cross_encoder_bad_model_raises_unavailable():
    from jung_archive.reranking.cross_encoder import LocalCrossEncoderReranker

    bad = LocalCrossEncoderReranker(model_name="not-a-real-org/not-a-model-zz")
    with pytest.raises(RerankerUnavailableError):
        bad.score_pairs("q", ["a", "b"])


def test_empty_candidates_short_circuits(rerank_env, tmp_path):
    provider, chunks, index = rerank_env
    pipe = make_pipeline(index, KeywordReranker(), tmp_path, chunks)
    resp = pipe.search(
        "qqqqzzzz nonexistentterm", top_k=3,
        filters={"document_id": ["does-not-exist"]})
    assert resp.results == []
    assert any("empty candidate pool" in w for w in resp.warnings)
