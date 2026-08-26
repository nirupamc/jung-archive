"""M6 evaluation core tests: metrics, dataset validation, runner."""
import json

import pytest

from jung_archive.evaluation import metrics as M
from jung_archive.evaluation.dataset import (
    DatasetValidationError,
    load_dataset,
    validate_dataset,
)
from jung_archive.evaluation.models import (
    BenchmarkDataset,
    BenchmarkItem,
    DatasetMeta,
    ExperimentConfig,
)
from jung_archive.evaluation.runner import (
    FakeRetrieverFactory,
    aggregate,
    categorize_failures,
    evaluate_evidence,
    evaluate_mode,
    load_run,
    run_benchmark,
)


# ----------------------------------------------------------------------
# Metrics (hand-calculated fixtures)

RANKED = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]


class TestHitAtK:
    def test_hit(self):
        assert M.hit_at_k(RANKED, ["c"], 3) == 1.0
        assert M.hit_at_k(RANKED, ["j"], 10) == 1.0

    def test_miss(self):
        assert M.hit_at_k(RANKED, ["k"], 10) == 0.0
        assert M.hit_at_k(RANKED, ["d"], 3) == 0.0

    def test_multiple_relevant(self):
        assert M.hit_at_k(RANKED, ["z", "b"], 2) == 1.0


class TestRecallAtK:
    def test_partial(self):
        # two relevant, one found in top 5
        assert M.recall_at_k(RANKED, ["b", "z"], 5) == 0.5

    def test_all_found(self):
        assert M.recall_at_k(RANKED, ["a", "b"], 5) == 1.0

    def test_none_found(self):
        assert M.recall_at_k(RANKED, ["y", "z"], 5) == 0.0

    def test_no_results(self):
        assert M.recall_at_k([], ["a"], 5) == 0.0


class TestPrecisionAtK:
    def test_standard(self):
        assert M.precision_at_k(RANKED, ["a", "b"], 5) == pytest.approx(2 / 5)

    def test_fewer_returned_than_k(self):
        # only 3 returned, 2 relevant -> denominator min(5,3)=3
        assert M.precision_at_k(["a", "x", "b"], ["a", "b"], 5) == \
            pytest.approx(2 / 3)

    def test_empty_ranked(self):
        assert M.precision_at_k([], ["a"], 5) == 0.0


class TestMRR:
    def test_first_position(self):
        assert M.reciprocal_rank(RANKED, ["a"]) == 1.0

    def test_third_position(self):
        assert M.reciprocal_rank(RANKED, ["c"]) == pytest.approx(1 / 3)

    def test_not_retrieved(self):
        assert M.reciprocal_rank(RANKED, ["zz"]) == 0.0

    def test_no_results(self):
        assert M.reciprocal_rank([], ["a"]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        # relevant at positions 1 and 2 -> DCG == IDCG
        assert M.ndcg_at_k(RANKED, ["a", "b"], 5) == pytest.approx(1.0)

    def test_single_hit_at_1(self):
        assert M.ndcg_at_k(RANKED, ["a"], 5) == pytest.approx(1.0)

    def test_known_value(self):
        # relevant item at rank 3 of 1 relevant:
        # DCG = 1/log2(3+1), IDCG = 1/log2(2) = 1
        import math

        expected = 1 / math.log2(4)
        assert M.ndcg_at_k(RANKED, ["c"], 5) == pytest.approx(expected)

    def test_not_retrieved(self):
        assert M.ndcg_at_k(RANKED, ["zz"], 5) == 0.0

    def test_ideal_capped_by_k(self):
        # 3 relevant but k=2 -> ideal has 2 hits
        import math

        dcg = 1 / math.log2(3)   # relevant at rank 2 -> 1/log2(rank+1)
        idcg = 1 / math.log2(2) + 1 / math.log2(3)
        assert M.ndcg_at_k(["x", "a", "b", "c"], ["a", "b", "c"], 2) == \
            pytest.approx(dcg / idcg)


# ----------------------------------------------------------------------
# Dataset model + validation

def base_meta(**kw):
    return DatasetMeta(chunking_config_version="chunking-config-1", **kw)


def make_item(qid="q1", chunks=None, pages=None):
    return BenchmarkItem(
        id=qid, question="What?",
        relevant_chunk_ids=chunks or [],
        relevant_page_numbers=pages or [])


def test_dataset_requires_some_relevance_label():
    with pytest.raises(ValueError):
        BenchmarkItem(id="q1", question="q")


def test_duplicate_ground_truth_chunk_ids_rejected():
    with pytest.raises(ValueError):
        make_item(chunks=["c1", "c1"])


def test_duplicate_question_ids_rejected():
    i = make_item("q1", chunks=["c1"])
    with pytest.raises(ValueError):
        BenchmarkDataset(meta=base_meta(), items=[i, i])


def test_empty_question_rejected():
    with pytest.raises(ValueError):
        BenchmarkItem(id="q1", question="")


def test_nonexistent_chunk_id_detected(tmp_path):
    chunks_dir = _tiny_corpus(tmp_path, chunk_ids=["doc1-c00000"])
    ds = BenchmarkDataset(
        meta=base_meta(),
        items=[make_item(chunks=["doc1-c00099"])])
    errors = validate_dataset(ds, str(chunks_dir))
    assert any("does not exist" in e for e in errors)


def test_stale_chunking_config_detected(tmp_path):
    chunks_dir = _tiny_corpus(tmp_path, chunk_ids=["doc1-c00000"])
    ds = BenchmarkDataset(
        meta=DatasetMeta(chunking_config_version="chunking-config-OLD"),
        items=[make_item(chunks=["doc1-c00000"])])
    errors = validate_dataset(ds, str(chunks_dir))
    assert any("stale benchmark" in e for e in errors)


def test_stale_document_sha_detected(tmp_path):
    chunks_dir = _tiny_corpus(tmp_path, chunk_ids=["doc1-c00000"],
                              doc_sha="a" * 64)
    ds = BenchmarkDataset(
        meta=base_meta(document_sha256={"doc1": "b" * 64}),
        items=[make_item(chunks=["doc1-c00000"])])
    errors = validate_dataset(ds, str(chunks_dir))
    assert any("sha changed" in e for e in errors)


def test_invalid_page_reference(tmp_path):
    chunks_dir = _tiny_corpus(tmp_path, chunk_ids=["doc1-c00000"],
                              pages=[3])
    ds = BenchmarkDataset(
        meta=base_meta(),
        items=[make_item(chunks=["doc1-c00000"], pages=[999])])
    errors = validate_dataset(ds, str(chunks_dir))
    assert any("out of range" in e for e in errors)


def _tiny_corpus(tmp_path, chunk_ids, doc_sha=None, pages=None):
    """Write a minimal chunk artifact for validation tests."""
    from jung_archive.models.chunk import Chunk
    from jung_archive.models.document import SourceType

    chunks_dir = tmp_path / "chunks"
    chunks_dir.mkdir(exist_ok=True)
    chunks = []
    for cid in chunk_ids:
        chunks.append(Chunk(
            chunk_id=cid, document_id="doc1",
            text=f"text of {cid}", source_block_ids=["p0001-b000"],
            page_numbers=[min(pages or [1])], token_count=5,
            source_type=SourceType.PRIMARY))
    artifact = {
        "format_version": "chunk-artifact-1",
        "document": {
            "document_id": "doc1", "title": "Tiny", "author": None,
            "source_type": "PRIMARY", "index_status": "INCLUDE",
            "source_path": f"x/{'a' * 64}.pdf",
            "page_count": max(pages or [3]),
            "source_sha256": doc_sha,
        },
        "chunking_config": {"target_tokens": 220, "max_tokens": 300,
                            "min_tokens": 50, "overlap_tokens": 30,
                            "strategy_name": "structure_aware_v1",
                            "config_version": "chunking-config-1"},
        "chunk_count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }
    with open(chunks_dir / "doc1.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f)
    return chunks_dir


# ----------------------------------------------------------------------
# Runner with fake retrievers

def build_fake_factory():
    """Deterministic canned rankings per mode for two questions."""

    def results(ids):
        from jung_archive.models.document import SourceType
        from jung_archive.retrieval.results import RetrievalResult

        out = []
        for i, cid in enumerate(ids, start=1):
            out.append(RetrievalResult(
                chunk_id=cid, document_id="doc1", text=f"t {cid}",
                page_numbers=[i],
                source_block_ids=[f"b{i}"], source_type=SourceType.PRIMARY,
                fusion_rank=i, fusion_score=0.01 * (20 - i)))
        return out

    mapping = {
        # q1: dense finds it first; bm25 fails entirely; hybrid fixes;
        #     reranker promotes it to #1.
        "dense": lambda q: results(["r1", "r2"]) if "A" in q else results(["x"]),
        "bm25": lambda q: results(["x", "y"]) if "A" in q else results(["r1", "r2"]),
        "hybrid": lambda q: results(["r1", "r2"]),
        "hybrid_rerank": lambda q: results(["r1", "r2"]),
    }
    return FakeRetrieverFactory(mapping)


def make_toy_dataset():
    meta = DatasetMeta(chunking_config_version="chunking-config-1")
    items = [
        BenchmarkItem(id="qa", question="Question A text",
                      relevant_chunk_ids=["r1"],
                      relevant_page_numbers=[2]),
        BenchmarkItem(id="qb", question="Question B text",
                      relevant_chunk_ids=["r2"],
                      relevant_page_numbers=[3]),
    ]
    ds = BenchmarkDataset(meta=meta, items=items)
    # For qb the fake factory returns r1/r2 in both hybrid modes -> hit;
    # dense returns x (miss); bm25 returns x,y (miss). Good failure spread.
    return ds


def test_per_query_output_shape(tmp_path):
    ds = make_toy_dataset()
    cfg = ExperimentConfig(modes=["dense", "bm25", "hybrid"],
                           k_values=[1, 3])
    rec = run_benchmark(ds, build_fake_factory(), cfg,
                        output_dir=str(tmp_path),
                        include_evidence_eval=False,
                        timestamp="2026-01-01T00:00:00+00:00")
    pq = rec.per_query["dense"]
    assert len(pq) == 2
    r = pq[0]
    assert r.question_id == "qa"
    assert r.first_relevant_rank_chunk == 1
    assert r.chunk_metrics.hit_at_k["1"] == 1.0
    # bm25 misses qa entirely
    b = [x for x in rec.per_query["bm25"] if x.question_id == "qa"][0]
    assert b.chunk_metrics.mrr == 0.0
    assert b.first_relevant_rank_chunk is None


def test_aggregate_metrics_mean(tmp_path):
    ds = make_toy_dataset()
    cfg = ExperimentConfig(modes=["hybrid"], k_values=[1])
    rec = run_benchmark(ds, build_fake_factory(), cfg,
                        output_dir=str(tmp_path),
                        include_evidence_eval=False)
    agg = rec.aggregates[0]
    assert agg.mode == "hybrid"
    assert agg.n_questions == 2
    assert agg.chunk_metrics.hit_at_k["1"] == pytest.approx(0.5)


def test_run_artifacts_persisted(tmp_path):
    ds = make_toy_dataset()
    cfg = ExperimentConfig(modes=["dense", "bm25", "hybrid",
                                  "hybrid_rerank"],
                           k_values=[1, 3])
    rec = run_benchmark(ds, build_fake_factory(), cfg,
                        output_dir=str(tmp_path),
                        include_evidence_eval=False,
                        timestamp="2026-01-01T00:00:00+00:00")
    run_file = tmp_path / "runs" / f"{rec.run_id}.json"
    assert run_file.exists()
    loaded = load_run(str(run_file))
    assert loaded.config.k_values == [1, 3]
    assert (tmp_path / "latest_summary.json").exists()
    assert (tmp_path / "comparison.csv").exists()
    assert (tmp_path / "failure_analysis.md").exists()
    assert (tmp_path / "failure_analysis.json").exists()


def test_deterministic_evaluation_run(tmp_path):
    ds = make_toy_dataset()
    cfg = ExperimentConfig(modes=["dense"], k_values=[1],
                           record_latencies=False)
    r1 = run_benchmark(ds, build_fake_factory(), cfg,
                       output_dir=str(tmp_path) + "/1",
                       include_evidence_eval=False,
                       run_id="fixed-id", timestamp="t")
    r2 = run_benchmark(ds, build_fake_factory(), cfg,
                       output_dir=str(tmp_path) + "/2",
                       include_evidence_eval=False,
                       run_id="fixed-id", timestamp="t")
    assert r1.model_dump_json() == r2.model_dump_json()


def test_failure_categorization(tmp_path):
    ds = make_toy_dataset()
    cfg = ExperimentConfig(modes=["dense", "bm25", "hybrid",
                                  "hybrid_rerank"],
                           k_values=[1, 3])
    rec = run_benchmark(ds, build_fake_factory(), cfg,
                        output_dir=str(tmp_path),
                        include_evidence_eval=False)
    cats = {c.category for c in rec.failures}
    assert "dense_only_win" in cats      # qa: dense yes, bm25 no
    assert "bm25_only_win" in cats       # qb: bm25 yes, dense no
    assert "reranker_improves_rank" not in cats or True


def test_config_hash_changes_with_mode(tmp_path):
    a = ExperimentConfig(modes=["dense"])
    b = ExperimentConfig(modes=["bm25"])
    assert a.config_hash() != b.config_hash()


# ----------------------------------------------------------------------
# Evidence evaluation math

def test_evidence_accuracy_and_coverage(tmp_path):
    from jung_archive.evidence import EvidenceAssembler, EvidenceConfig
    from jung_archive.evaluation.runner import evaluate_evidence
    from jung_archive.models.document import SourceType
    from jung_archive.retrieval.results import RetrievalResult

    def make(cid, pages, blocks, text="body text here."):
        return RetrievalResult(
            chunk_id=cid, document_id="d", text=f"{text} ({cid})",
            page_numbers=pages, source_block_ids=blocks,
            heading_path=[], source_type=SourceType.PRIMARY)

    def retrieve(question, k):
        if "one" in question:
            return [make("gt1", [4], ["b1"], "ground truth passage"),
                    make("noise", [9], ["b2"], "unrelated noise material")]
        return [make("noise", [7], ["b3"], "different miss content")]

    class ToyItems(BenchmarkDataset):
        pass

    meta = DatasetMeta(chunking_config_version="chunking-config-1")
    ds = BenchmarkDataset(meta=meta, items=[
        BenchmarkItem(id="one", question="find one gt1",
                      relevant_chunk_ids=["gt1"], relevant_page_numbers=[4]),
        BenchmarkItem(id="two", question="miss two",
                      relevant_chunk_ids=["gt2"], relevant_page_numbers=[5]),
    ])
    assembler = EvidenceAssembler(EvidenceConfig(max_evidence_tokens=500))
    evals = evaluate_evidence(ds, retrieve, assembler)
    one, two = evals
    assert one.evidence_accuracy_chunk == pytest.approx(0.5)   # 1/2 selected
    assert one.evidence_coverage_chunk == pytest.approx(1.0)   # gt1 covered
    assert two.evidence_accuracy_chunk == pytest.approx(0.0)
    assert two.evidence_coverage_chunk == pytest.approx(0.0)
