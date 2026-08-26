"""Evaluation runner (M6).

Runs the EXACT production retrieval stack (HybridRetriever /
RerankingPipeline / EvidenceAssembler) against a validated benchmark
dataset, computes metrics at chunk and page relevance levels,
categorizes failures, and persists reproducible run artifacts under
data/evaluation/.

Models are loaded once per run and shared across all questions/modes.
"""
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from jung_archive.evaluation import metrics as M
from jung_archive.evaluation.models import (
    BenchmarkDataset,
    EvidenceEvalItem,
    ExperimentConfig,
    FailureCase,
    GenerationEvaluationRecord,
    LevelMetrics,
    ModeAggregate,
    PerQueryResult,
    RunRecord,
)
from jung_archive.models.document import SourceType

EVIDENCE_DIR = Path("data/evaluation")


def compute_level_metrics(ranked: List[str], relevant: List[str],
                          k_values: List[int]) -> LevelMetrics:
    return LevelMetrics(
        hit_at_k={str(k): M.hit_at_k(ranked, relevant, k) for k in k_values},
        recall_at_k={str(k): round(M.recall_at_k(ranked, relevant, k), 6)
                     for k in k_values},
        precision_at_k={str(k): round(M.precision_at_k(ranked, relevant, k), 6)
                        for k in k_values},
        mrr=round(M.reciprocal_rank(ranked, relevant), 6),
        ndcg_at_k={str(k): round(M.ndcg_at_k(ranked, relevant, k), 6)
                   for k in k_values},
    )


# ----------------------------------------------------------------------
# Production retriever factory

class ProductionRetrieverFactory:
    """Builds mode->retrieve callables from the real production stack.

    A single VectorIndex/BM25/reranker set is created lazily and reused
    for the whole evaluation run.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self._services = None

    def _get(self):
        if self._services is None:
            from jung_archive.embedding.provider import LocalSentenceTransformerProvider
            from jung_archive.indexing.vector_index import VectorIndex
            from jung_archive.retrieval.hybrid import HybridRetriever, HybridRetrieverConfig
            from jung_archive.retrieval.lexical import BM25Retriever
            from jung_archive.reranking.cross_encoder import LocalCrossEncoderReranker

            cfg = self.config
            provider = LocalSentenceTransformerProvider()
            vi = VectorIndex(provider, persist_dir=cfg.chroma_dir)
            bm25 = BM25Retriever(chunks_dir=cfg.chunks_dir,
                                 state_dir=cfg.bm25_state_dir)
            reranker = LocalCrossEncoderReranker(
                model_name=cfg.reranker_model)
            hybrid = HybridRetriever(vi, bm25, HybridRetrieverConfig(
                dense_candidate_k=cfg.dense_candidate_k,
                bm25_candidate_k=cfg.bm25_candidate_k,
                rrf_k=cfg.rrf_k,
                final_top_k=max(cfg.fusion_candidate_k, cfg.rerank_top_k),
                mode="hybrid",
            ))
            pipeline = None
            from jung_archive.retrieval.pipeline import RerankingPipeline, \
                RerankingPipelineConfig

            pipeline = RerankingPipeline(
                vi, bm25, reranker,
                RerankingPipelineConfig(
                    dense_candidate_k=cfg.dense_candidate_k,
                    bm25_candidate_k=cfg.bm25_candidate_k,
                    rrf_k=cfg.rrf_k,
                    fusion_candidate_k=max(cfg.fusion_candidate_k,
                                           cfg.rerank_top_k),
                    rerank_top_k=cfg.rerank_top_k,
                ))
            self._services = {"hybrid": hybrid, "pipeline": pipeline}
        return self._services

    def retriever_for_mode(self, mode: str) -> Callable[[str, int], list]:
        if mode not in ("dense", "bm25", "hybrid", "hybrid_rerank"):
            raise ValueError(f"invalid evaluation mode {mode!r}")
        services = self._get()

        def retrieve(query: str, k: int) -> list:
            if mode == "hybrid_rerank":
                resp = services["pipeline"].search(query, top_k=k)
            else:
                resp = services["hybrid"].search(query, top_k=k, mode=mode)
            return resp.results

        return retrieve


class FakeRetrieverFactory:
    """Test double: maps modes to canned ranked results."""

    def __init__(self, mapping: Dict[str, Callable[[str], list]]):
        self.mapping = mapping

    def retriever_for_mode(self, mode: str):
        fn = self.mapping[mode]

        def retrieve(query: str, k: int) -> list:
            return fn(query)[:k]

        return retrieve


# ----------------------------------------------------------------------
# Runner

def evaluate_mode(mode: str, dataset: BenchmarkDataset, retrieve: Callable,
                  k_values: List[int], max_k: int,
                  record_score_paths: bool = True,
                  record_latencies: bool = True) -> List[PerQueryResult]:
    results: List[PerQueryResult] = []
    for item in dataset.items:
        started = time.perf_counter()
        res_list = retrieve(item.question, max_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        retrieved_chunks = [r.chunk_id for r in res_list]
        retrieved_pages: List[int] = []
        for r in res_list:
            for p in r.page_numbers:
                if p not in retrieved_pages:
                    retrieved_pages.append(p)

        score_paths = []
        if record_score_paths:
            for r in res_list:
                score_paths.append({
                    "chunk_id": r.chunk_id,
                    "dense_rank": r.dense_rank,
                    "dense_score": r.dense_score,
                    "bm25_rank": r.bm25_rank,
                    "bm25_score": r.bm25_score,
                    "fusion_rank": r.fusion_rank,
                    "fusion_score": r.fusion_score,
                    "reranker_rank": r.reranker_rank,
                    "reranker_score": r.reranker_score,
                })

        results.append(PerQueryResult(
            question_id=item.id,
            question=item.question,
            mode=mode,
            retrieved_chunk_ids=retrieved_chunks,
            retrieved_page_numbers=retrieved_pages,
            relevant_chunk_ids=list(item.relevant_chunk_ids),
            relevant_page_numbers=list(item.relevant_page_numbers),
            first_relevant_rank_chunk=M.first_relevant_rank(
                retrieved_chunks, item.relevant_chunk_ids)
            if item.relevant_chunk_ids else None,
            first_relevant_rank_page=M.first_relevant_rank(
                # rank pages by order of appearance in retrieval
                retrieved_pages, item.relevant_page_numbers)
            if item.relevant_page_numbers else None,
            chunk_metrics=compute_level_metrics(
                retrieved_chunks, item.relevant_chunk_ids, k_values),
            page_metrics=compute_level_metrics(
                [str(p) for p in retrieved_pages],
                [str(p) for p in item.relevant_page_numbers], k_values),
            score_paths=score_paths,
            latency_ms=round(latency_ms, 1) if record_latencies else None,
        ))
    return results


def aggregate(results: List[PerQueryResult], mode: str,
              k_values: List[int]) -> ModeAggregate:
    n = len(results)
    if n == 0:
        raise ValueError("cannot aggregate empty result list")

    def mean_level(get: Callable[[PerQueryResult], LevelMetrics]) -> LevelMetrics:
        first = get(results[0])
        agg = LevelMetrics(
            hit_at_k={}, recall_at_k={}, precision_at_k={}, ndcg_at_k={})
        for k in k_values:
            key = str(k)
            agg.hit_at_k[key] = round(
                sum(get(r).hit_at_k[key] for r in results) / n, 4)
            agg.recall_at_k[key] = round(
                sum(get(r).recall_at_k[key] for r in results) / n, 4)
            agg.precision_at_k[key] = round(
                sum(get(r).precision_at_k[key] for r in results) / n, 4)
            agg.ndcg_at_k[key] = round(
                sum(get(r).ndcg_at_k[key] for r in results) / n, 4)
        agg.mrr = round(sum(get(r).mrr for r in results) / n, 4)
        assert first is not None
        return agg

    latencies = [r.latency_ms for r in results if r.latency_ms is not None]
    return ModeAggregate(
        mode=mode,
        n_questions=n,
        chunk_metrics=mean_level(lambda r: r.chunk_metrics),
        page_metrics=mean_level(lambda r: r.page_metrics),
        avg_latency_ms=round(sum(latencies) / len(latencies), 1)
        if latencies else None,
    )


def categorize_failures(per_query: Dict[str, List[PerQueryResult]],
                        dataset: BenchmarkDataset) -> List[FailureCase]:
    """Deterministic failure analysis across modes."""
    by_q: Dict[str, Dict[str, PerQueryResult]] = {}
    for mode, results in per_query.items():
        for r in results:
            by_q.setdefault(r.question_id, {})[mode] = r
    items = {i.id: i for i in dataset.items}

    def hit_at_max(r: Optional[PerQueryResult], max_k: int) -> bool:
        if r is None:
            return False
        vals = r.chunk_metrics.hit_at_k
        return bool(vals.get(str(max_k), 0.0))

    failures: List[FailureCase] = []
    for qid, modes in by_q.items():
        item = items.get(qid)
        question = item.question if item else qid
        gt = item.relevant_chunk_ids if item else []
        dense = modes.get("dense")
        bm25 = modes.get("bm25")
        hybrid = modes.get("hybrid")
        rerank = modes.get("hybrid_rerank")
        base = {
            "ground_truth_chunk_ids": gt,
            "relevant_pages": item.relevant_page_numbers if item else [],
        }

        def add(category: str, detail: Dict):
            failures.append(FailureCase(
                category=category, question_id=qid, question=question,
                detail={**base, **detail}))

        all_ks = [int(k) for m in modes.values()
                  for k in m.chunk_metrics.hit_at_k]
        max_k = max(all_ks) if all_ks else 10

        if dense and bm25:
            dh, bh = (hit_at_max(dense, max_k), hit_at_max(bm25, max_k))
            if dh and not bh:
                add("dense_only_win", {
                    "dense_first_rank": dense.first_relevant_rank_chunk,
                    "bm25_first_rank": bm25.first_relevant_rank_chunk})
            elif bh and not dh:
                add("bm25_only_win", {
                    "dense_first_rank": dense.first_relevant_rank_chunk,
                    "bm25_first_rank": bm25.first_relevant_rank_chunk})
        if dense and bm25 and hybrid:
            if not hit_at_max(dense, max_k) and \
                    not hit_at_max(bm25, max_k) and \
                    hit_at_max(hybrid, max_k):
                add("hybrid_fixes_both", {
                    "hybrid_first_rank": hybrid.first_relevant_rank_chunk})
        if hybrid and rerank:
            hr, rr = (hybrid.first_relevant_rank_chunk,
                      rerank.first_relevant_rank_chunk)
            if hr and rr and rr < hr:
                add("reranker_improves_rank",
                    {"hybrid_rank": hr, "rerank_rank": rr})
            elif hr and rr and rr > hr:
                add("reranker_hurts_rank",
                    {"hybrid_rank": hr, "rerank_rank": rr})
        if all(not hit_at_max(m, max_k) for m in modes.values()):
            add("all_methods_fail", {})
    return failures


def evaluate_evidence(dataset: BenchmarkDataset, retrieve_fn: Callable,
                      assembler, max_items: int = 8,
                      max_tokens: int = 2500) -> List[EvidenceEvalItem]:
    """Evidence/citation quality on the reranked evidence path.

    Definitions (documented):
      evidence_accuracy_chunk = selected items whose chunk_id is ground
        truth / total selected items
      evidence_accuracy_page  = selected items whose page overlaps any
        ground-truth page / total selected items
      evidence_coverage_chunk = distinct ground-truth chunks covered /
        total ground-truth chunks
      evidence_coverage_page  = distinct ground-truth pages covered /
        total ground-truth pages
    """
    out = []
    for item in dataset.items:
        res_list = retrieve_fn(item.question, 8)
        pack = assembler.assemble(item.question, res_list)
        n = len(pack.items)
        acc_c = sum(1 for i in pack.items
                    if i.chunk_id in set(item.relevant_chunk_ids)) / n if n else 0.0
        gt_pages = set(item.relevant_page_numbers)
        acc_p = sum(1 for i in pack.items
                    if gt_pages & set(i.page_numbers)) / n if n else 0.0
        covered_c = len(set(i.chunk_id for i in pack.items)
                        & set(item.relevant_chunk_ids))
        cov_c = covered_c / len(item.relevant_chunk_ids) \
            if item.relevant_chunk_ids else 0.0
        covered_p = len(gt_pages & set(
            p for i in pack.items for p in i.page_numbers))
        cov_p = covered_p / len(item.relevant_page_numbers) \
            if item.relevant_page_numbers else 0.0
        out.append(EvidenceEvalItem(
            question_id=item.id,
            n_items=n,
            tokens_used=pack.tokens_used,
            max_tokens=max_tokens,
            evidence_accuracy_chunk=round(acc_c, 4),
            evidence_accuracy_page=round(acc_p, 4),
            evidence_coverage_chunk=round(cov_c, 4),
            evidence_coverage_page=round(cov_p, 4),
        ))
    return out


def write_failure_markdown(record: RunRecord, path: Path) -> None:
    lines = ["# Failure Analysis", ""]
    cats: Dict[str, List[FailureCase]] = {}
    for f in record.failures:
        cats.setdefault(f.category, []).append(f)
    for cat in sorted(cats):
        lines.append(f"## {cat} ({len(cats[cat])})")
        lines.append("")
        for f in cats[cat][:20]:
            lines.append(f"- **{f.question_id}** — {f.question}")
            gt = f.detail.get("ground_truth_chunk_ids", [])
            lines.append(f"  - ground truth: `{', '.join(gt) or '—'}`")
            extra = {k: v for k, v in f.detail.items()
                     if k not in ("ground_truth_chunk_ids",
                                  "relevant_pages")}
            if extra:
                lines.append(f"  - {extra}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_comparison_csv(records: List[RunRecord], path: Path,
                         k_values: List[int]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["run_id", "mode", "n_questions"]
        for level in ("chunk_metrics", "page_metrics"):
            for metric in ("hit_at_k", "recall_at_k", "precision_at_k",
                           "ndcg_at_k"):
                for k in k_values:
                    header.append(f"{level}.{metric}@{k}")
            header.append(f"{level}.mrr")
        writer.writerow(header)
        for rec in records:
            for agg in rec.aggregates:
                row = [rec.run_id, agg.mode, agg.n_questions]
                for level in ("chunk_metrics", "page_metrics"):
                    lm = getattr(agg, level)
                    for metric in ("hit_at_k", "recall_at_k",
                                   "precision_at_k", "ndcg_at_k"):
                        for k in k_values:
                            row.append(lm.__getattribute__(metric).get(str(k), ""))
                    row.append(lm.mrr)
                writer.writerow(row)


def run_benchmark(
    dataset: BenchmarkDataset,
    factory,
    config: ExperimentConfig,
    output_dir: Optional[str] = None,
    include_evidence_eval: bool = True,
    run_id: Optional[str] = None,
    timestamp: Optional[str] = None,
) -> RunRecord:
    """Execute an evaluation run. `factory` supplies per-mode retrievers.

    Deterministic apart from timestamps/latencies when a fixed clock and
    FakeRetrieverFactory are supplied (used by tests).
    """
    out_root = Path(output_dir) if output_dir else EVIDENCE_DIR
    runs_dir = out_root / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    cfg = config.model_copy(update={
        "dataset_fingerprint": config.dataset_fingerprint or
        dataset.fingerprint()})
    rid = run_id or (
        cfg.config_hash() + "-"
        + (timestamp or datetime.now(timezone.utc)
           .strftime("%Y%m%dT%H%M%SZ")))
    # Filesystem-safe id (Windows forbids ':' in names).
    rid = rid.replace(":", "").replace("+", "")
    ts = timestamp or datetime.now(timezone.utc).isoformat(timespec="seconds")

    started = time.perf_counter()
    aggregates: List[ModeAggregate] = []
    per_query: Dict[str, List[PerQueryResult]] = {}
    all_latencies: List[float] = []

    for mode in cfg.modes:
        retrieve = factory.retriever_for_mode(mode)
        max_k = max(cfg.k_values)
        results = evaluate_mode(mode, dataset, retrieve, cfg.k_values, max_k,
                                record_latencies=cfg.record_latencies)
        per_query[mode] = results
        aggregates.append(aggregate(results, mode, cfg.k_values))
        all_latencies.extend(
            r.latency_ms for r in results if r.latency_ms is not None)

    failures = categorize_failures(per_query, dataset)

    evidence_evals: List[EvidenceEvalItem] = []
    if include_evidence_eval and "hybrid_rerank" in cfg.modes:
        from jung_archive.evidence import EvidenceAssembler, EvidenceConfig

        retrieve = factory.retriever_for_mode("hybrid_rerank")
        assembler = EvidenceAssembler(EvidenceConfig(
            max_evidence_tokens=2500, max_evidence_items=8))
        evidence_evals = evaluate_evidence(
            dataset, retrieve, assembler)

    total_s = time.perf_counter() - started
    record = RunRecord(
        run_id=rid,
        timestamp=ts,
        dataset_path="",   # filled by caller when known
        dataset_version=dataset.meta.dataset_version,
        dataset_fingerprint=dataset.fingerprint(),
        config=cfg,
        aggregates=aggregates,
        per_query=per_query,
        failures=failures,
        evidence_evals=evidence_evals,
        generation_eval=GenerationEvaluationRecord(),
        total_time_s=round(total_s, 1),
        avg_query_time_ms=round(sum(all_latencies) / len(all_latencies), 1)
        if all_latencies else None,
    )

    # Persist artifacts.
    run_path = runs_dir / f"{rid}.json"
    run_path.write_text(
        record.model_dump_json(indent=2), encoding="utf-8")
    (out_root / "latest_summary.json").write_text(
        json.dumps(summary_dict(record), indent=2), encoding="utf-8")
    # Cumulative comparison CSV across every persisted run.
    all_records = []
    for p in sorted(runs_dir.glob("*.json")):
        try:
            all_records.append(load_run(str(p)))
        except Exception:
            continue
    if record not in all_records:
        all_records.append(record)
    k_all = sorted({k for rec in all_records
                    for a in rec.aggregates
                    for k in a.chunk_metrics.hit_at_k}, key=int)
    write_comparison_csv(all_records, out_root / "comparison.csv", k_all)
    (out_root / "failure_analysis.json").write_text(
        json.dumps([f.model_dump() for f in record.failures], indent=2),
        encoding="utf-8")
    write_failure_markdown(record, out_root / "failure_analysis.md")
    return record


def summary_dict(record: RunRecord) -> dict:
    """Compact aggregate view used by latest_summary.json + the API/UI."""
    return {
        "run_id": record.run_id,
        "timestamp": record.timestamp,
        "dataset_version": record.dataset_version,
        "dataset_fingerprint": record.dataset_fingerprint,
        "config": record.config.model_dump(),
        "total_time_s": record.total_time_s,
        "avg_query_time_ms": record.avg_query_time_ms,
        "aggregates": [
            {
                "mode": a.mode,
                "n_questions": a.n_questions,
                "avg_latency_ms": a.avg_latency_ms,
                "chunk_metrics": a.chunk_metrics.model_dump(),
                "page_metrics": a.page_metrics.model_dump(),
            }
            for a in record.aggregates
        ],
        "failure_counts": _failure_counts(record),
        "evidence_summary": _evidence_summary(record),
        "generation_eval": record.generation_eval.model_dump(),
    }


def _failure_counts(record: RunRecord) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for f in record.failures:
        counts[f.category] = counts.get(f.category, 0) + 1
    return counts


def _evidence_summary(record: RunRecord) -> Dict[str, float]:
    if not record.evidence_evals:
        return {}
    n = len(record.evidence_evals)
    keys = ("evidence_accuracy_chunk", "evidence_accuracy_page",
            "evidence_coverage_chunk", "evidence_coverage_page")
    return {
        k: round(sum(getattr(e, k) for e in record.evidence_evals) / n, 4)
        for k in keys
    } | {"questions": n}


def load_run(path: str) -> RunRecord:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return RunRecord(**data)


def list_runs(evaluation_dir: Path = EVIDENCE_DIR) -> List[dict]:
    runs = []
    runs_dir = evaluation_dir / "runs"
    if runs_dir.exists():
        for p in sorted(runs_dir.glob("*.json")):
            try:
                rec = json.loads(p.read_text(encoding="utf-8"))
                runs.append({
                    "run_id": rec["run_id"],
                    "timestamp": rec["timestamp"],
                    "dataset_version": rec.get("dataset_version"),
                    "modes": rec.get("config", {}).get("modes", []),
                    "run_name": rec.get("config", {}).get("run_name"),
                    "path": str(p),
                })
            except Exception:
                continue
    return runs


