"""M4 real-document verification: reranking + evidence on The Undiscovered Self."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jung_archive.embedding.provider import LocalSentenceTransformerProvider
from jung_archive.evidence import EvidenceAssembler, EvidenceConfig, \
    render_evidence_pack
from jung_archive.indexing.vector_index import VectorIndex
from jung_archive.retrieval.hybrid import HybridRetriever, HybridRetrieverConfig
from jung_archive.retrieval.lexical import BM25Retriever
from jung_archive.retrieval.pipeline import RerankingPipeline, \
    RerankingPipelineConfig
from jung_archive.reranking.cross_encoder import LocalCrossEncoderReranker

QUERIES = [
    ("A-exact", "mass-mindedness"),
    ("B-semantic", "Why does the individual lose independence in a mass society?"),
    ("C-conceptual", "self-knowledge"),
    ("D-relational", "What protects an individual from being absorbed into "
                     "the psychology of the mass?"),
]


def main():
    provider = LocalSentenceTransformerProvider()
    vi = VectorIndex(provider, persist_dir="data/chroma")
    bm25 = BM25Retriever(chunks_dir="data/chunks", state_dir="data/bm25")

    hybrid = HybridRetriever(vi, bm25, HybridRetrieverConfig(
        dense_candidate_k=30, bm25_candidate_k=30, final_top_k=20))

    print("Initializing cross-encoder (cold) ...")
    t0 = time.perf_counter()
    reranker = LocalCrossEncoderReranker()
    reranker._ensure_model()
    cold_s = time.perf_counter() - t0
    print(f"  model={reranker.model_name} device={reranker.device} "
          f"max_len={reranker.model_max_length} batch={reranker.batch_size} "
          f"cold_load={cold_s:.2f}s")
    print()

    pipe = RerankingPipeline(
        vi, bm25, reranker,
        RerankingPipelineConfig(dense_candidate_k=30, bm25_candidate_k=30,
                                fusion_candidate_k=20, rerank_top_k=8))

    for label, q in QUERIES:
        base = hybrid.search(q, top_k=10, mode="hybrid")
        resp = pipe.search(q, top_k=8)

        print("=" * 70)
        print(f"QUERY {label}: {q}")
        print("-" * 70)
        print("HYBRID (before reranking):")
        for r in base.results:
            print(f"  #{r.fusion_rank:2d} {r.chunk_id} pages={r.page_numbers}"
                  f" dense={r.dense_score} bm25={r.bm25_score}"
                  f" rrf={r.fusion_score}")
            print(f"       {r.preview(110)}")
        print("AFTER RERANKING:")
        for r in resp.results:
            print(f"  rerank#{r.reranker_rank} (fusion #{r.fusion_rank})"
                  f" {r.chunk_id} pages={r.page_numbers}"
                  f" score={r.reranker_score}")
            print(f"       {r.preview(110)}")
        if resp.warnings:
            print("warnings:", resp.warnings)
        print()

    # Evidence assembly + provenance trace for query D
    q = QUERIES[3][1]
    resp = pipe.search(q, top_k=8)
    t1 = time.perf_counter()
    pack = EvidenceAssembler(EvidenceConfig()).assemble(q, resp.results)
    asm_ms = (time.perf_counter() - t1) * 1000

    warm = []
    for _ in range(3):
        t = time.perf_counter()
        pipe.search(q, top_k=8)
        warm.append((time.perf_counter() - t) * 1000)

    print("=" * 70)
    print(f"EVIDENCE PACK for: {q}")
    print(f"candidates considered: {pack.candidates_considered}")
    print(f"duplicates suppressed : {len(pack.suppressed_duplicates)}"
          f"  {[s.chunk_id + ':' + s.reason for s in pack.suppressed_duplicates]}")
    print(f"diversity suppressed  : {len(pack.suppressed_diversity)}")
    print(f"final items           : {len(pack.items)}")
    print(f"tokens used/budget    : {pack.tokens_used}/{pack.max_evidence_tokens}")
    print(f"assembler latency     : {asm_ms:.1f} ms; "
          f"warm pipeline runs: {[round(w, 1) for w in warm]} ms")
    print()

    if pack.items:
        it = pack.items[0]
        print("PROVENANCE CHAIN S1:")
        print(f"  S1 -> chunk {it.chunk_id}")
        print(f"      -> blocks {it.source_block_ids[:6]}"
              f"{' ...' if len(it.source_block_ids) > 6 else ''}")
        print(f"      -> pages {it.page_numbers}")
        print(f"      -> document {it.document_id} "
              f"(title={it.title!r}, author={it.author!r}, "
              f"type={it.source_type.value})")
        print(f"      -> heading_path {it.heading_path}")
        print(f"      cleaned={it.was_cleaned} ops={it.cleanup_operations}")
        print(f"      tokens={it.token_count}")
    print()
    print("RENDERED PACK (truncated to 1200 chars):")
    print(render_evidence_pack(pack)[:1200])


if __name__ == "__main__":
    main()
