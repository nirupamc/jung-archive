"""M3 real-query verification: dense vs bm25 vs hybrid on the live index."""
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from jung_archive.embedding.provider import LocalSentenceTransformerProvider
from jung_archive.indexing.vector_index import VectorIndex
from jung_archive.retrieval.hybrid import HybridRetriever, HybridRetrieverConfig
from jung_archive.retrieval.lexical import BM25Retriever

QUERIES = {
    "A (exact terminology)": "mass-mindedness",
    "B (semantic paraphrase)": "Why does the individual lose independence in a mass society?",
    "C (Jung concept)": "self-knowledge",
    "D (long natural language)": "What protects an individual from being absorbed into the psychology of the mass?",
}


def main():
    provider = LocalSentenceTransformerProvider()
    vi = VectorIndex(provider, persist_dir="data/chroma")
    bm25 = BM25Retriever(chunks_dir="data/chunks", state_dir="data/bm25")
    retriever = HybridRetriever(
        vi, bm25, HybridRetrieverConfig(dense_candidate_k=20,
                                        bm25_candidate_k=20, rrf_k=60)
    )

    for label, query in QUERIES.items():
        print("=" * 74)
        print(f"QUERY {label}: {query}")
        print("-" * 74)
        for mode in ("dense", "bm25", "hybrid"):
            resp = retriever.search(query, top_k=5, mode=mode)
            print(f"\n{mode.upper()} Top 5"
                  + (f"  [{resp.latency_ms:.0f} ms]" if resp.latency_ms else ""))
            for r in resp.results:
                parts = [f"#{r.fusion_rank} {r.chunk_id} p{r.page_numbers}"]
                if mode == "hybrid":
                    d = f"d{r.dense_rank}({r.dense_score:.3f})" if r.dense_rank else "-"
                    b = f"b{r.bm25_rank}({r.bm25_score:.2f})" if r.bm25_rank else "-"
                    parts.append(f"{d} {b} f={r.fusion_score}")
                elif mode == "dense":
                    parts.append(f"sim={r.dense_score:.4f}")
                else:
                    parts.append(f"score={r.bm25_score:.3f}")
                preview = " ".join(r.text.split())[:90]
                print("   " + " ".join(parts))
                print(f"      {preview!r}")


if __name__ == "__main__":
    main()
