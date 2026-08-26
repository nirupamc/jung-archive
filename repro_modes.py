import traceback
import os

# Test 1: DENSE mode only (no reranker)
print("=== Test DENSE mode ===")
try:
    from jung_archive.embedding.provider import LocalSentenceTransformerProvider
    from jung_archive.indexing.vector_index import VectorIndex
    from jung_archive.retrieval.dense import DenseRetriever
    from jung_archive.retrieval.lexical import BM25Retriever
    from jung_archive.retrieval.hybrid import HybridRetriever, HybridRetrieverConfig
    from jung_archive.retrieval.results import RetrievalResult

    provider = LocalSentenceTransformerProvider()
    vi = VectorIndex(provider, persist_dir="data/chroma")
    bm25 = BM25Retriever(chunks_dir="data/chunks", state_dir="data/bm25")

    retriever = HybridRetriever(vi, bm25, HybridRetrieverConfig(mode="dense"))
    resp = retriever.search("How does Jung describe the Self?", top_k=5, mode="dense")
    print(f"DENSE results={len(resp.results)} warnings={resp.warnings}")
    for r in resp.results[:2]:
        print(f"  chunk={r.chunk_id[:12]} doc={r.document_id[:12]} title={str(r.title or '')[:40]}")
except Exception as e:
    print(f"DENSE ERROR: {e}")
    traceback.print_exc()
print()

# Test 2: BM25 mode
print("=== Test BM25 mode ===")
try:
    bm252 = BM25Retriever(chunks_dir="data/chunks", state_dir="data/bm25")
    retriever2 = HybridRetriever(vi, bm252, HybridRetrieverConfig(mode="bm25"))
    resp = retriever2.search("How does Jung describe the Self?", top_k=5, mode="bm25")
    print(f"BM25 results={len(resp.results)} warnings={resp.warnings}")
    for r in resp.results[:2]:
        print(f"  chunk={r.chunk_id[:12]} doc={r.document_id[:12]} title={str(r.title or '')[:40]}")
except Exception as e:
    print(f"BM25 ERROR: {e}")
    traceback.print_exc()
print()

# Test 3: HYBRID mode
print("=== Test HYBRID mode ===")
try:
    retriever3 = HybridRetriever(vi, bm25, HybridRetrieverConfig(mode="hybrid"))
    resp = retriever3.search("How does Jung describe the Self?", top_k=5, mode="hybrid")
    print(f"HYBRID results={len(resp.results)} warnings={resp.warnings}")
    for r in resp.results[:2]:
        print(f"  chunk={r.chunk_id[:12]} doc={r.document_id[:12]} title={str(r.title or '')[:40]}")
except Exception as e:
    print(f"HYBRID ERROR: {e}")
    traceback.print_exc()
print()

# Test 4: HYBRID + RERANK mode
print("=== Test HYBRID + RERANK mode ===")
try:
    from jung_archive.reranking.cross_encoder import LocalCrossEncoderReranker
    from jung_archive.retrieval.pipeline import RerankingPipeline, RerankingPipelineConfig

    reranker = LocalCrossEncoderReranker()
    pipe = RerankingPipeline(vi, bm25, reranker, RerankingPipelineConfig())
    resp = pipe.search("How does Jung describe the Self?", top_k=5)
    print(f"HYBRID_RERANK results={len(resp.results)} warnings={resp.warnings}")
    for r in resp.results[:2]:
        print(f"  chunk={r.chunk_id[:12]} doc={r.document_id[:12]} title={str(r.title or '')[:40]}")
except Exception as e:
    print(f"HYBRID_RERANK ERROR: {e}")
    traceback.print_exc()
