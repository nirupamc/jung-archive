#!/usr/bin/env python
"""
Jung Archive CLI - Document Inspection Tool
"""

import argparse
import json
import sys
from pathlib import Path

# Windows consoles often default to cp1252 and choke on typographic
# ligatures from the corpus; force UTF-8 with replacement.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from jung_archive.ingestion.pdf import PDFIngestor
from jung_archive.models.document import LayoutType, PageClassification


def main():
    parser = argparse.ArgumentParser(
        prog="jung-archive",
        description="Jung Archive Document Intelligence CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Inspect command
    inspect_parser = subparsers.add_parser("inspect", help="Inspect a PDF document")
    inspect_parser.add_argument("pdf_path", help="Path to PDF file")
    inspect_parser.add_argument("--ocr", action="store_true", help="Enable OCR")
    inspect_parser.add_argument("--output-dir", default="data", help="Output directory")

    # Chunk command
    chunk_parser = subparsers.add_parser(
        "chunk", help="Chunk a PDF into provenance-preserving chunks"
    )
    chunk_parser.add_argument("pdf_path", help="Path to PDF file")
    chunk_parser.add_argument("--ocr", action="store_true", help="Enable OCR")
    chunk_parser.add_argument("--output-dir", default="data", help="Output directory")
    chunk_parser.add_argument("--target-tokens", type=int, default=220)
    chunk_parser.add_argument("--max-tokens", type=int, default=300)
    chunk_parser.add_argument("--min-tokens", type=int, default=50)
    chunk_parser.add_argument("--overlap-tokens", type=int, default=30)

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Chunk (if needed) and embed a PDF into the vector index"
    )
    index_parser.add_argument("pdf_path", help="Path to PDF file")
    index_parser.add_argument("--ocr", action="store_true", help="Enable OCR")
    index_parser.add_argument("--output-dir", default="data", help="Output directory")
    index_parser.add_argument("--persist-dir", default="data/chroma")
    index_parser.add_argument("--force", action="store_true",
                              help="Re-index even if unchanged")

    # Search command (M3; M4 adds hybrid-rerank)
    search_parser = subparsers.add_parser(
        "search", help="Hybrid retrieval over the indexed corpus"
    )
    search_parser.add_argument("query", help="Query text")
    search_parser.add_argument("--mode",
                               choices=["dense", "bm25", "hybrid",
                                        "hybrid-rerank"],
                               default="hybrid")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--dense-k", type=int, default=20,
                               help="Dense candidate depth")
    search_parser.add_argument("--bm25-k", type=int, default=20,
                               help="BM25 candidate depth")
    search_parser.add_argument("--fusion-k", type=int, default=20,
                               help="Fused candidate pool size before reranking")
    search_parser.add_argument("--rrf-k", type=int, default=60)
    search_parser.add_argument("--reranker-model",
                               default=None,
                               help="Cross-encoder model for hybrid-rerank")
    search_parser.add_argument("--allow-no-reranker", action="store_true",
                               help="Fall back to unreranked ordering if the "
                                    "reranker fails (default: hard error)")
    search_parser.add_argument("--document-id", action="append", default=None,
                               help="Filter by document_id (repeatable)")
    search_parser.add_argument("--author", action="append", default=None,
                               help="Filter by author, enforced (repeatable)")
    search_parser.add_argument("--title", action="append", default=None,
                               help="Filter by title, enforced (repeatable)")
    search_parser.add_argument("--source-type", action="append", default=None,
                               choices=["PRIMARY", "SECONDARY", "UNKNOWN"],
                               help="Filter by source type (repeatable)")
    search_parser.add_argument("--persist-dir", default="data/chroma")
    search_parser.add_argument("--chunks-dir", default="data/chunks")

    # Evidence command (M4)
    evidence_parser = subparsers.add_parser(
        "evidence", help="Assemble a budgeted evidence pack for a query"
    )
    evidence_parser.add_argument("query", help="Question text")
    evidence_parser.add_argument("--max-tokens", type=int, default=2500)
    evidence_parser.add_argument("--max-items", type=int, default=8)
    evidence_parser.add_argument("--top-k", type=int, default=8,
                                 help="Reranked evidence candidates considered")
    evidence_parser.add_argument("--dense-k", type=int, default=30)
    evidence_parser.add_argument("--bm25-k", type=int, default=30)
    evidence_parser.add_argument("--fusion-k", type=int, default=20)
    evidence_parser.add_argument("--rrf-k", type=int, default=60)
    evidence_parser.add_argument("--reranker-model", default=None)
    evidence_parser.add_argument("--document-id", action="append", default=None)
    evidence_parser.add_argument("--author", action="append", default=None)
    evidence_parser.add_argument("--title", action="append", default=None)
    evidence_parser.add_argument("--source-type", action="append", default=None,
                                 choices=["PRIMARY", "SECONDARY", "UNKNOWN"])
    evidence_parser.add_argument("--render", action="store_true",
                                 help="Print the rendered prompt-style pack")
    evidence_parser.add_argument("--json", action="store_true",
                                 help="Also print the structured pack as JSON")
    evidence_parser.add_argument("--persist-dir", default="data/chroma")
    evidence_parser.add_argument("--chunks-dir", default="data/chunks")

    # Evaluation Lab commands (M6)
    eval_parser = subparsers.add_parser(
        "eval", help="Evaluation Lab: benchmarks, comparisons, artifacts")
    eval_sub = eval_parser.add_subparsers(dest="eval_command", required=True)

    er = eval_sub.add_parser(
        "retrieval", help="Run retrieval/evidence benchmark on a dataset")
    er.add_argument("--dataset", default="data/evaluation/dataset.json")
    er.add_argument("--mode", action="append",
                    choices=["dense", "bm25", "hybrid", "hybrid-rerank"],
                    default=None,
                    help="Modes to evaluate (default: all four)")
    er.add_argument("--k", action="append", type=int, default=None,
                    help="K values (default: 1 3 5 10)")
    er.add_argument("--run-name", default="cli-run")
    er.add_argument("--dataset-version-note", default="")
    er.add_argument("--dense-k", type=int, default=30)
    er.add_argument("--bm25-k", type=int, default=30)
    er.add_argument("--rrf-k", type=int, default=60)
    er.add_argument("--fusion-k", type=int, default=20)
    er.add_argument("--rerank-top-k", type=int, default=10)
    er.add_argument("--chunks-dir", default="data/chunks")
    er.add_argument("--chroma-dir", default="data/chroma")
    er.add_argument("--bm25-state-dir", default="data/bm25")
    er.add_argument("--output-dir", default="data/evaluation")
    er.add_argument("--skip-evidence", action="store_true")

    ec = eval_sub.add_parser("compare", help="Compare two evaluation runs")
    ec.add_argument("run_a")
    ec.add_argument("run_b")
    ec.add_argument("--runs-dir", default="data/evaluation/runs")

    ex = eval_sub.add_parser(
        "chunksize",
        help="Build an isolated chunk-size namespace and benchmark it "
             "(page-level relevance; chunk labels move with boundaries)")
    ex.add_argument("--pdf", default="primary/The Undiscovered Self.pdf")
    ex.add_argument("--name", default=None,
                    help="Namespace name (default: chunk_<target>)")
    ex.add_argument("--target-tokens", type=int, default=150)
    ex.add_argument("--max-tokens", type=int, default=None,
                    help="Default: target + 80")
    ex.add_argument("--mode", action="append",
                    choices=["dense", "bm25", "hybrid", "hybrid-rerank"],
                    default=None)
    ex.add_argument("--k", action="append", type=int, default=None)
    ex.add_argument("--dataset", default="data/evaluation/dataset.json")
    ex.add_argument("--output-dir", default="data/evaluation")
    ex.add_argument("--force", action="store_true")

    # Knowledge graph commands (M7)
    g_parser = subparsers.add_parser(
        "graph", help="Evidence-backed knowledge graph")
    g_sub = g_parser.add_subparsers(dest="graph_command", required=True)
    gb = g_sub.add_parser("build", help="Build + persist the graph")
    gb.add_argument("--chunks-dir", default="data/chunks")
    gb.add_argument("--output-dir", default="data/graph")

    # Corpus discovery + batch ingestion (post-M7)
    c_parser = subparsers.add_parser(
        "corpus", help="Discover and batch-ingest the whole corpus")
    c_sub = c_parser.add_subparsers(dest="corpus_command", required=True)
    cl = c_sub.add_parser("list", help="List every discovered PDF + status")
    cl.add_argument("--json", action="store_true",
                    help="Emit machine-readable JSON")
    ci = c_sub.add_parser(
        "ingest",
        help="Batch ingest every registry-approved INCLUDE document")
    ci.add_argument("--sections", default="primary,secondary",
                    help="Comma-separated sections to scan")
    ci.add_argument("--force", action="store_true",
                    help="Re-embed even when index state is unchanged")
    ci.add_argument("--limit", type=int, default=None,
                    help="Process at most N candidate documents")
    ci.add_argument("--dry-run", action="store_true",
                    help="Show what would be ingested; do nothing")

    args = parser.parse_args()

    if args.command == "inspect":
        inspect_command(args)
    elif args.command == "chunk":
        chunk_command(args)
    elif args.command == "index":
        index_command(args)
    elif args.command == "search":
        search_command(args)
    elif args.command == "evidence":
        evidence_command(args)
    elif args.command == "eval":
        if args.eval_command == "retrieval":
            eval_retrieval_command(args)
        elif args.eval_command == "compare":
            eval_compare_command(args)
        else:
            eval_chunksize_command(args)
    elif args.command == "graph":
        graph_build_command(args)
    elif args.command == "corpus":
        if args.corpus_command == "list":
            corpus_list_command(args)
        else:
            corpus_ingest_command(args)
    else:
        parser.print_help()
        sys.exit(1)


def inspect_command(args):
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    # Create output directories
    processed_dir = Path(args.output_dir) / "processed"
    diagnostics_dir = Path(args.output_dir) / "diagnostics"
    processed_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # Ingest the document
    ingestor = PDFIngestor(enable_ocr=args.ocr)
    document = ingestor.ingest(str(pdf_path))

    # Save canonical JSON
    output_json = processed_dir / f"{document.document_id}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)

    # Generate and save diagnostics
    diagnostics = generate_diagnostics(document)
    diagnostics_json = diagnostics_dir / f"{document.document_id}.json"
    with open(diagnostics_json, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)

    # Print human-readable output
    print_inspection(document, output_json, diagnostics_json, diagnostics)


def generate_diagnostics(document):
    """Generate diagnostics from document."""
    from jung_archive.models.document import BlockType

    # Zero-fill known categories so reports are stable/comparable;
    # observed counts always come from actual processing.
    classification_counts = {c.value: 0 for c in PageClassification}
    layout_counts = {l.value: 0 for l in LayoutType}
    block_counts = {b.value: 0 for b in BlockType}
    measured_confidences = []
    warnings = []

    for page in document.pages:
        classification_counts[page.classification.value] += 1
        layout_counts[page.layout.value] += 1

        for block in page.blocks:
            block_counts[block.block_type.value] += 1
            if block.confidence is not None:  # measured values only
                measured_confidences.append(block.confidence)

        warnings.extend(page.warnings)

    avg_confidence = (
        round(sum(measured_confidences) / len(measured_confidences), 4)
        if measured_confidences
        else None
    )

    return {
        "document_id": document.document_id,
        "page_count": document.page_count,
        "classification_counts": classification_counts,
        "layout_counts": layout_counts,
        "block_counts": block_counts,
        "total_block_count": sum(block_counts.values()),
        "average_extraction_confidence": avg_confidence,
        "measured_confidence_block_count": len(measured_confidences),
        "warnings": warnings,
    }


def print_inspection(document, output_json, diagnostics_json, diagnostics):
    """Print human-readable inspection output."""
    print("=" * 50)
    print("JUNG ARCHIVE - DOCUMENT INSPECTION")
    print("=" * 50)
    print()
    print(f"Document: {document.title}")
    print(f"Pages: {document.page_count}")
    print()

    print("Classification")
    for cls in PageClassification:
        print(f"  {cls.value:<20} {diagnostics['classification_counts'][cls.value]}")
    print()

    print("Layout")
    for layout in LayoutType:
        print(f"  {layout.value:<20} {diagnostics['layout_counts'][layout.value]}")
    print()

    print("Blocks")
    for block_type, count in sorted(diagnostics["block_counts"].items()):
        print(f"  {block_type:<20} {count}")
    print()

    print(f"Average extraction confidence: {diagnostics['average_extraction_confidence']} "
          f"(measured blocks: {diagnostics['measured_confidence_block_count']})")
    print()

    print(f"Canonical output: {output_json}")
    print(f"Diagnostics: {diagnostics_json}")


def _ingest_or_load(pdf_path: str, ocr: bool, output_dir: str):
    """Ingest the PDF and persist canonical + diagnostics outputs."""
    from jung_archive.cli import generate_diagnostics as gd  # local alias

    processed_dir = Path(output_dir) / "processed"
    diagnostics_dir = Path(output_dir) / "diagnostics"
    processed_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    ingestor = PDFIngestor(enable_ocr=ocr)
    document = ingestor.ingest(pdf_path)

    output_json = processed_dir / f"{document.document_id}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(document.to_dict(), f, indent=2, ensure_ascii=False)
    diagnostics = gd(document)
    with open(diagnostics_dir / f"{document.document_id}.json", "w",
              encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, ensure_ascii=False)
    return document, diagnostics


def chunk_command(args):
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    document, diagnostics = _ingest_or_load(str(pdf_path), args.ocr,
                                            args.output_dir)

    from jung_archive.chunking.artifacts import save_chunk_artifact
    from jung_archive.chunking.chunker import StructureAwareChunker
    from jung_archive.chunking.tokenizer import active_counter_name
    from jung_archive.chunking.validation import validate_chunks
    from jung_archive.models.chunk import ChunkingConfig

    config = ChunkingConfig(
        target_tokens=args.target_tokens,
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    chunks = StructureAwareChunker(config).chunk_document(document)
    validation = validate_chunks(chunks, document)

    if document.index_status.value == "EXCLUDE":
        print("Document index status is EXCLUDE; chunk artifact not written.")
        print(f"Reason: see config/document_metadata.json")
        sys.exit(2)

    artifact_path = save_chunk_artifact(chunks, document, config,
                                        str(Path(args.output_dir) / "chunks"))

    token_counts = [c.token_count for c in chunks]
    multi_page = sum(1 for c in chunks if c.start_page != c.end_page)

    print("=" * 50)
    print("JUNG ARCHIVE - CHUNKING")
    print("=" * 50)
    print()
    print(f"Document: {document.title}")
    print(f"  author      : {document.author or '(unverified)'}")
    print(f"  source_type : {document.source_type.value}")
    print(f"  index_status: {document.index_status.value}")
    print(f"Pages: {document.page_count}")
    print(f"Blocks: {diagnostics['total_block_count']}")
    print()
    print("Chunks")
    print(f"  count          : {len(chunks)}")
    print(f"  token range    : {min(token_counts) if token_counts else 0}"
          f" - {max(token_counts) if token_counts else 0}")
    print(f"  average tokens : "
          f"{round(sum(token_counts) / len(token_counts)) if token_counts else 0}")
    print(f"  multi-page     : {multi_page}")
    print(f"  pages covered  : "
          f"{len({p for c in chunks for p in c.page_numbers})}")
    print()
    print(f"Tokenizer           : {active_counter_name()}")
    print(f"Chunking config     : v{ChunkingConfig.CONFIG_VERSION}")
    print(f"Provenance validation: {'PASS' if validation.ok else 'FAIL'}")
    if not validation.ok:
        for err in validation.errors[:10]:
            print(f"  ! {err}")
    print()
    print(f"Chunk artifact: {artifact_path}")
    sys.exit(0 if validation.ok else 3)


def index_command(args):
    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}", file=sys.stderr)
        sys.exit(1)

    document, diagnostics = _ingest_or_load(str(pdf_path), args.ocr,
                                            args.output_dir)

    from jung_archive.chunking.artifacts import save_chunk_artifact
    from jung_archive.chunking.chunker import StructureAwareChunker
    from jung_archive.chunking.validation import validate_chunks
    from jung_archive.embedding.provider import LocalSentenceTransformerProvider
    from jung_archive.indexing.vector_index import VectorIndex
    from jung_archive.models.chunk import ChunkingConfig

    if document.index_status == "EXCLUDE":
        print(f"Document '{document.title}' is EXCLUDEd from the index; aborting.")
        sys.exit(2)
    if document.index_status == "REVIEW":
        print(f"Document '{document.title}' has index status REVIEW.")
        print("Register an explicit decision in config/document_metadata.json "
              "before indexing.")
        sys.exit(4)

    config = ChunkingConfig()
    chunks = StructureAwareChunker(config).chunk_document(document)
    validation = validate_chunks(chunks, document)
    if not validation.ok:
        print("Provenance validation FAILED; refusing to index:")
        for err in validation.errors[:10]:
            print(f"  ! {err}")
        sys.exit(3)

    save_chunk_artifact(chunks, document, config,
                        str(Path(args.output_dir) / "chunks"))

    provider = LocalSentenceTransformerProvider()
    index = VectorIndex(provider, persist_dir=args.persist_dir)

    report = index.index_chunks(
        chunks,
        source_sha256=document.source_sha256 or "",
        chunking_config_version=ChunkingConfig.CONFIG_VERSION,
        force=args.force,
    )

    meta = index.collection_metadata()
    state = index.load_state().get("documents", {}).get(document.document_id, {})

    print("=" * 50)
    print("JUNG ARCHIVE - INDEXING")
    print("=" * 50)
    print()
    print(f"Document: {document.title}")
    print(f"  source_type : {document.source_type.value}")
    print(f"  sha256      : {(document.source_sha256 or '')[:16]}...")
    print()
    print(f"Index status : {document.index_status.value}")
    print(f"Chunks indexed: {report.get('indexed', 0)}"
          f"{' (skipped: ' + report['skipped'] + ')' if report.get('indexed') == 0 else ''}")
    print(f"Embedding model: {provider.model_name}")
    print(f"Dimensions   : {meta.get('embedding_dimension')}")
    print(f"Normalized   : {meta.get('normalized')}")
    print(f"Duplicates   : 0 (deterministic upsert)")
    print(f"Persistent collection: {args.persist_dir}/{index.collection_name}")
    print(f"Total vectors in collection: {index.count()}")


def _build_filters(args):
    filters = {}
    if getattr(args, "document_id", None):
        filters["document_id"] = args.document_id
    if getattr(args, "author", None):
        filters["author"] = args.author
    if getattr(args, "title", None):
        filters["title"] = args.title
    if getattr(args, "source_type", None):
        filters["source_type"] = args.source_type
    return filters


def search_command(args):
    from jung_archive.embedding.provider import LocalSentenceTransformerProvider
    from jung_archive.indexing.vector_index import VectorIndex
    from jung_archive.retrieval.hybrid import HybridRetriever, HybridRetrieverConfig
    from jung_archive.retrieval.lexical import BM25Retriever

    if not args.query.strip():
        print("Error: empty query", file=sys.stderr)
        sys.exit(1)
    if args.top_k < 1:
        print("Error: --top-k must be >= 1", file=sys.stderr)
        sys.exit(1)

    filters = _build_filters(args)

    provider = LocalSentenceTransformerProvider()
    vi = VectorIndex(provider, persist_dir=args.persist_dir)
    bm25 = BM25Retriever(chunks_dir=args.chunks_dir,
                         state_dir=str(Path(args.persist_dir).parent / "bm25"))

    rerank_mode = args.mode == "hybrid-rerank"
    try:
        if rerank_mode:
            from jung_archive.reranking.cross_encoder import \
                LocalCrossEncoderReranker
            from jung_archive.retrieval.pipeline import (
                RerankingPipeline,
                RerankingPipelineConfig,
            )

            reranker_kwargs = {}
            if args.reranker_model:
                reranker_kwargs["model_name"] = args.reranker_model
            reranker = LocalCrossEncoderReranker(**reranker_kwargs)
            config = RerankingPipelineConfig(
                dense_candidate_k=max(args.dense_k, 30),
                bm25_candidate_k=max(args.bm25_k, 30),
                rrf_k=args.rrf_k,
                fusion_candidate_k=max(args.fusion_k, args.top_k),
                rerank_top_k=args.top_k,
                allow_reranker_fallback=args.allow_no_reranker,
            )
            pipeline = RerankingPipeline(vi, bm25, reranker, config)
            response = pipeline.search(args.query, top_k=args.top_k,
                                       filters=filters)
        else:
            config = HybridRetrieverConfig(
                dense_candidate_k=args.dense_k,
                bm25_candidate_k=args.bm25_k,
                rrf_k=args.rrf_k,
                final_top_k=args.top_k,
                mode=args.mode.replace("-", "_"),
            )
            retriever = HybridRetriever(vi, bm25, config)
            response = retriever.search(args.query, top_k=args.top_k,
                                        filters=filters)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    print("=" * 50)
    print("JUNG ARCHIVE - RETRIEVAL")
    print("=" * 50)
    print()
    print("Query")
    print(f"  {args.query}")
    print()
    print(f"Mode: {response.mode.upper()}   Top-K: {response.top_k}")
    for w in response.warnings:
        print(f"  warning: {w}")
    if response.latency_ms is not None:
        print(f"Latency: {response.latency_ms:.1f} ms")
    if response.candidates_retrieved is not None:
        print(f"Candidates retrieved: {response.candidates_retrieved}"
              f"   Reranked: {response.candidates_reranked}"
              + (f"   Pairs truncated: {response.pairs_truncated}"
                 if response.pairs_truncated else ""))
    print()
    print("Results")
    print("-" * 40)
    if not response.results:
        print("  (no results)")
    for res in response.results:
        label = res.reranker_rank if res.reranker_rank is not None \
            else res.fusion_rank
        print(f"\n#{label}")
        print(f"Chunk      : {res.chunk_id}")
        print(f"Document   : {res.document_id}"
              + (f" ({res.title})" if res.title else ""))
        print(f"Pages      : {res.page_numbers}")
        if res.dense_rank is not None:
            print(f"Dense      : rank {res.dense_rank}, score {res.dense_score}")
        else:
            print("Dense      : -")
        if res.bm25_rank is not None:
            print(f"BM25       : rank {res.bm25_rank}, score {res.bm25_score}")
        else:
            print("BM25       : -")
        fusion = f"{res.fusion_score}" if res.fusion_score is not None \
            else "(single-leg ordering)"
        print(f"Fusion     : rank {res.fusion_rank}, score {fusion}")
        if res.reranker_rank is not None:
            print(f"Reranker   : rank {res.reranker_rank}, "
                  f"score {res.reranker_score}")
        else:
            print("Reranker   : -")
        print(f"Blocks     : {len(res.source_block_ids)} source blocks")
        print(f"Preview    : {res.preview()}")


def evidence_command(args):
    from jung_archive.embedding.provider import LocalSentenceTransformerProvider
    from jung_archive.evidence import EvidenceAssembler, EvidenceConfig, \
        render_evidence_pack
    from jung_archive.indexing.vector_index import VectorIndex
    from jung_archive.retrieval.lexical import BM25Retriever
    from jung_archive.retrieval.pipeline import (
        RerankingPipeline,
        RerankingPipelineConfig,
    )
    from jung_archive.reranking.cross_encoder import LocalCrossEncoderReranker

    if not args.query.strip():
        print("Error: empty query", file=sys.stderr)
        sys.exit(1)
    if args.max_tokens < 1 or args.max_items < 1 or args.top_k < 1:
        print("Error: --max-tokens/--max-items/--top-k must be >= 1",
              file=sys.stderr)
        sys.exit(1)

    filters = _build_filters(args)

    provider = LocalSentenceTransformerProvider()
    vi = VectorIndex(provider, persist_dir=args.persist_dir)
    bm25 = BM25Retriever(chunks_dir=args.chunks_dir,
                         state_dir=str(Path(args.persist_dir).parent / "bm25"))

    reranker_kwargs = {}
    if args.reranker_model:
        reranker_kwargs["model_name"] = args.reranker_model
    reranker = LocalCrossEncoderReranker(**reranker_kwargs)
    pipeline_config = RerankingPipelineConfig(
        dense_candidate_k=args.dense_k,
        bm25_candidate_k=args.bm25_k,
        rrf_k=args.rrf_k,
        fusion_candidate_k=max(args.fusion_k, args.top_k),
        rerank_top_k=args.top_k,
    )
    pipeline = RerankingPipeline(vi, bm25, reranker, pipeline_config)

    try:
        response = pipeline.search(args.query, top_k=args.top_k,
                                   filters=filters)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    assembler = EvidenceAssembler(EvidenceConfig(
        max_evidence_tokens=args.max_tokens,
        max_evidence_items=args.max_items,
    ))
    pack = assembler.assemble(args.query, response.results)

    print("=" * 50)
    print("JUNG ARCHIVE - EVIDENCE PACK")
    print("=" * 50)
    print()
    print("Query")
    print(f"  {args.query}")
    for w in response.warnings + pack.warnings:
        print(f"  warning: {w}")
    print()
    print(f"Candidates retrieved : {response.candidates_retrieved}")
    print(f"Candidates reranked  : {response.candidates_reranked}")
    if response.pairs_truncated:
        print(f"Pairs truncated      : {response.pairs_truncated}")
    print()
    print("Final evidence")
    print("-" * 40)
    if not pack.items:
        print("  (no evidence assembled)")
    for item in pack.items:
        s = item.scores
        print(f"\n{item.evidence_id}")
        print(f"Chunk       : {item.chunk_id}")
        print(f"Document    : {item.title or item.document_id}"
              f" ({item.document_id})")
        print(f"Pages       : {item.pages_display()}")
        print(f"Fusion rank : {s.fusion_rank}"
              + (f" (score {s.fusion_score})" if s.fusion_score else ""))
        print(f"Reranker    : rank {s.reranker_rank}, score {s.reranker_score}")
        print(f"Token count : {item.token_count}")
        if item.was_cleaned:
            print(f"Cleanup     : {', '.join(item.cleanup_operations)}")
        print(f"Preview     : {item.preview()}")
    print()
    print(f"Evidence budget : {pack.tokens_used} / {pack.max_evidence_tokens} tokens"
          f"   items: {len(pack.items)} / {pack.max_evidence_items}")
    print(f"Suppressed duplicates : {len(pack.suppressed_duplicates)}")
    for sup in pack.suppressed_duplicates:
        print(f"  - {sup.chunk_id}: {sup.reason}")
    if pack.suppressed_diversity:
        print(f"Suppressed (diversity): {len(pack.suppressed_diversity)}")
        for sup in pack.suppressed_diversity:
            print(f"  - {sup.chunk_id}: {sup.reason}")
    if pack.skipped_oversized:
        print(f"Skipped oversized      : {len(pack.skipped_oversized)}")
        for sup in pack.skipped_oversized:
            print(f"  - {sup.chunk_id}: {sup.reason}")

    if args.render:
        print()
        print("=" * 50)
        print("RENDERED EVIDENCE PACK")
        print("=" * 50)
        print()
        print(render_evidence_pack(pack), end="")
    if args.json:
        import json as _json

        print()
        print(_json.dumps(pack.to_dict(), indent=2, ensure_ascii=False))


def _print_aggregate_table(aggregates):
    from jung_archive.evaluation.runner import summary_dict  # noqa: F401

    print()
    header = (f"{'Mode':<18} {'Hit@1':>6} {'Recall@5':>9} "
              f"{'MRR':>7} {'NDCG@5':>8} {'Recall@10':>10}")
    print(header)
    print("-" * len(header))
    for agg in aggregates:
        cm = agg.chunk_metrics
        print(f"{agg.mode:<18} "
              f"{cm.hit_at_k.get('1', 0):>6.3f} "
              f"{cm.recall_at_k.get('5', 0):>9.3f} "
              f"{cm.mrr:>7.3f} "
              f"{cm.ndcg_at_k.get('5', 0):>8.3f} "
              f"{cm.recall_at_k.get('10', 0):>10.3f}")


def corpus_list_command(args):
    from jung_archive.corpus import PipelineStatus, corpus_report, \
        discover_corpus

    docs = discover_corpus()
    if args.json:
        import json as _json

        print(_json.dumps({
            "report": corpus_report(docs),
            "documents": [d.__dict__ for d in docs],
        }, indent=2, ensure_ascii=False))
        return

    rep = corpus_report(docs)
    print("=" * 50)
    print("JUNG ARCHIVE - CORPUS DISCOVERY")
    print("=" * 50)
    print()
    print(f"discovered : {rep['discovered_total']} PDFs "
          f"({rep['pages_total']} pages)")
    for section, n in rep["by_section"].items():
        print(f"  {section:<10}: {n}")
    print()
    print("status counts:")
    for s in PipelineStatus:
        print(f"  {s.value:<11}: {rep['by_status'][s.value]}")
    print()
    for d in docs:
        line = (f"[{d.status:<10}] {d.section:<9} {d.path}  "
                f"({d.page_count} pp")
        if not d.registered:
            line += ", UNREGISTERED"
        line += ")"
        print(line)


def corpus_ingest_command(args):
    from jung_archive.corpus import discover_corpus
    from jung_archive.ingestion.batch import ingest_batch, save_batch_report

    sections = [s.strip().lower() for s in args.sections.split(",")
                if s.strip()]
    if args.dry_run:
        docs = discover_corpus(sections=sections)
        candidates = [d for d in docs
                      if d.index_status == "INCLUDE"
                      and d.status != "ERROR"]
        print("DRY RUN - would ingest:")
        for d in candidates[:args.limit or len(candidates)]:
            print(f"  [{d.status:<9}] {d.path}")
        held = [d for d in docs if d.index_status != "INCLUDE"]
        print(f"\ncandidates: {len(candidates)}   held back: {len(held)} "
              f"(REVIEW/EXCLUDE/UNKNOWN never touched)")
        return

    print("=" * 50)
    print("JUNG ARCHIVE - BATCH INGESTION")
    print("=" * 50)
    report = ingest_batch(
        sections=sections,
        force_index=args.force,
        limit=args.limit,
        progress=lambda msg: print(msg),
    )
    path = save_batch_report(report)

    print()
    print("-" * 50)
    ok = len(report["processed_ok"])
    print(f"candidates          : {report['candidates']}")
    print(f"processed ok        : {ok}")
    print(f"artifacts reused    : {report['artifacts_reused']}")
    print(f"freshly ingested    : {report['freshly_ingested']}")
    print(f"held back           : {len(report.get('held_back', []))} "
          f"(REVIEW/EXCLUDE/UNKNOWN never touched)")
    print(f"skipped             : {len(report['skipped'])}")
    print(f"failed              : {len(report['failed'])}")
    t = report["totals"]
    print(f"pages/blocks/chunks : {t['pages']} / {t['blocks']} / "
          f"{t['chunks']}")
    print(f"vectors embedded    : {t['vectors_indexed']} "
          f"(unchanged docs: {t['index_unchanged']})")
    print(f"collection total    : {report['collection_vectors_after']} vectors")
    print(f"elapsed             : {report['elapsed_s']}s")
    print(f"report              : {path}")
    if report["skipped"]:
        print("\nSkipped documents (honest ledger):")
        for s in report["skipped"]:
            print(f"  - {s['path']}: {s['reason']}")
    if report["failed"]:
        print("\nFailed documents:")
        for f_ in report["failed"]:
            print(f"  - {f_['path']}: {f_['error']}")


def graph_build_command(args):
    from collections import Counter

    from jung_archive.evaluation.runner import list_runs  # noqa: F401
    from jung_archive.graph.build import build_graph, save_graph, \
        staleness_report
    from jung_archive.graph.vocabulary import Vocabulary

    print("=" * 50)
    print("JUNG ARCHIVE - GRAPH BUILD")
    print("=" * 50)
    vocab = Vocabulary()
    print(f"vocabulary : {vocab.version} "
          f"({len(vocab.concepts)} concepts, "
          f"{len(vocab.alias_to_canonical)} aliases)")
    print("extracting evidence-backed relations ...")
    graph = build_graph(chunks_dir=args.chunks_dir)
    errors = graph.validate()
    if errors:
        print("VALIDATION FAILED; refusing to persist:", file=sys.stderr)
        for e in errors[:10]:
            print(f"  ! {e}", file=sys.stderr)
        sys.exit(3)
    path = save_graph(graph, Path(args.output_dir))
    stale = staleness_report(graph, args.chunks_dir)

    statuses = Counter(e.status for e in graph.edges)
    relations = Counter(e.relationship_type for e in graph.edges)
    print()
    print(f"documents/chunks : {len(set(e.document_id for e in graph.evidence))}"
          f" docs analyzed")
    print(f"nodes            : {len(graph.nodes)}")
    print(f"edges            : {len(graph.edges)}")
    for status in ("TRUSTED", "WEAK", "UNVERIFIED"):
        print(f"  {status:<11}: {statuses.get(status, 0)}")
    print("relationships:")
    for rel, n in relations.most_common():
        print(f"  {rel:<16}: {n}")
    print(f"evidence spans   : {len(graph.evidence)}")
    trusted = [e for e in graph.edges if e.status == 'TRUSTED']
    if trusted:
        avg = sum(e.evidence_count for e in trusted) / len(trusted)
        print(f"avg evidence per trusted edge: {avg:.1f}")
    print(f"build time       : {graph.state.build_time_s}s")
    print(f"stale            : {'yes - ' + '; '.join(stale) if stale else 'no'}")
    print(f"persisted        : {path}")


def eval_retrieval_command(args):
    from jung_archive.evaluation.dataset import load_dataset, require_valid
    from jung_archive.evaluation.models import ExperimentConfig
    from jung_archive.evaluation.runner import ProductionRetrieverFactory, \
        run_benchmark

    modes = [m.replace("-", "_") for m in (args.mode or
                                           ["dense", "bm25", "hybrid",
                                            "hybrid-rerank"])]
    ks = args.k or [1, 3, 5, 10]

    dataset = load_dataset(args.dataset)
    require_valid(dataset, args.chunks_dir)

    config = ExperimentConfig(
        run_name=args.run_name,
        modes=modes,
        k_values=ks,
        dense_candidate_k=args.dense_k,
        bm25_candidate_k=args.bm25_k,
        rrf_k=args.rrf_k,
        fusion_candidate_k=args.fusion_k,
        rerank_top_k=args.rerank_top_k,
        chunks_dir=args.chunks_dir,
        chroma_dir=args.chroma_dir,
        bm25_state_dir=args.bm25_state_dir,
        notes=args.dataset_version_note,
    )

    print("=" * 50)
    print("JUNG ARCHIVE - EVALUATION RUN")
    print("=" * 50)
    print(f"dataset   : {args.dataset}")
    print(f"version   : {dataset.meta.dataset_version} "
          f"(fingerprint {dataset.fingerprint()})")
    print(f"questions : {len(dataset.items)}")
    print(f"modes     : {', '.join(modes)}")
    print(f"k values  : {ks}")
    if "hybrid_rerank" in modes:
        print("loading models (embedding + cross-encoder) ...")

    factory = ProductionRetrieverFactory(config)
    record = run_benchmark(
        dataset, factory, config,
        output_dir=args.output_dir,
        include_evidence_eval=not args.skip_evidence,
        run_id=f"{config.config_hash()}-"
               f"{Path(args.dataset).stem}",
    )
    record.dataset_path = str(args.dataset)

    print(f"\nrun_id: {record.run_id}")
    _print_aggregate_table(record.aggregates)
    counts = {}
    for f in record.failures:
        counts[f.category] = counts.get(f.category, 0) + 1
    print("\nFailure categories:")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    if record.evidence_evals:
        n = len(record.evidence_evals)
        acc = sum(e.evidence_accuracy_chunk for e in record.evidence_evals) / n
        cov = sum(e.evidence_coverage_chunk for e in record.evidence_evals) / n
        print(f"\nEvidence (reranked path): accuracy(chunk)={acc:.3f} "
              f"coverage(chunk)={cov:.3f} over {n} questions")
    print(f"\nTotal time: {record.total_time_s}s "
          f"(avg {record.avg_query_time_ms} ms/query)")
    print(f"Artifacts: {args.output_dir}/runs/{record.run_id}.json")


def eval_chunksize_command(args):
    from jung_archive.evaluation.dataset import load_dataset, require_valid
    from jung_archive.evaluation.experiments import build_experiment_corpus
    from jung_archive.evaluation.models import ExperimentConfig
    from jung_archive.evaluation.runner import ProductionRetrieverFactory, \
        run_benchmark

    modes = [m.replace("-", "_") for m in (args.mode or
                                           ["bm25", "dense", "hybrid"])]
    ks = args.k or [1, 3, 5, 10]
    name = args.name or f"chunk_{args.target_tokens}"
    max_tokens = args.max_tokens or (args.target_tokens + 80)

    print("=" * 50)
    print(f"JUNG ARCHIVE - CHUNK-SIZE EXPERIMENT ({name})")
    print("=" * 50)
    print(f"building namespace: data/experiments/{name} "
          f"(target {args.target_tokens} tokens) ...")
    ns = build_experiment_corpus(
        args.pdf, name, args.target_tokens, max_tokens, force=args.force)
    print(f"namespace ready: chunk_count={ns.get('chunk_count', 'reused')}")

    dataset = load_dataset(args.dataset)
    # Chunk labels move when boundaries move: validate with relaxed
    # chunk checks; this experiment is evaluated at PAGE relevance level.
    require_valid(dataset, ns["chunks_dir"], ignore_chunk_labels=True)

    config = ExperimentConfig(
        run_name=f"chunksize-{name}",
        modes=modes,
        k_values=ks,
        chunks_dir=ns["chunks_dir"],
        chroma_dir=ns["chroma_dir"],
        bm25_state_dir=ns["bm25_state_dir"],
        notes="chunk-size experiment; page-level relevance only "
              "(chunk ids differ from ground truth)",
    )
    factory = ProductionRetrieverFactory(config)
    record = run_benchmark(
        dataset, factory, config,
        output_dir=args.output_dir,
        include_evidence_eval=False,
        run_id=f"{config.config_hash()}-{name}",
    )
    record.dataset_path = args.dataset

    print(f"\nrun_id: {record.run_id}")
    _print_aggregate_table(record.aggregates)
    print("\nPAGE-LEVEL metrics (primary for this experiment):")
    header = (f"{'Mode':<18} {'Hit@5':>7} {'Recall@5':>9} "
              f"{'MRR':>7} {'NDCG@5':>8}")
    print(header)
    print("-" * len(header))
    for agg in record.aggregates:
        pm = agg.page_metrics
        print(f"{agg.mode:<18} "
              f"{pm.hit_at_k.get('5', 0):>7.3f} "
              f"{pm.recall_at_k.get('5', 0):>9.3f} "
              f"{pm.mrr:>7.3f} "
              f"{pm.ndcg_at_k.get('5', 0):>8.3f}")
    print(f"\nTotal time: {record.total_time_s}s "
          f"(avg {record.avg_query_time_ms} ms/query)")


def eval_compare_command(args):
    from jung_archive.evaluation.runner import load_run

    path_a = Path(args.runs_dir) / f"{args.run_a}.json"
    path_b = Path(args.runs_dir) / f"{args.run_b}.json"
    a = load_run(str(path_a))
    b = load_run(str(path_b))

    def by_mode(rec):
        return {agg.mode: agg for agg in rec.aggregates}

    ma, mb = by_mode(a), by_mode(b)
    common = [m for m in ma if m in mb]
    if not common:
        print("Error: runs share no common modes", file=sys.stderr)
        sys.exit(1)

    print("=" * 50)
    print("JUNG ARCHIVE - RUN COMPARISON")
    print("=" * 50)
    print(f"A: {a.run_id} ({a.timestamp})")
    print(f"B: {b.run_id} ({b.timestamp})")
    ks = sorted({int(k) for agg in a.aggregates
                 for k in agg.chunk_metrics.hit_at_k})
    for mode in common:
        aa, ab = ma[mode].chunk_metrics, mb[mode].chunk_metrics
        print(f"\n{mode.upper()}  ({a.run_id} -> {b.run_id})")
        print(f"{'metric':<14}"
              + "".join(f"{'@' + str(k):>10}" for k in ks)
              + f"{'MRR':>10}")
        for metric in ("hit_at_k", "recall_at_k", "precision_at_k",
                       "ndcg_at_k"):
            va = getattr(aa, metric)
            vb = getattr(ab, metric)
            deltas = [vb.get(str(k), 0) - va.get(str(k), 0) for k in ks]
            print(f"{metric:<14}"
                  + "".join(f"{d:>+10.3f}" for d in deltas)
                  + f"{ab.mrr - aa.mrr:>+10.3f}")


if __name__ == "__main__":
    main()