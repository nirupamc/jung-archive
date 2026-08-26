"""Chunk-size experiment support (M6).

Builds SEPARATE artifact/index namespaces under data/experiments/<name>/
so the production corpus is never destroyed:

    data/experiments/chunk_150/chunks/   chunk artifacts
    data/experiments/chunk_150/bm25/     lexical state
    data/experiments/chunk_150/chroma/   vector index

The evaluation runner accepts these namespaces via ExperimentConfig.
"""
import shutil
from pathlib import Path
from typing import Dict, Optional

from jung_archive.chunking.chunker import StructureAwareChunker
from jung_archive.chunking.validation import require_valid
from jung_archive.models.chunk import ChunkingConfig

EXPERIMENTS_ROOT = Path("data/experiments")


def build_experiment_corpus(
    pdf_path: str,
    name: str,
    target_tokens: int,
    max_tokens: int,
    min_tokens: int = 40,
    overlap_tokens: int = 25,
    root: Path = EXPERIMENTS_ROOT,
    force: bool = False,
) -> Dict[str, str]:
    """Chunk + index a PDF into an isolated experiment namespace.

    Returns the namespace paths for use in ExperimentConfig. If the
    namespace already exists and force is False, it is reused as-is.
    """
    ns = root / name
    chunks_dir = ns / "chunks"
    bm25_dir = ns / "bm25"
    chroma_dir = ns / "chroma"

    if (chunks_dir.exists() and any(chunks_dir.glob("*.json"))
            and not force):
        return {
            "chunks_dir": str(chunks_dir),
            "bm25_state_dir": str(bm25_dir),
            "chroma_dir": str(chroma_dir),
        }

    from jung_archive.chunking.artifacts import save_chunk_artifact
    from jung_archive.embedding.provider import LocalSentenceTransformerProvider
    from jung_archive.indexing.vector_index import VectorIndex
    from jung_archive.ingestion.pdf import PDFIngestor

    if ns.exists() and force:
        shutil.rmtree(ns)
    for d in (chunks_dir, bm25_dir, chroma_dir):
        d.mkdir(parents=True, exist_ok=True)

    ingestor = PDFIngestor(enable_ocr=False)
    document = ingestor.ingest(pdf_path)

    config = ChunkingConfig(
        target_tokens=target_tokens,
        max_tokens=max_tokens,
        min_tokens=min_tokens,
        overlap_tokens=overlap_tokens,
    )
    chunks = StructureAwareChunker(config).chunk_document(document)
    require_valid(chunks, document)
    save_chunk_artifact(chunks, document, config, str(chunks_dir))

    provider = LocalSentenceTransformerProvider()
    index = VectorIndex(provider, persist_dir=str(chroma_dir))
    report = index.index_chunks(
        chunks,
        source_sha256=document.source_sha256 or "",
        chunking_config_version=ChunkingConfig.CONFIG_VERSION,
        force=True,
    )

    return {
        "chunks_dir": str(chunks_dir),
        "bm25_state_dir": str(bm25_dir),
        "chroma_dir": str(chroma_dir),
        "chunk_count": str(report.get("indexed", 0)),
    }
