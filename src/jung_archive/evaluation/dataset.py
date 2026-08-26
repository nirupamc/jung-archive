"""Benchmark dataset loading and integrity validation (M6).

Validation is strict: evaluation must FAIL CLEARLY on stale or invalid
labels instead of silently producing numbers.
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from jung_archive.chunking.chunker import StructureAwareChunker  # noqa: F401 (version import)
from jung_archive.evaluation.models import BenchmarkDataset, DatasetMeta
from jung_archive.models.chunk import ChunkingConfig


class DatasetValidationError(Exception):
    """Raised when a benchmark dataset fails integrity validation."""


def load_dataset(path: str) -> BenchmarkDataset:
    p = Path(path)
    if not p.exists():
        raise DatasetValidationError(f"dataset not found: {path}")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    # tolerate legacy flat format {meta:{}, items:[]}
    return BenchmarkDataset(**data)


def load_corpus(chunks_dir: str) -> Dict[str, dict]:
    """document_id-level registry + chunk map from chunk artifacts."""
    from jung_archive.retrieval.lexical import BM25Retriever

    retriever = BM25Retriever(chunks_dir=chunks_dir,
                              state_dir=str(Path(chunks_dir).parent / "bm25_eval_tmp"))
    chunks, doc_meta = retriever._load_all_chunks()
    return {
        "chunks": {c.chunk_id: c for c in chunks},
        "docs": doc_meta,
    }


def validate_dataset(dataset: BenchmarkDataset, chunks_dir: str,
                     ignore_chunk_labels: bool = False) -> List[str]:
    """Return a list of validation error strings (empty == valid).

    Checks:
      - every relevant chunk id exists in the corpus
      - every relevant document id exists
      - page references exist within the document's page count
      - dataset chunking_config_version matches the current chunker
      - document sha256 matches the artifact registry when recorded

    ignore_chunk_labels=True relaxes the CHUNK-ID existence/staleness
    checks (used by chunk-size experiments where boundaries move); such
    runs must be evaluated at PAGE relevance level.
    """
    errors: List[str] = []
    corpus = load_corpus(chunks_dir)
    chunks = corpus["chunks"]
    docs = corpus["docs"]

    meta = dataset.meta
    current_ccv = ChunkingConfig.CONFIG_VERSION
    if meta.chunking_config_version and \
            meta.chunking_config_version != current_ccv:
        if not ignore_chunk_labels:
            errors.append(
                f"stale benchmark: dataset chunking_config_version "
                f"{meta.chunking_config_version!r} != corpus "
                f"{current_ccv!r} (corpus was re-chunked)")

    for item in dataset.items:
        if not ignore_chunk_labels:
            for cid in item.relevant_chunk_ids:
                if cid not in chunks:
                    errors.append(
                        f"{item.id}: relevant chunk id {cid!r} does not exist "
                        f"in corpus {chunks_dir}")
        for did in item.relevant_document_ids:
            if did not in docs:
                errors.append(
                    f"{item.id}: relevant document id {did!r} not indexed")
        for pid in item.relevant_page_numbers:
            matching_docs = set()
            for cid in item.relevant_chunk_ids:
                c = chunks.get(cid)
                if c is not None:
                    matching_docs.add(c.document_id)
            if len(matching_docs) > 1:
                continue  # ambiguous; skip strict page check
            if matching_docs:
                doc_id = next(iter(matching_docs))
                meta_doc = docs.get(doc_id, {})
                page_count = meta_doc.get("page_count")
                if page_count and not (1 <= pid <= int(page_count)):
                    errors.append(
                        f"{item.id}: page {pid} out of range for {doc_id} "
                        f"(1..{page_count})")

    # sha256 staleness check against artifact registry
    if meta.document_sha256 and not ignore_chunk_labels:
        for doc_id, sha in meta.document_sha256.items():
            artifact_sha = (docs.get(doc_id, {}) or {}).get("source_sha256")
            if artifact_sha and artifact_sha != sha:
                errors.append(
                    f"stale benchmark: document {doc_id} sha changed "
                    f"(dataset {sha[:8]}… vs corpus {artifact_sha[:8]}…)")
    return errors


def require_valid(dataset: BenchmarkDataset, chunks_dir: str,
                  ignore_chunk_labels: bool = False) -> None:
    errors = validate_dataset(dataset, chunks_dir,
                              ignore_chunk_labels=ignore_chunk_labels)
    if errors:
        raise DatasetValidationError(
            "dataset validation failed:\n  - " + "\n  - ".join(errors))


def make_dataset_meta(document_sha: Dict[str, str]) -> DatasetMeta:
    return DatasetMeta(
        chunking_config_version=ChunkingConfig.CONFIG_VERSION,
        document_sha256=document_sha,
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
