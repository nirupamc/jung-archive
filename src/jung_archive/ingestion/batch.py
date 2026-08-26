"""
Batch ingestion over the discovered corpus (post-M7).

Runs every registry-approved INCLUDE document through the existing
inspect -> chunk -> embed -> index pipeline, reusing incremental/idempotent
mechanisms so unchanged documents are never regenerated:

  * canonical inspect artifact is reused when present (no re-parse)
  * provenance validation gates indexing exactly like the CLI
  * VectorIndex.index_chunks stays a no-op while (sha, config, model)
    fingerprints match

REVIEW and EXCLUDE documents are never ingested by this module; every
candidate is re-checked against the live registry immediately before its
chunk artifact is written.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional

from jung_archive.corpus import REPO_ROOT, DiscoveredDocument, \
    discover_corpus
from jung_archive.metadata import DEFAULT_REGISTRY_PATH, MetadataRegistry


def load_processed_document(document_id: str, processed_dir: Path):
    """Rebuild a Document from its canonical artifact (no re-parse)."""
    import json as _json

    from jung_archive.models.document import Document

    path = processed_dir / f"{document_id}.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return Document.model_validate(_json.load(f))


def ingest_batch(
    *,
    repo_root: Path = REPO_ROOT,
    sections=("primary", "secondary"),
    force_index: bool = False,
    limit: Optional[int] = None,
    only_paths: Optional[List[str]] = None,
    progress: Callable[[str], None] = lambda msg: None,
    provider=None,
) -> Dict:
    """Ingest all approved INCLUDE documents; returns an honest report.

    ``provider`` allows injecting an embedding provider (tests); by default
    the production LocalSentenceTransformerProvider is used lazily.
    """
    from jung_archive.chunking.artifacts import save_chunk_artifact
    from jung_archive.chunking.chunker import StructureAwareChunker
    from jung_archive.chunking.validation import validate_chunks
    from jung_archive.indexing.vector_index import VectorIndex
    from jung_archive.models.chunk import ChunkingConfig

    root = Path(repo_root)
    started = time.time()
    docs = discover_corpus(repo_root=root, sections=sections)
    if only_paths is not None:
        wanted = {p.replace("\\", "/") for p in only_paths}
        docs = [d for d in docs if d.path in wanted]
    candidates = [d for d in docs if d.index_status == "INCLUDE"]
    if limit is not None:
        candidates = candidates[:limit]

    registry = MetadataRegistry.load(str(root / DEFAULT_REGISTRY_PATH))
    config = ChunkingConfig()
    if provider is None:
        from jung_archive.embedding.provider import \
            LocalSentenceTransformerProvider

        provider = LocalSentenceTransformerProvider()
    index = VectorIndex(provider, persist_dir=str(root / "data" / "chroma"))

    report: Dict = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "sections": [s.upper() for s in sections],
        "candidates": len(candidates),
        "processed_ok": [],
        "skipped": [],
        # honest ledger of everything deliberately NOT touched
        "held_back": [
            {
                "path": d.path,
                "status": d.status,
                "registry_decision": d.index_status,
                "reason": d.reason
                or ("unregistered: folder location alone never grants trust"
                    if not d.registered else None),
            }
            for d in docs if d.index_status != "INCLUDE"
        ],
        "failed": [],
        "totals": {
            "pages": 0, "blocks": 0, "chunks": 0,
            "vectors_indexed": 0, "index_unchanged": 0,
        },
        "artifacts_reused": 0,
        "freshly_ingested": 0,
        "collection_vectors_after": None,
    }

    def skip(doc: DiscoveredDocument, reason: str):
        report["skipped"].append({"path": doc.path, "reason": reason})
        progress(f"skip  {doc.path}: {reason}")

    for doc in candidates:
        if doc.status == "ERROR":
            skip(doc, doc.error or "unreadable")
            continue
        try:
            processed_dir = root / "data" / "processed"
            diagnostics_dir = root / "data" / "diagnostics"
            processed_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

            assert doc.document_id is not None
            document = load_processed_document(
                doc.document_id, processed_dir)
            # Reuse only when the artifact provably matches the CURRENT
            # file bytes; a stale artifact is re-parsed, never trusted.
            if document is not None and \
                    document.source_sha256 != doc.sha256:
                progress(f"stale canonical artifact (sha mismatch): "
                         f"{doc.path}; re-parsing")
                document = None
            if document is not None:
                report["artifacts_reused"] += 1
            else:
                from jung_archive.cli import generate_diagnostics
                from jung_archive.ingestion.pdf import PDFIngestor

                progress(f"inspect (fresh parse): {doc.path}")
                ingestor = PDFIngestor()
                document = ingestor.ingest(str(root / doc.path))
                with open(processed_dir / f"{doc.document_id}.json", "w",
                          encoding="utf-8") as f:
                    json.dump(document.to_dict(), f, indent=2,
                              ensure_ascii=False)
                with open(
                    diagnostics_dir / f"{doc.document_id}.json", "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(generate_diagnostics(document), f, indent=2,
                              ensure_ascii=False)
                report["freshly_ingested"] += 1

            # Safety gate: reapply the LIVE registry; folder location and
            # stale artifacts never grant trust.
            registry.apply(document)
            if document.index_status.value != "INCLUDE":
                skip(doc, f"registry decision is "
                          f"{document.index_status.value}; refusing to index")
                continue

            chunks = StructureAwareChunker(config).chunk_document(document)
            validation = validate_chunks(chunks, document)
            if not validation.ok:
                report["failed"].append({
                    "path": doc.path,
                    "error": "provenance validation failed: "
                             + "; ".join(validation.errors[:3]),
                })
                progress(f"FAIL  {doc.path}: provenance validation")
                continue

            save_chunk_artifact(chunks, document, config,
                                str(root / "data" / "chunks"))

            index_report = index.index_chunks(
                chunks,
                source_sha256=document.source_sha256 or "",
                chunking_config_version=ChunkingConfig.CONFIG_VERSION,
                force=force_index,
            )

            n_blocks = sum(len(p.blocks) for p in document.pages)
            indexed_now = int(index_report.get("indexed", 0))
            report["processed_ok"].append({
                "path": doc.path,
                "document_id": document.document_id,
                "title": document.title,
                "pages": document.page_count,
                "blocks": n_blocks,
                "chunks": len(chunks),
                "vectors_indexed": indexed_now,
                "index_status_detail": index_report.get("skipped", "ok"),
            })
            totals = report["totals"]
            totals["pages"] += document.page_count
            totals["blocks"] += n_blocks
            totals["chunks"] += len(chunks)
            totals["vectors_indexed"] += indexed_now
            if indexed_now == 0:
                totals["index_unchanged"] += 1
            progress(
                f"ok    {doc.path}: {len(chunks)} chunks "
                f"({indexed_now} newly embedded)")
        except Exception as e:  # one bad document must not stop the batch
            report["failed"].append({"path": doc.path, "error": str(e)})
            progress(f"FAIL  {doc.path}: {e}")

    report["collection_vectors_after"] = index.count()
    report["finished_at"] = datetime.now(timezone.utc).isoformat()
    report["elapsed_s"] = round(time.time() - started, 1)
    return report


def save_batch_report(report: Dict, repo_root: Path = REPO_ROOT) -> Path:
    out_dir = Path(repo_root) / "data" / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    path = out_dir / f"batch_ingest_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    latest = out_dir / "batch_ingest_latest.json"
    with open(latest, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
