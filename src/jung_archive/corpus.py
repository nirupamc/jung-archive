"""
Corpus discovery layer (post-M7).

Lists every PDF under primary/ and secondary/ - including documents that
have never been processed - and derives an honest pipeline status for each.

Pipeline statuses (DiscoveredDocument.status):

  DISCOVERED  file found on disk, approved by the registry, no canonical
              artifact produced yet
  REVIEW      registry decision is REVIEW, or the file is unregistered
              (folder location alone NEVER grants trust)
  EXCLUDED    registry decision is EXCLUDE; must never enter the index
  PROCESSED   canonical inspect artifact exists (data/processed), not chunked
  CHUNKED     chunk artifact exists (data/chunks), but the vector index
              state has no matching record for this document
  INDEXED     chunked AND recorded in the Chroma index state with a
              matching source SHA-256 and chunk count
  ERROR       the PDF could not be opened/read at all

Curation decisions always come from the explicit metadata registry
(config/document_metadata.json); see jung_archive.metadata.
"""
import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from jung_archive.metadata import DEFAULT_REGISTRY_PATH, MetadataRegistry, \
    sha256_of_file

REPO_ROOT = Path(__file__).resolve().parents[2]
CORPUS_SECTIONS = ("primary", "secondary")


class PipelineStatus(str, Enum):
    DISCOVERED = "DISCOVERED"
    REVIEW = "REVIEW"
    EXCLUDED = "EXCLUDED"
    PROCESSED = "PROCESSED"
    CHUNKED = "CHUNKED"
    INDEXED = "INDEXED"
    ERROR = "ERROR"


@dataclass
class DiscoveredDocument:
    """One discovered corpus file with its honest pipeline state."""
    path: str                       # repo-relative posix path
    file_name: str
    section: str                    # PRIMARY | SECONDARY
    sha256: str
    size_bytes: int
    page_count: int
    document_id: Optional[str]      # deterministic id (sha256 of abs path)
    embedded_title: Optional[str]
    embedded_author: Optional[str]
    title: Optional[str]            # effective title (registry > embedded)
    author: Optional[str]           # effective author (registry > embedded)
    source_type: str                # PRIMARY | SECONDARY | UNKNOWN
    index_status: str               # INCLUDE | REVIEW | EXCLUDE | UNKNOWN
    registered: bool
    reason: Optional[str]           # registry reason, when present
    status: str                     # PipelineStatus value
    error: Optional[str] = None     # populated for ERROR rows


def generate_document_id(pdf_path: str) -> str:
    """Same deterministic rule as PDFIngestor._generate_document_id."""
    import hashlib

    return hashlib.sha256(
        os.path.abspath(pdf_path).encode("utf-8")).hexdigest()[:12]


def _read_pdf_basics(abs_path: Path):
    """Return (page_count, embedded_title, embedded_author) or raise."""
    import fitz

    with fitz.open(str(abs_path)) as doc:
        meta = doc.metadata or {}
        return doc.page_count, \
            (meta.get("title") or None), (meta.get("author") or None)


def _processed_artifact_path(document_id: str, processed_dir: Path) -> Path:
    return processed_dir / f"{document_id}.json"


def _chunk_artifact_path(document_id: str, chunks_dir: Path) -> Path:
    return chunks_dir / f"{document_id}.json"


def load_index_state(chroma_dir: Path) -> Dict:
    state_file = chroma_dir / "index_state.json"
    if not state_file.exists():
        return {}
    try:
        with open(state_file, encoding="utf-8") as f:
            return json.load(f).get("documents", {})
    except Exception:
        return {}


def derive_status(
    *,
    readable: bool,
    index_status_value: str,
    registered: bool,
    has_processed: bool,
    has_chunks: bool,
    index_state: Optional[Dict],
) -> str:
    """Single source of truth for pipeline-status derivation."""
    if not readable:
        return PipelineStatus.ERROR.value
    if index_status_value == "EXCLUDE":
        return PipelineStatus.EXCLUDED.value
    if index_status_value != "INCLUDE":
        # REVIEW decision or unregistered file: held out of the pipeline.
        return PipelineStatus.REVIEW.value
    if index_state is not None and has_chunks:
        return PipelineStatus.INDEXED.value
    if has_chunks:
        return PipelineStatus.CHUNKED.value
    if has_processed:
        return PipelineStatus.PROCESSED.value
    return PipelineStatus.DISCOVERED.value


def discover_corpus(
    repo_root: Path = REPO_ROOT,
    registry: Optional[MetadataRegistry] = None,
    sections=CORPUS_SECTIONS,
) -> List[DiscoveredDocument]:
    """Scan corpus folders and classify every PDF found."""
    root = Path(repo_root)
    reg = registry if registry is not None else MetadataRegistry.load(
        str(root / DEFAULT_REGISTRY_PATH))
    index_states = load_index_state(root / "data" / "chroma")

    docs: List[DiscoveredDocument] = []
    for section in sections:
        folder = root / section
        if not folder.exists():
            continue
        for pdf in sorted(folder.glob("*.pdf")):
            rel = pdf.relative_to(root).as_posix()
            sha = sha256_of_file(str(pdf))
            document_id = generate_document_id(str(pdf))
            size_bytes = pdf.stat().st_size

            readable = True
            page_count = 0
            embedded_title = embedded_author = None
            error = None
            try:
                page_count, embedded_title, embedded_author = \
                    _read_pdf_basics(pdf)
            except Exception as e:  # unreadable file stays visible as ERROR
                readable = False
                error = f"unreadable PDF: {e}"

            meta = reg.lookup(rel, sha)
            registered = meta is not None
            title = embedded_title
            author = embedded_author
            source_type = "SECONDARY" if section == "secondary" \
                else ("PRIMARY" if section == "primary" else "UNKNOWN")
            index_status_value = "UNKNOWN"
            reason = None
            if registered:
                source_type = meta.source_type.value
                index_status_value = meta.index_status.value
                reason = meta.reason
                title = meta.title_override or title
                author = meta.author or author

            has_processed = _processed_artifact_path(
                document_id, root / "data" / "processed").exists()
            has_chunks = _chunk_artifact_path(
                document_id, root / "data" / "chunks").exists()

            # INDEXED requires a matching record in the vector-index state;
            # a stale/mismatched record means only CHUNKED (honesty first).
            index_state_entry = index_states.get(document_id)
            indexed_ok = (
                isinstance(index_state_entry, dict)
                and index_state_entry.get("source_sha256") == sha
                and bool(index_state_entry.get("chunk_count"))
            )

            status = derive_status(
                readable=readable,
                index_status_value=index_status_value,
                registered=registered,
                has_processed=has_processed,
                has_chunks=has_chunks,
                index_state=index_state_entry if indexed_ok else None,
            )
            if status == PipelineStatus.INDEXED.value and not has_processed:
                # artifacts were pruned underneath the index; be honest
                status = PipelineStatus.CHUNKED.value

            docs.append(DiscoveredDocument(
                path=rel,
                file_name=pdf.name,
                section=section.upper(),
                sha256=sha,
                size_bytes=size_bytes,
                page_count=page_count,
                document_id=document_id,
                embedded_title=embedded_title,
                embedded_author=embedded_author,
                title=title or pdf.stem,
                author=author,
                source_type=source_type,
                index_status=index_status_value,
                registered=registered,
                reason=reason,
                status=status,
                error=error,
            ))
    return docs


def corpus_report(docs: List[DiscoveredDocument]) -> Dict:
    """Aggregate counts used by the CLI report and API stats."""
    from collections import Counter

    statuses = Counter(d.status for d in docs)
    return {
        "discovered_total": len(docs),
        "by_section": {
            s.upper(): sum(1 for d in docs if d.section == s.upper())
            for s in CORPUS_SECTIONS
        },
        "by_status": {s.value: statuses.get(s.value, 0)
                      for s in PipelineStatus},
        "pages_total": sum(d.page_count for d in docs),
        "included": statuses.get("INDEXED", 0)
        + statuses.get("CHUNKED", 0) + statuses.get("PROCESSED", 0)
        + statuses.get("DISCOVERED", 0),
        "excluded": statuses.get("EXCLUDED", 0),
        "review": statuses.get("REVIEW", 0),
        "error": statuses.get("ERROR", 0),
    }
