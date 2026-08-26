"""
Explicit document metadata registry.

Folder location alone must never grant trust: a file under primary/ is not
automatically authentic Jung material (see the Bookey Aion summary incident).
This registry records explicit curation decisions keyed by source SHA-256
and/or path substring, and every unregistered document defaults to
IndexStatus.REVIEW so nothing silently enters the production index.
"""
import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from jung_archive.models.document import IndexStatus, SourceType

DEFAULT_REGISTRY_PATH = "config/document_metadata.json"


@dataclass
class DocumentMetadata:
    """Curated metadata for one source document."""
    title_override: Optional[str] = None
    author: Optional[str] = None
    source_type: SourceType = SourceType.UNKNOWN
    index_status: IndexStatus = IndexStatus.REVIEW
    reason: Optional[str] = None


def sha256_of_file(path: str) -> str:
    """SHA-256 of file contents, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class MetadataRegistry:
    """Loads and applies explicit document metadata overrides.

    Entries are matched by full SHA-256 first (strongest), then by path
    substring. Unregistered documents get REVIEW status by default.
    """

    entries_by_sha: Dict[str, DocumentMetadata] = field(default_factory=dict)
    entries_by_path: List[tuple] = field(default_factory=list)

    @classmethod
    def load(cls, path: Optional[str] = None) -> "MetadataRegistry":
        registry = cls()
        p = Path(path or DEFAULT_REGISTRY_PATH)
        if not p.exists():
            return registry
        with open(p, encoding="utf-8") as f:
            raw = json.load(f)
        for entry in raw.get("documents", []):
            meta = DocumentMetadata(
                title_override=entry.get("title"),
                author=entry.get("author"),
                source_type=SourceType(entry.get("source_type", "UNKNOWN")),
                index_status=IndexStatus(entry.get("index_status", "REVIEW")),
                reason=entry.get("reason"),
            )
            if entry.get("sha256"):
                registry.entries_by_sha[entry["sha256"]] = meta
                # A dual-keyed entry also registers its path substring so a
                # truncated/stale checksum can never orphan the decision.
                if entry.get("path_contains"):
                    registry.entries_by_path.append(
                        (entry["path_contains"], meta))
            elif entry.get("path_contains"):
                registry.entries_by_path.append((entry["path_contains"], meta))
        return registry

    def lookup(self, source_path: str, sha256: Optional[str] = None) -> Optional[DocumentMetadata]:
        if sha256 and sha256 in self.entries_by_sha:
            return self.entries_by_sha[sha256]
        normalized = source_path.replace("\\", "/").lower()
        for needle, meta in self.entries_by_path:
            if needle.lower() in normalized:
                return meta
        return None

    def apply(self, document) -> None:
        """Apply registry decisions to a Document in place.

        - Registered documents get their explicit source_type / index_status /
          author / title override.
        - Unregistered documents default to REVIEW; folder inference may set
          a tentative source_type, but never INCLUDE status.
        """
        meta = self.lookup(document.source_path, document.source_sha256)
        if meta is not None:
            document.source_type = meta.source_type
            document.index_status = meta.index_status
            if meta.author is not None:
                document.author = meta.author
            if meta.title_override is not None:
                document.title = meta.title_override
        else:
            # Tentative folder-based typing at most; always REVIEW.
            document.index_status = IndexStatus.REVIEW
