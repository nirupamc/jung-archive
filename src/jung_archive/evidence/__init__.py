"""Evidence assembly pipeline (M4)."""
from jung_archive.evidence.assembler import EvidenceAssembler, EvidenceConfig
from jung_archive.evidence.cleanup import clean_evidence_text
from jung_archive.evidence.dedup import (
    block_overlap,
    find_duplicates,
    is_duplicate,
    text_containment,
)
from jung_archive.evidence.models import (
    EvidenceItem,
    EvidencePack,
    ScorePath,
    SuppressedItem,
)
from jung_archive.evidence.render import render_evidence_pack

__all__ = [
    "EvidenceAssembler",
    "EvidenceConfig",
    "clean_evidence_text",
    "block_overlap",
    "find_duplicates",
    "is_duplicate",
    "text_containment",
    "EvidenceItem",
    "EvidencePack",
    "ScorePath",
    "SuppressedItem",
    "render_evidence_pack",
]
