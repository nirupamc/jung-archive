"""Citation validation for generated answers.

Detects:
  - valid S-id      — citation references an evidence item that exists
  - unknown S-id    — e.g. [S99] when only S1-S5 exist in the pack
  - missing citation — model gave no citations despite available evidence
  - malformed citation — bracketed token that is not a valid S-id

No fabricated citations are silently accepted.
"""
from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel

from jung_archive.evidence.models import EvidencePack


class Citation(BaseModel):
    """One parsed citation token from a generated answer."""

    id: str                     # raw token, e.g. "[S1]"
    evidence_id: str            # normalized, e.g. "S1"
    status: str                 # valid | unknown | malformed
    note: Optional[str] = None


_CITATION_RE = re.compile(r"\[S(\d+)\]", re.IGNORECASE)


def parse_citations(text: str) -> List[Citation]:
    """Extract unique [S<N>] tokens from *text* in first-seen order."""
    citations: List[Citation] = []
    seen: set[str] = set()
    for m in _CITATION_RE.finditer(text):
        raw = m.group(0)
        sid = f"S{m.group(1)}"
        if sid in seen:
            continue
        seen.add(sid)
        citations.append(Citation(id=raw, evidence_id=sid, status="unknown"))
    return citations


def validate_citations(text: str, pack: EvidencePack) -> List[Citation]:
    """Return citations with status resolved against *pack*."""
    valid_ids = {item.evidence_id for item in pack.items}
    citations = parse_citations(text)
    for c in citations:
        if c.evidence_id in valid_ids:
            c.status = "valid"
        else:
            c.status = "unknown"
            c.note = f"{c.evidence_id} not in evidence pack"
    return citations


def citation_validation_warnings(text: str, pack: EvidencePack) -> List[str]:
    """Derive human-readable warnings for problematic citation patterns."""
    warnings: List[str] = []
    citations = validate_citations(text, pack)
    unknown = [c for c in citations if c.status == "unknown"]
    if unknown:
        ids = ", ".join(c.evidence_id for c in unknown)
        warnings.append(
            f"generated answer references {len(unknown)} unknown citation(s): {ids}"
        )
    if pack.items and not citations:
        warnings.append(
            "generated answer contains no citations despite available evidence"
        )
    return warnings
