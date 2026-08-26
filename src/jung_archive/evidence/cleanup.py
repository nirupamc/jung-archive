"""Source-preserving evidence cleanup (M4).

Derives `clean_text` from a chunk's original text WITHOUT mutating the
canonical chunk. Only obvious page furniture is removed:

  - folios: lines that are pure page numbers at the chunk boundary
  - running heads: short boundary lines that repeat the document title
    or the leading heading path entry

Every removal is recorded as an explainable operation string. Body
prose is never rewritten; Jung's wording is untouched. No LLM.
"""
import re
from dataclasses import dataclass, field
from typing import List, Tuple

_FOLIO_RE = re.compile(r"^\s*[-—–|.]?\s*(\d{1,4}|[ivxlcdm]{1,7})\s*[-—–|.]*\s*$",
                       re.IGNORECASE)
_MAX_FURNITURE_LINE_LEN = 80


@dataclass
class CleanupResult:
    clean_text: str
    operations: List[str] = field(default_factory=list)

    @property
    def was_cleaned(self) -> bool:
        return bool(self.operations)


def _lines(text: str) -> List[str]:
    return text.split("\n")


def _is_folio(line: str) -> bool:
    return bool(_FOLIO_RE.match(line.strip())) and len(line.strip()) <= 12


def _normalized(s: str) -> str:
    return " ".join(s.lower().split())


def _strip_edge_numbers(s: str) -> str:
    """Remove leading/trailing page-number tokens from a line."""
    return re.sub(r"^\s*\d{1,4}\s+|\s+\d{1,4}\s*$", " ", s).strip()


def _matches_running_head(line: str, title: str,
                          heading_path: List[str]) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_FURNITURE_LINE_LEN:
        return False
    norm = _normalized(stripped)
    # Running heads frequently combine folio + title, e.g.
    # "39 the undiscovered self". Compare the number-stripped form too.
    variants = {norm}
    stripped_nums = _strip_edge_numbers(stripped)
    if stripped_nums and _normalized(stripped_nums) != norm:
        variants.add(_normalized(stripped_nums))
    candidates = {_normalized(title)} if title else set()
    for h in heading_path[:2]:
        candidates.add(_normalized(h))
        # Running heads often concatenate title + section.
        if title:
            candidates.add(_normalized(f"{title} {h}"))
    candidates = {c for c in candidates if c}
    return bool(variants & candidates)


def clean_evidence_text(text: str, title: str = "",
                        heading_path: List[str] | None = None) -> CleanupResult:
    """Return cleaned text plus the list of cleanup operations applied.

    Deterministic: identical input always yields identical output.
    """
    heading_path = heading_path or []
    lines = _lines(text)
    ops: List[str] = []
    start = 0
    end = len(lines)

    # Leading furniture: consecutive folio / running-head lines at top.
    while start < end:
        line = lines[start]
        if _is_folio(line):
            ops.append(f"removed_folio:leading_line_{start + 1}")
            start += 1
        elif _matches_running_head(line, title, heading_path):
            ops.append(f"removed_running_header:line_{start + 1}")
            start += 1
        else:
            break

    # Trailing furniture.
    while end - 1 >= start:
        line = lines[end - 1]
        if not line.strip():
            end -= 1
            continue
        if _is_folio(line):
            ops.append(f"removed_folio:trailing_line_{end}")
            end -= 1
        elif _matches_running_head(line, title, heading_path):
            ops.append(f"removed_running_header:line_{end}")
            end -= 1
        else:
            break

    # Repeated interior duplicate of the running head appearing more
    # than once in full (page furniture repeated across a chunk seam).
    # If a leading running head was already removed, remaining interior
    # copies count as duplicates immediately.
    seen_head = any(op.startswith("removed_running_header")
                    for op in ops)
    for i in range(start, end):
        line = lines[i]
        if line.strip() and _matches_running_head(line, title, heading_path):
            if seen_head:
                ops.append(f"removed_duplicate_running_header:line_{i + 1}")
                lines[i] = ""
            seen_head = True

    clean = "\n".join(lines[start:end]).strip()
    if not clean:
        # Never return empty evidence: fall back to the original text
        # and record why. This cannot destroy legitimate content.
        ops.append("cleanup_aborted:would_remove_all_text")
        return CleanupResult(clean_text=text.strip(), operations=ops)
    return CleanupResult(clean_text=clean, operations=ops)
