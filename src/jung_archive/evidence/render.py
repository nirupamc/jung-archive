"""Deterministic EvidencePack rendering (M4).

Renders the structured pack into prompt-shaped text. The renderer never
invents metadata: title falls back to document_id when absent; section
line is omitted when no heading path exists. Structured objects remain
separate from the rendered string.
"""
from jung_archive.evidence.models import EvidencePack


def render_evidence_pack(pack: EvidencePack) -> str:
    lines = []
    lines.append("QUESTION:")
    lines.append(pack.question)
    lines.append("")
    lines.append("EVIDENCE:")
    lines.append("")
    if not pack.items:
        lines.append("(no evidence available)")
        return "\n".join(lines)

    for item in pack.items:
        lines.append(f"[{item.evidence_id}]")
        lines.append(
            f"Document: {item.title if item.title else item.document_id}")
        lines.append(f"Pages: {item.pages_display()}")
        if item.heading_path:
            lines.append("Section: " + " > ".join(item.heading_path))
        lines.append("Text:")
        lines.append(item.clean_text)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
