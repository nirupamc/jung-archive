"""Grounded prompt assembly from EvidencePack.

Builds a strict evidence-only prompt for the generation provider. The
prompt never invents metadata: every claim must be traceable to the
supplied evidence items.
"""

SYSTEM_INSTRUCTIONS = (
    "You are a careful research assistant answering questions about Jung's works.\n"
    "Use ONLY the EVIDENCE supplied below. Do not invent facts outside the evidence.\n"
    "Cite claims using the bracketed IDs from the evidence, e.g. [S1], [S3].\n"
    "Do not invent citation IDs. If the evidence is insufficient, say so explicitly.\n"
    "Do not claim certainty beyond the evidence."
)


def build_ask_prompt(question: str, pack) -> str:
    """Return the full user prompt string.

    Layout:
        <system instructions>
        EVIDENCE:
        [S1] Document: ...
             Pages: ...
             Text: ...

        [S2] ...

        QUESTION: <question>
        ANSWER:
    """
    lines: list[str] = [SYSTEM_INSTRUCTIONS, "", "EVIDENCE:", ""]
    if pack.items:
        for item in pack.items:
            lines.append(f"[{item.evidence_id}]")
            lines.append(
                f"Document: {item.title if item.title else item.document_id}"
            )
            lines.append(f"Pages: {item.pages_display()}")
            if item.heading_path:
                lines.append("Section: " + " > ".join(item.heading_path))
            lines.append("Text:")
            lines.append(item.clean_text)
            lines.append("")
    else:
        lines.append("(no evidence available)")
        lines.append("")
    lines.append(f"QUESTION: {question}")
    lines.append("")
    lines.append("ANSWER:")
    return "\n".join(lines)
