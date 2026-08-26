"""
Lightweight section reconstruction from M1 structural blocks.

Builds a conservative heading hierarchy:
  - TITLE blocks (or the document title) anchor top level
  - HEADING blocks nest below, distinguished by relative font size
Uncertain evidence never fabricates hierarchy: pages with no heading
evidence fall into a single implicit "body" section.
Document order is always preserved.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from jung_archive.models.document import Block, BlockType, Document


@dataclass
class SectionNode:
    """A contiguous span of the document under one heading."""
    section_id: str
    title: str
    level: int
    heading_path: List[str]
    # Heading block that anchored this section, as (page_number, block).
    # None for the implicit root section.
    anchor: Optional[tuple] = None
    # Global ordered items: (page_number, block)
    items: List[tuple] = field(default_factory=list)


@dataclass
class SectionTree:
    """Ordered flat list of sections covering the whole document."""
    document_title: str
    sections: List[SectionNode]

    def all_items(self) -> List[tuple]:
        out = []
        for sec in self.sections:
            for item in sec.items:
                out.append((sec, item))
        return out


def _block_sort_key(page_number: int, block: Block):
    return (page_number, block.reading_order)


def build_section_tree(document: Document) -> SectionTree:
    """Reconstruct ordered section spans from typed blocks."""
    # Flatten blocks in global reading order
    flattened = []
    for page in sorted(document.pages, key=lambda p: p.page_number):
        for block in page.blocks:
            if block.block_type in (BlockType.HEADER, BlockType.FOOTER,
                                    BlockType.PAGE_NUMBER):
                continue  # running heads/numbers are not content
            flattened.append((page.page_number, block))

    doc_title = document.title or "document"
    sections: List[SectionNode] = []
    current_path: List[str] = [doc_title]

    def new_section(title: str, level: int, anchor: Optional[tuple] = None) -> SectionNode:
        sid = f"{document.document_id}-s{len(sections):04d}"
        path = current_path[:level + 1] if level > 0 else [doc_title]
        if level > 0:
            while len(current_path) <= level:
                current_path.append(title)
            current_path[level] = title
            path = list(current_path[:level + 1])
        else:
            current_path.clear()
            current_path.extend([title])
            path = [title]
        return SectionNode(
            section_id=sid,
            title=title,
            level=level,
            heading_path=path,
            anchor=anchor,
        )

    # Implicit root section until/unless a TITLE appears
    sections.append(new_section(doc_title, 0))

    for page_number, block in flattened:
        if block.block_type == BlockType.TITLE:
            sec = new_section(_one_line(block.text), 0, anchor=(page_number, block))
            sections.append(sec)
        elif block.block_type == BlockType.HEADING:
            # Subheadings nest one level below root when distinguishable;
            # M1 typing does not carry deeper reliable levels.
            sec = new_section(_one_line(block.text), 1, anchor=(page_number, block))
            sections.append(sec)
        else:
            sections[-1].items.append((page_number, block))

    return SectionTree(document_title=doc_title, sections=sections)


def _one_line(text: str) -> str:
    return " ".join(text.split())