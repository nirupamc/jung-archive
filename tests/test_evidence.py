"""M4 evidence tests: cleanup, dedup, diversity, budget, rendering."""
import json

import pytest

from jung_archive.chunking.tokenizer import count_tokens
from jung_archive.evidence.assembler import EvidenceAssembler, EvidenceConfig
from jung_archive.evidence.cleanup import clean_evidence_text
from jung_archive.evidence.dedup import (
    block_overlap,
    find_duplicates,
    is_duplicate,
    text_containment,
)
from jung_archive.evidence.models import EvidenceItem, EvidencePack
from jung_archive.evidence.render import render_evidence_pack
from jung_archive.models.document import SourceType
from jung_archive.retrieval.results import RetrievalResult


def make_result(chunk_id="c1", text="Some body text about the shadow.",
                blocks=None, pages=None, doc="d1", section="s1",
                rerank_score=1.0, rerank_rank=1, fusion_rank=1,
                title="Test Book"):
    return RetrievalResult(
        chunk_id=chunk_id,
        document_id=doc,
        text=text,
        page_numbers=pages or [1],
        source_block_ids=blocks or ["b1"],
        heading_path=["Chapter", "Section"],
        source_type=SourceType.PRIMARY,
        title=title,
        author="A. Author",
        section_id=section,
        fusion_rank=fusion_rank,
        fusion_score=0.05,
        dense_rank=1, dense_score=0.42,
        bm25_rank=2, bm25_score=3.7,
        reranker_rank=rerank_rank, reranker_score=rerank_score,
    )


DISTINCT = [
    "Alpha mechanics govern the rotating drum assembly precisely.",
    "Botanical illustrations depict fern reproduction cycles clearly.",
    "Cartographic surveys measured the coastline with triangulation.",
    "Digestive enzymes break proteins into absorbable amino fragments.",
    "Electromagnetic induction generates current within copper coils.",
]


def distinct_results(n, **kwargs):
    return [
        make_result(chunk_id=f"c{i}", text=DISTINCT[i % len(DISTINCT)],
                    blocks=[f"uniq-{i}"], pages=[i * 30], section=f"sec{i}",
                    rerank_rank=i + 1, **kwargs)
        for i in range(n)
    ]


# ----------------------------------------------------------------------
# Cleanup (14-18)

def test_original_text_preserved_and_clean_derived():
    original = make_result(
        text="The Undiscovered Self\n\nBody prose here.\n\n24",
        title="The Undiscovered Self")
    original.heading_path = ["The Undiscovered Self"]
    assembler = EvidenceAssembler(EvidenceConfig(max_evidence_tokens=1000))
    pack = assembler.assemble("q", [original])
    item = pack.items[0]
    assert item.text == original.text          # untouched original
    assert item.clean_text == "Body prose here."
    assert item.was_cleaned
    assert any(op.startswith("removed_folio") for op in item.cleanup_operations)
    # source result object not mutated
    assert original.text.endswith("24")


def test_running_header_cleanup():
    cleaned = clean_evidence_text(
        "The Undiscovered Self\n\nReal paragraph content.\n\nThe Undiscovered Self",
        title="The Undiscovered Self",
        heading_path=["The Undiscovered Self"],
    )
    assert cleaned.clean_text == "Real paragraph content."
    ops = cleaned.operations
    assert any("removed_running_header" in op for op in ops)


def test_duplicate_running_header_interior_removed():
    cleaned = clean_evidence_text(
        "Header X\n\nPara one.\n\nHeader X\n\nPara two.",
        title="Header X", heading_path=["Header X"])
    assert "Para one." in cleaned.clean_text
    assert cleaned.clean_text.count("Header X") <= 1
    assert any("duplicate_running_header" in op for op in cleaned.operations)


def test_folio_cleanup():
    cleaned = clean_evidence_text("12\n\nLegitimate argument continues.\n\n13")
    assert cleaned.clean_text == "Legitimate argument continues."
    assert sum(1 for op in cleaned.operations if "folio" in op) == 2


def test_combined_folio_title_running_head_cleanup():
    """Real corpus furniture: '39 the undiscovered self' style lines."""
    cleaned = clean_evidence_text(
        "39 the undiscovered self\n\nReal argument text remains here.\n\n"
        "the undiscovered self 40",
        title="The Undiscovered Self",
        heading_path=["Carl Gustav Jung"],
    )
    assert cleaned.clean_text == "Real argument text remains here."
    assert any("running_header" in op for op in cleaned.operations)


def test_legitimate_body_text_not_removed():
    body = ("The individual becomes merely an abstract number in the "
            "bureaucratic machine of the mass state.")
    cleaned = clean_evidence_text(body)
    assert cleaned.clean_text == body
    assert not cleaned.was_cleaned
    # numeric-looking sentence content must survive too
    cleaned2 = clean_evidence_text("In 1912 he published Studies in Word Association.")
    assert "1912" in cleaned2.clean_text


def test_cleanup_never_empties_text():
    cleaned = clean_evidence_text("25")
    assert cleaned.clean_text != ""
    assert any("would_remove_all_text" in op for op in cleaned.operations)


def test_cleanup_deterministic():
    t = "Title Page\n\nContent stays.\n\n7"
    a = clean_evidence_text(t, title="Title Page")
    b = clean_evidence_text(t, title="Title Page")
    assert a.clean_text == b.clean_text
    assert a.operations == b.operations


# ----------------------------------------------------------------------
# Deduplication (19-21)

def test_block_overlap_and_shared_provenance_dup():
    a = make_result(chunk_id="a", blocks=["p1-b0", "p1-b1"], pages=[24])
    b = make_result(chunk_id="b", blocks=["p1-b1", "p1-b2"], pages=[25],
                    text="Different wording entirely for the second item.")
    c = make_result(chunk_id="c", blocks=["p9-b0"], pages=[90],
                    text="Third candidate discusses unrelated matters.")
    assert block_overlap(a.source_block_ids, b.source_block_ids) > 0
    dup, why = is_duplicate(b, a)
    assert dup and "shared_provenance" in why
    dup_c, _ = is_duplicate(c, a)
    assert not dup_c


def test_text_overlap_dup():
    base = ("Mass-mindedness dissolves the individual into an anonymous "
            "crowd that cannot be held morally responsible. ")
    a = make_result(chunk_id="a", text=base * 2, blocks=["x1"], pages=[10])
    b = make_result(chunk_id="b", text=base + "Extra tail material.", 
                    blocks=["y9"], pages=[11])
    assert text_containment(a.text, b.text) >= 0.8
    dup, why = is_duplicate(b, a)
    assert dup and "text_overlap" in why


def test_adjacent_page_chunks_not_auto_duplicates():
    """Adjacent pages alone are legitimate; only corroboration suppresses."""
    a = make_result(chunk_id="a", text="Completely different topic one.",
                    blocks=["a-only"], pages=[24])
    b = make_result(chunk_id="b", text="Wholly unrelated subject matter two.",
                    blocks=["b-only"], pages=[25])
    dup, _ = is_duplicate(b, a)
    assert not dup


def test_find_duplicates_keeps_best_and_records_reason():
    a = make_result(chunk_id="keep", rerank_rank=1,
                    text="Alpha beta gamma delta epsilon.")
    b = make_result(chunk_id="dup", rerank_rank=2,
                    text="Alpha beta gamma delta epsilon.", blocks=["b2"])
    kept, suppressed = find_duplicates([a, b])
    assert [r["candidate"].chunk_id for r in kept] == ["keep"]
    assert len(suppressed) == 1
    cand, reason = suppressed[0]
    assert cand.chunk_id == "dup"
    assert reason.startswith("duplicate_of:keep:")


def test_nonduplicates_all_retained():
    results = distinct_results(5)
    kept, suppressed = find_duplicates(results)
    assert len(kept) == 5
    assert suppressed == []


# ----------------------------------------------------------------------
# Assembler: identifiers, serialization, provenance (11-13, 28)

def test_stable_evidence_identifiers():
    pack = EvidenceAssembler().assemble("q", distinct_results(4))
    assert [item.evidence_id for item in pack.items] == ["S1", "S2", "S3", "S4"]


def test_evidence_item_serialization_roundtrip():
    pack = EvidenceAssembler(EvidenceConfig(max_evidence_items=2)).assemble(
        "question?", distinct_results(3))
    item = pack.items[0]
    data = item.to_dict()
    raw = json.dumps(data)   # JSON-clean
    restored = EvidenceItem.from_dict(json.loads(raw))
    assert restored == item


def test_evidence_pack_serialization_roundtrip():
    pack = EvidenceAssembler().assemble("question?", distinct_results(2))
    raw = json.dumps(pack.to_dict())
    restored = EvidencePack.from_dict(json.loads(raw))
    assert len(restored.items) == len(pack.items)
    assert restored.tokens_used == pack.tokens_used
    assert restored.items[0].scores == pack.items[0].scores


def test_metadata_provenance_survives_assembly():
    res = make_result(pages=[24, 25], blocks=["p24-b0", "p25-b1"])
    pack = EvidenceAssembler().assemble("q", [res])
    item = pack.items[0]
    assert item.page_numbers == [24, 25]
    assert item.source_block_ids == ["p24-b0", "p25-b1"]
    assert item.heading_path == ["Chapter", "Section"]
    assert item.title == "Test Book"
    assert item.author == "A. Author"
    assert item.scores.dense_score == 0.42
    assert item.scores.bm25_score == 3.7
    assert item.scores.fusion_rank == 1
    assert item.scores.reranker_rank == 1
    assert item.scores.reranker_score == 1.0
    assert item.source_type == SourceType.PRIMARY


# ----------------------------------------------------------------------
# Diversity (23)

def test_diversity_page_region_cap():
    cfg = EvidenceConfig(max_chunks_per_page_region=2, page_region_size=8,
                         max_evidence_items=10, max_evidence_tokens=5000)
    results = []
    for i in range(4):
        results.append(make_result(
            chunk_id=f"c{i}", pages=[i],  # all in region 0
            blocks=[f"uniq-block-{i}"], section=f"sec{i}",
            rerank_rank=i + 1, text=DISTINCT[i]))
    pack = EvidenceAssembler(cfg).assemble("q", results)
    assert len(pack.items) == 2
    assert {s.reason for s in pack.suppressed_diversity} == \
        {"max_chunks_per_page_region"}


def test_diversity_section_cap():
    cfg = EvidenceConfig(max_chunks_per_section=1, max_evidence_tokens=5000,
                         max_evidence_items=10)
    results = [
        make_result(chunk_id=f"c{i}", section="same-section",
                    blocks=[f"bk{i}"], pages=[i * 20],
                    rerank_rank=i + 1, text=DISTINCT[i])
        for i in range(3)
    ]
    pack = EvidenceAssembler(cfg).assemble("q", results)
    assert len(pack.items) == 1


# ----------------------------------------------------------------------
# Token budget (24-26)

def test_token_budget_respected():
    results = [make_result(chunk_id=f"c{i}", text=DISTINCT[i % len(DISTINCT)] * 3,
                           blocks=[f"b{i}"], pages=[i], rerank_rank=i + 1)
               for i in range(6)]
    cfg = EvidenceConfig(max_evidence_tokens=80, max_evidence_items=10)
    pack = EvidenceAssembler(cfg).assemble("q", results)
    assert 0 < pack.tokens_used <= 80
    assert len(pack.items) >= 2
    # items follow reranked relevance order
    assert [i.evidence_id for i in pack.items] == \
        [f"S{j}" for j in range(1, len(pack.items) + 1)]
    # the next candidate would have exceeded the budget
    if len(pack.items) < 6:
        next_tokens = count_tokens(
            DISTINCT[len(pack.items) % len(DISTINCT)] * 3)
        assert pack.tokens_used + next_tokens > 80


def test_budget_never_exceeded():
    results = [make_result(chunk_id=f"c{i}",
                           text=" ".join(f"u{i}x{j}" for j in range(30)),
                           blocks=[f"b{i}"], pages=[i], rerank_rank=i + 1)
               for i in range(8)]
    cfg = EvidenceConfig(max_evidence_tokens=70, max_evidence_items=20)
    pack = EvidenceAssembler(cfg).assemble("q", results)
    total = sum(count_tokens(i.clean_text) for i in pack.items)
    assert total == pack.tokens_used
    assert total <= 70


def test_oversized_item_handled_explicitly():
    huge = make_result(chunk_id="huge", text="giant " * 3000,
                       blocks=["h1"], rerank_rank=1)
    small = make_result(chunk_id="small", text="tiny but relevant passage.",
                        blocks=["s1"], pages=[2], rerank_rank=2)
    cfg = EvidenceConfig(max_evidence_tokens=2500, max_evidence_items=8)
    pack = EvidenceAssembler(cfg).assemble("q", [huge, small])
    assert pack.items[0].chunk_id == "small"
    assert any(s.chunk_id == "huge" for s in pack.skipped_oversized)


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        EvidenceConfig(max_evidence_tokens=0)
    with pytest.raises(ValueError):
        EvidenceConfig(max_evidence_items=-1)


def test_empty_candidates_pack():
    pack = EvidenceAssembler().assemble("q", [])
    assert pack.items == []
    assert pack.candidates_considered == 0
    assert pack.warnings


# ----------------------------------------------------------------------
# Rendering (27)

def test_render_shape_no_invented_metadata():
    pack = EvidenceAssembler().assemble(
        "What is the shadow?",
        [make_result(chunk_id="c1", text="Shadow content paragraph.",
                     pages=[25, 26])],
    )
    rendered = render_evidence_pack(pack)
    assert rendered.startswith("QUESTION:\nWhat is the shadow?")
    assert "[S1]" in rendered
    assert "Document: Test Book" in rendered      # from metadata, real
    assert "Pages: 25-26" in rendered
    assert "Section: Chapter > Section" in rendered
    assert "Shadow content paragraph." in rendered
    assert "(no evidence available)" not in rendered


def test_render_falls_back_to_document_id_without_title():
    res = make_result()
    res.title = None
    pack = EvidenceAssembler().assemble("q", [res])
    assert f"Document: {res.document_id}" in render_evidence_pack(pack)


def test_render_empty_pack():
    pack = EvidencePack(question="q", max_evidence_tokens=10,
                        max_evidence_items=2)
    out = render_evidence_pack(pack)
    assert "(no evidence available)" in out


# ----------------------------------------------------------------------
# Duplicate group bookkeeping inside assembly

def test_assembly_records_suppressed_duplicates():
    base = "Identical overlapping evidence sentence repeated fully here. "
    a = make_result(chunk_id="first", text=base * 2, blocks=["s1"],
                    rerank_rank=1)
    b = make_result(chunk_id="second", text=base * 2, blocks=["s1"],
                    rerank_rank=2)
    c = make_result(chunk_id="other", text="Different content entirely ok.",
                    blocks=["zz"], pages=[77], rerank_rank=3)
    pack = EvidenceAssembler(EvidenceConfig(max_evidence_tokens=4000)).assemble(
        "q", [a, b, c])
    assert [i.chunk_id for i in pack.items] == ["first", "other"]
    assert len(pack.suppressed_duplicates) == 1
    assert pack.suppressed_duplicates[0].chunk_id == "second"
