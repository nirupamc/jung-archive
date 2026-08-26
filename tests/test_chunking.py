import pytest

from jung_archive.chunking.chunker import StructureAwareChunker
from jung_archive.chunking.tokenizer import count_tokens
from jung_archive.chunking.validation import ProvenanceError, validate_chunks
from jung_archive.models.chunk import Chunk, ChunkingConfig
from jung_archive.models.document import IndexStatus, SourceType

PARA = (
    "The psyche is a self-regulating system that maintains its equilibrium "
    "just as the body does. Jung argued that the unconscious compensates "
    "for the one-sidedness of the conscious attitude. "
)


def para(n_words_repeat=1):
    return (PARA * n_words_repeat).strip()


def simple_doc(doc_factory):
    return doc_factory([
        [("TITLE", "The Structure of the Psyche")],
        [("HEADING", "Part One"), ("PARAGRAPH", para(2)), ("PARAGRAPH", para(3))],
        [("PARAGRAPH", para(4)), ("PARAGRAPH", para(5))],
    ])


class TestChunkModel:
    def test_serialization_round_trip(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = StructureAwareChunker().chunk_document(document)
        for ch in chunks:
            data = ch.to_dict()
            rebuilt = Chunk.from_dict(data)
            assert rebuilt == ch

    def test_deterministic_ids(self, doc_factory):
        a = StructureAwareChunker().chunk_document(simple_doc(doc_factory))
        b = StructureAwareChunker().chunk_document(simple_doc(doc_factory))
        assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
        assert all(c.chunk_id.startswith(document_id := "testdoc000-")
                   for c in a)

    def test_no_empty_chunks(self, doc_factory):
        document = doc_factory([[("TITLE", "x")], [("PARAGRAPH", "   ")]])
        # whitespace-only block should never yield an empty chunk
        chunker = StructureAwareChunker()
        chunks = chunker.chunk_document(document)
        assert all(c.text.strip() for c in chunks)


class TestStructureAwareBehavior:
    def test_paragraph_boundaries_preserved(self, doc_factory):
        p1, p2 = para(2), para(3)
        document = doc_factory([[("PARAGRAPH", p1)], [("PARAGRAPH", p2)]])
        chunks = StructureAwareChunker().chunk_document(document)
        joined = "\n".join(c.text for c in chunks)
        assert p1 in joined and p2 in joined
        # each block appears intact in exactly one chunk
        holders = [sum(1 for c in chunks if p1 in c.text),
                   sum(1 for c in chunks if p2 in c.text)]
        assert holders == [1, 1]

    def test_heading_stays_with_following_content(self, doc_factory):
        document = doc_factory([
            [("HEADING", "The Shadow"),
             ("PARAGRAPH", para(6))],
            [("PARAGRAPH", para(8))],
        ])
        chunks = StructureAwareChunker().chunk_document(document)
        first = chunks[0]
        assert first.text.startswith("The Shadow")
        assert "p0001-b000" in first.source_block_ids
        assert any("Shadow" in hp for hp in first.heading_path)

    def test_max_token_enforcement(self, doc_factory):
        cfg = ChunkingConfig(target_tokens=120, max_tokens=150,
                             min_tokens=30, overlap_tokens=20)
        huge = " ".join(f"sentence number {i} about alchemy and the self." for i in range(200))
        document = doc_factory([[("PARAGRAPH", huge)]])
        chunks = StructureAwareChunker(cfg).chunk_document(document)
        assert len(chunks) > 1
        for c in chunks:
            assert c.token_count <= 152, f"{c.chunk_id} exceeded max: {c.token_count}"

    def test_min_tokens_respected_mid_section(self, doc_factory):
        cfg = ChunkingConfig(target_tokens=200, max_tokens=260,
                             min_tokens=60, overlap_tokens=25)
        paras = [("PARAGRAPH", para(1)) for _ in range(8)]  # ~35 tokens each
        document = doc_factory([paras])
        chunks = StructureAwareChunker(cfg).chunk_document(document)
        # only the final section-boundary chunk may fall below min
        for c in chunks[:-1]:
            assert c.token_count >= 55

    def test_multipage_chunk_provenance(self, doc_factory):
        document = doc_factory([
            [("PARAGRAPH", para(1))],
            [("PARAGRAPH", para(1))],
        ])
        cfg = ChunkingConfig(target_tokens=200, max_tokens=300,
                             min_tokens=50, overlap_tokens=20)
        chunks = StructureAwareChunker(cfg).chunk_document(document)
        merged = [c for c in chunks if len(c.page_numbers) == 2]
        assert merged, "expected at least one multi-page chunk"
        for c in merged:
            assert c.start_page == min(c.page_numbers)
            assert c.end_page == max(c.page_numbers)

    def test_source_blocks_and_pages_recorded(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = StructureAwareChunker().chunk_document(document)
        for c in chunks:
            assert c.source_block_ids
            assert c.created_from_blocks == c.source_block_ids
            for bid in c.source_block_ids:
                page_no = int(bid[1:5])
                assert page_no in c.page_numbers

    def test_chunk_text_derives_from_source_blocks(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = StructureAwareChunker().chunk_document(document)
        result = validate_chunks(chunks, document)
        assert result.ok, result.errors

    def test_stable_ordering(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks_a = StructureAwareChunker().chunk_document(simple_doc(doc_factory))
        chunks_b = StructureAwareChunker().chunk_document(simple_doc(doc_factory))
        keys_a = [(c.chunk_index, c.chunk_id, c.source_block_ids) for c in chunks_a]
        keys_b = [(c.chunk_index, c.chunk_id, c.source_block_ids) for c in chunks_b]
        assert keys_a == keys_b
        assert [c.chunk_index for c in chunks_a] == list(range(len(chunks_a)))

    def test_token_counts_accurate(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = StructureAwareChunker().chunk_document(document)
        for c in chunks:
            assert c.token_count == count_tokens(c.text)
            assert c.char_count == len(c.text)

    def test_metadata_and_source_type_propagation(self, doc_factory):
        document = doc_factory(
            [[("TITLE", "T"), ("PARAGRAPH", para(3))]],
            source_type="SECONDARY",
        )
        chunks = StructureAwareChunker().chunk_document(document)
        for c in chunks:
            assert c.source_type == SourceType.SECONDARY
            assert c.metadata["section_title"]
            assert c.strategy == ChunkingConfig().strategy_name


class TestProvenanceValidation:
    def _chunks(self, doc_factory):
        return StructureAwareChunker().chunk_document(simple_doc(doc_factory))

    def test_unknown_block_id_fails(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = self._chunks(doc_factory)
        chunks[0].source_block_ids.append("p9999-b999")
        result = validate_chunks(chunks, document)
        assert not result.ok
        from jung_archive.chunking.validation import require_valid
        with pytest.raises(ProvenanceError):
            require_valid(chunks, document)

    def test_tampered_token_count_fails(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = self._chunks(doc_factory)
        chunks[0].token_count += 100
        result = validate_chunks(chunks, document)
        assert not result.ok

    def test_duplicate_chunk_ids_fail(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = self._chunks(doc_factory)
        chunks[1].chunk_id = chunks[0].chunk_id
        result = validate_chunks(chunks, document)
        assert not result.ok

    def test_empty_text_fails(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = self._chunks(doc_factory)
        chunks[0].text = "   "
        result = validate_chunks(chunks, document)
        assert not result.ok

    def test_page_out_of_range_fails(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = self._chunks(doc_factory)
        chunks[0].page_numbers.append(999)
        result = validate_chunks(chunks, document)
        assert not result.ok

    def test_silent_source_loss_detected(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = self._chunks(doc_factory)[:-1]  # drop last chunk
        result = validate_chunks(chunks, document)
        assert not result.ok
        assert any("never appear" in e for e in result.errors)

    def test_valid_chunks_pass(self, doc_factory):
        document = simple_doc(doc_factory)
        chunks = self._chunks(doc_factory)
        assert validate_chunks(chunks, document).ok


class TestConfigSanity:
    def test_invalid_config_rejected(self):
        with pytest.raises(Exception):
            ChunkingConfig(min_tokens=0, target_tokens=10, overlap_tokens=50)
        with pytest.raises(Exception):
            ChunkingConfig(max_tokens=50, target_tokens=100)
