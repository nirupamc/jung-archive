/**
 * Deterministic fixtures mirroring the backend API contracts.
 */
import type { BlockOut, ChunkOut, DocumentSummary, PageInspection } from "@/lib/types";
import type { AskResponse, CitationOut, EvidencePack, RetrievalResponse, RetrievalResult } from "@/lib/types";

export const DOC: DocumentSummary = {
  document_id: "381d2da4b68e",
  title: "The Undiscovered Self",
  author: "Carl Gustav Jung",
  source_type: "PRIMARY",
  index_status: "INCLUDE",
  page_count: 88,
  chunk_count: 211,
  source_path: "primary/The Undiscovered Self.pdf",
  has_pdf: true,
  status: "INDEXED",
  section: "PRIMARY",
  registered: true,
  registered_reason: null,
  sha256: "cc6a34980a755cd64dafe847cb5fe168e46bf56b089c4f75e87351c3372c8daf",
};

/** A discovered-but-unprocessed document (honest status screen case). */
export const UNPROCESSED_DOC: DocumentSummary = {
  document_id: "deadbeefcafe",
  title: "Jung's Answer to Job: A Commentary (publisher preview)",
  author: "Paul Bishop",
  source_type: "SECONDARY",
  index_status: "REVIEW",
  page_count: 24,
  chunk_count: 0,
  source_path: "primary/preview-9781317710721_A23899886.pdf",
  has_pdf: true,
  status: "REVIEW",
  section: "SECONDARY",
  registered: true,
  registered_reason:
    "Third-party academic commentary preview, not primary Jung material.",
  sha256: "f5dbec2cec7cc9e13fc7cf00a6efc85fda61f01a47a2843cdbf7ee4cb89d18c3",
};

export const PAGE_17: PageInspection = {
  document_id: DOC.document_id,
  page_number: 17,
  width: 360,
  height: 576,
  classification: "NATIVE",
  classification_confidence: 0.95,
  classification_reason: "sufficient native text detected",
  layout: "SINGLE_COLUMN",
  layout_confidence: 0.85,
  layout_reason: "no interior gutter in x-occupancy profile",
  ocr_confidence: null,
  warnings: [],
  blocks: [
    {
      block_id: "p0017-b000",
      block_type: "PARAGRAPH",
      text: "results of their investigations as though these had come into existence without man's intervention.",
      bbox: { x0: 50, y0: 70, x1: 310, y1: 200 },
      reading_order: 2,
      extraction_method: "NATIVE",
      confidence: null,
      heuristic_quality_score: null,
      font_name: "ScalaSans",
      font_size: 9.5,
      page_number: 17,
    },
    {
      block_id: "p0017-b001",
      block_type: "UNKNOWN",
      text: "the undiscovered self",
      bbox: { x0: 127, y0: 47, x1: 231, y1: 57 },
      reading_order: 1,
      extraction_method: "NATIVE",
      confidence: null,
      heuristic_quality_score: null,
      font_name: "ScalaSans-Caps",
      font_size: 9,
      page_number: 17,
    },
    {
      block_id: "p0017-b002",
      block_type: "PAGE_NUMBER",
      text: "9",
      bbox: { x0: 176, y0: 540, x1: 184, y1: 550 },
      reading_order: 3,
      extraction_method: "NATIVE",
      confidence: null,
      heuristic_quality_score: null,
      font_name: null,
      font_size: null,
      page_number: 17,
    },
  ],
};

export function blockFor(id: string): BlockOut {
  return PAGE_17.blocks.find((b) => b.block_id === id)!;
}

export const CHUNKS: ChunkOut[] = [
  {
    chunk_id: "381d2da4b68e-c00028",
    document_id: DOC.document_id,
    heading_path: ["Carl Gustav Jung", "THE PLIGHT OF THE INDIVIDUAL"],
    page_numbers: [18, 19],
    token_count: 37,
    source_type: "PRIMARY",
    source_block_ids: ["p0018-b002", "p0019-b001"],
    strategy: "structure_aware_v1",
    section_id: "381d2da4b68e-s0008",
    chunk_index: 28,
    start_page: 18,
    end_page: 19,
    char_count: 210,
    text: "Apart from agglomerations of huge masses of people, in which the individual disappears anyway.",
  },
  {
    chunk_id: "381d2da4b68e-c00102",
    document_id: DOC.document_id,
    heading_path: ["Carl Gustav Jung", "THE INDIVIDUAL'S UNDERSTANDING OF HIMSELF"],
    page_numbers: [49],
    token_count: 294,
    source_type: "PRIMARY",
    source_block_ids: ["p0049-b000"],
    strategy: "structure_aware_v1",
    section_id: "381d2da4b68e-s0021",
    chunk_index: 102,
    start_page: 49,
    end_page: 49,
    char_count: 1500,
    text: "our blindness in this respect is extremely dangerous.",
  },
];

function result(overrides: Partial<RetrievalResult>): RetrievalResult {
  return {
    chunk_id: "381d2da4b68e-c00028",
    document_id: DOC.document_id,
    text: "Apart from agglomerations of huge masses of people.",
    page_numbers: [18],
    source_block_ids: ["p0018-b002"],
    heading_path: ["Carl Gustav Jung", "THE PLIGHT OF THE INDIVIDUAL"],
    source_type: "PRIMARY",
    dense_rank: 1,
    dense_score: 0.603658,
    bm25_rank: 4,
    bm25_score: 10.552379,
    fusion_rank: 1,
    fusion_score: 0.028442,
    reranker_rank: null,
    reranker_score: null,
    author: "Carl Gustav Jung",
    title: "The Undiscovered Self",
    section_id: null,
    ...overrides,
  };
}

export const HYBRID_RESPONSE: RetrievalResponse = {
  query: "mass psychology",
  mode: "hybrid",
  top_k: 2,
  filters: {},
  results: [
    result({}),
    result({
      chunk_id: "381d2da4b68e-c00102",
      page_numbers: [49],
      source_block_ids: ["p0049-b000"],
      dense_rank: 2,
      dense_score: 0.408638,
      bm25_rank: 1,
      bm25_score: 13.343685,
      fusion_rank: 2,
      fusion_score: 0.029958,
    }),
  ],
  warnings: [],
  latency_ms: 42.5,
  candidates_retrieved: null,
  candidates_reranked: null,
  pairs_truncated: null,
};

export const RERANK_RESPONSE: RetrievalResponse = {
  ...HYBRID_RESPONSE,
  mode: "hybrid_rerank",
  top_k: 2,
  results: [
    result({ reranker_rank: 2, reranker_score: -1.53, fusion_rank: 2, fusion_score: 0.029958, dense_rank: null, dense_score: null }),
    result({ reranker_rank: 1, reranker_score: -1.69 }),
  ],
  candidates_retrieved: 20,
  candidates_reranked: 20,
  pairs_truncated: 0,
};

export const EVIDENCE_PACK: EvidencePack = {
  question: "mass psychology?",
  items: [
    {
      evidence_id: "S1",
      chunk_id: "381d2da4b68e-c00028",
      document_id: DOC.document_id,
      text: "the undiscovered self\n\nApart from agglomerations of huge masses of people.\n\n9",
      clean_text: "Apart from agglomerations of huge masses of people.",
      page_numbers: [18, 19],
      source_block_ids: ["p0018-b002", "p0019-b001"],
      heading_path: ["Carl Gustav Jung", "THE PLIGHT OF THE INDIVIDUAL"],
      source_type: "PRIMARY",
      author: "Carl Gustav Jung",
      title: "The Undiscovered Self",
      section_id: null,
      scores: {
        dense_rank: 4,
        dense_score: 0.603658,
        bm25_rank: 4,
        bm25_score: 10.552379,
        fusion_rank: 4,
        fusion_score: 0.028442,
        reranker_rank: 1,
        reranker_score: -1.532655,
      },
      token_count: 30,
      was_cleaned: true,
      cleanup_operations: [
        "removed_running_header:line_8",
        "removed_folio:trailing_line_6",
      ],
      duplicate_group: 1,
      selection_reason: "reranked_relevance",
    },
  ],
  tokens_used: 1634,
  max_evidence_tokens: 2500,
  max_evidence_items: 8,
  candidates_considered: 8,
  suppressed_duplicates: [
    { chunk_id: "381d2da4b68e-c00027", reason: "duplicate_of:381d2da4b68e-c00028:text_overlap:0.87" },
  ],
  suppressed_diversity: [],
  skipped_oversized: [],
  warnings: [],
};

export const ASK_RESPONSE: AskResponse = {
  answer:
    "Jung treats the Self as a totality larger than the conscious ego [S1]. He also connects individuation with the integration of unconscious contents [S2].",
  citations: [
    {
      id: "[S1]",
      evidence_id: "S1",
      status: "valid",
      note: null,
    },
    {
      id: "[S2]",
      evidence_id: "S2",
      status: "unknown",
      note: "S2 not in evidence pack",
    },
  ],
  evidence_pack: EVIDENCE_PACK,
  provider: "openai_compatible",
  model: "llama-3-8b-instruct",
  local_or_remote: "REMOTE",
  retrieval_metadata: {
    mode: "hybrid",
    top_k: 20,
    latency_ms: 142.3,
    results: 12,
    warnings: [],
  },
  warnings: [
    "generation provider is REMOTE; corpus evidence is being sent off-machine",
    "generated answer references 1 unknown citation(s): S2",
  ],
};

