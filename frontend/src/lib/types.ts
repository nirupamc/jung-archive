// Typed API contracts mirroring jung_archive.api.schemas / backend models.

export type BlockType =
  | "TITLE"
  | "HEADING"
  | "PARAGRAPH"
  | "LIST"
  | "TABLE"
  | "FIGURE"
  | "CAPTION"
  | "HEADER"
  | "FOOTER"
  | "PAGE_NUMBER"
  | "UNKNOWN";

export interface BBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export type CorpusStatus =
  | "DISCOVERED"
  | "REVIEW"
  | "EXCLUDED"
  | "PROCESSED"
  | "CHUNKED"
  | "INDEXED"
  | "ERROR";

export interface DocumentSummary {
  document_id: string;
  title: string | null;
  author: string | null;
  source_type: string;
  index_status: string;
  page_count: number;
  chunk_count: number;
  source_path: string | null;
  has_pdf: boolean;
  /** honest pipeline status (corpus discovery layer) */
  status: CorpusStatus;
  /** corpus lane: PRIMARY | SECONDARY | UNKNOWN */
  section: string;
  /** explicit registry decision recorded? */
  registered: boolean;
  registered_reason: string | null;
  sha256: string | null;
}

/** Documents whose full inspection pipeline can be browsed. */
export function isProcessedDoc(d: DocumentSummary | null): boolean {
  return !!d && ["PROCESSED", "CHUNKED", "INDEXED"].includes(d.status);
}

export interface BlockOut {
  block_id: string;
  block_type: BlockType;
  text: string;
  bbox: BBox;
  reading_order: number;
  extraction_method: string;
  confidence: number | null;
  heuristic_quality_score: number | null;
  font_name: string | null;
  font_size: number | null;
  page_number: number;
}

export interface PageInspection {
  document_id: string;
  page_number: number;
  width: number;
  height: number;
  classification: string;
  classification_confidence: number | null;
  classification_reason: string | null;
  layout: string;
  layout_confidence: number | null;
  layout_reason: string | null;
  ocr_confidence: number | null;
  warnings: string[];
  blocks: BlockOut[];
}

export interface ChunkOut {
  chunk_id: string;
  document_id: string;
  heading_path: string[];
  page_numbers: number[];
  token_count: number;
  source_type: string;
  source_block_ids: string[];
  strategy: string | null;
  section_id: string | null;
  chunk_index: number | null;
  start_page: number | null;
  end_page: number | null;
  char_count: number | null;
  text: string;
}

// ----------------------------------------------------------------------
// Retrieval + evidence contracts (mirror backend models)

export const RETRIEVAL_MODES = ["dense", "bm25", "hybrid", "hybrid_rerank"] as const;
export type RetrievalMode = (typeof RETRIEVAL_MODES)[number];

export interface RetrievalResult {
  chunk_id: string;
  document_id: string;
  text: string;
  page_numbers: number[];
  source_block_ids: string[];
  heading_path: string[];
  source_type: string;
  dense_rank: number | null;
  dense_score: number | null;
  bm25_rank: number | null;
  bm25_score: number | null;
  fusion_rank: number | null;
  fusion_score: number | null;
  reranker_rank: number | null;
  reranker_score: number | null;
  author: string | null;
  title: string | null;
  section_id: string | null;
}

export interface RetrievalResponse {
  query: string;
  mode: string;
  top_k: number;
  filters: Record<string, unknown>;
  results: RetrievalResult[];
  warnings: string[];
  latency_ms: number | null;
  candidates_retrieved: number | null;
  candidates_reranked: number | null;
  pairs_truncated: number | null;
}

export interface ScorePath {
  dense_rank: number | null;
  dense_score: number | null;
  bm25_rank: number | null;
  bm25_score: number | null;
  fusion_rank: number | null;
  fusion_score: number | null;
  reranker_rank: number | null;
  reranker_score: number | null;
}

export interface EvidenceItem {
  evidence_id: string;
  chunk_id: string;
  document_id: string;
  text: string;
  clean_text: string;
  page_numbers: number[];
  source_block_ids: string[];
  heading_path: string[];
  source_type: string;
  author: string | null;
  title: string | null;
  section_id: string | null;
  scores: ScorePath;
  token_count: number;
  was_cleaned: boolean;
  cleanup_operations: string[];
  duplicate_group: number | null;
  selection_reason: string;
}

export interface SuppressedItem {
  chunk_id: string;
  reason: string;
}

export interface EvidencePack {
  question: string;
  items: EvidenceItem[];
  tokens_used: number;
  max_evidence_tokens: number;
  max_evidence_items: number;
  candidates_considered: number;
  suppressed_duplicates: SuppressedItem[];
  suppressed_diversity: SuppressedItem[];
  skipped_oversized: SuppressedItem[];
  warnings: string[];
}
