import type {
  AskResponse,
  BlockOut,
  ChunkOut,
  DocumentSummary,
  EvidencePack,
  PageInspection,
  RetrievalResponse,
} from "./types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export class ApiError extends Error {
  status: number;
  detail: string | null;
  constructor(status: number, message: string, detail: string | null = null) {
    super(message);
    this.status = status;
    this.detail = detail;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch (e) {
    throw new ApiError(0, `backend unreachable at ${API_BASE}`, String(e));
  }
  if (!res.ok) {
    let detail: string | null = null;
    try {
      const body = await res.json();
      detail = body.detail ?? body.error ?? null;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, `${res.status} ${res.statusText}`, detail);
  }
  return (await res.json()) as T;
}

export function pdfUrl(documentId: string): string {
  return `${API_BASE}/api/documents/${documentId}/pdf`;
}

export const api = {
  documents: () => request<DocumentSummary[]>("/api/documents"),
  document: (id: string) =>
    request<DocumentSummary>(`/api/documents/${id}`),
  page: (id: string, page: number) =>
    request<PageInspection>(`/api/documents/${id}/pages/${page}`),
  structure: (id: string) =>
    request<BlockOut[]>(`/api/documents/${id}/structure`),
  chunks: (id: string) =>
    request<ChunkOut[]>(`/api/documents/${id}/chunks`),
  pdfUrl: (id: string) => `${API_BASE}/api/documents/${id}/pdf`,
  search: (body: {
    query: string;
    mode: string;
    top_k: number;
    filters?: Record<string, unknown>;
  }) =>
    request<RetrievalResponse>(
      "/api/retrieval/search",
      { method: "POST", body: JSON.stringify(body) },
    ),
  evidence: (body: {
    question: string;
    top_k: number;
    max_tokens: number;
    max_items: number;
    filters?: Record<string, unknown>;
  }) => request<EvidencePack>("/api/evidence/assemble", {
    method: "POST",
    body: JSON.stringify(body),
  }),
  ask: (body: {
    query: string;
    filters?: Record<string, unknown>;
    generation?: Record<string, unknown>;
  }) =>
    request<AskResponse>("/api/ask", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  evaluationRuns: () =>
    request<import("./evaluations").EvalRunListItem[]>(
      "/api/evaluation/runs"),
  evaluationLatest: () =>
    request<import("./evaluations").EvalLatestSummary>(
      "/api/evaluation/latest"),
  evaluationRun: (id: string) =>
    request<import("./evaluations").EvalRunDetail>(
      `/api/evaluation/runs/${encodeURIComponent(id)}`),
  graphOverview: () =>
    request<import("./graph").GraphOverview>("/api/graph"),
  graphNode: (id: string) =>
    request<import("./graph").GraphNodeDetail>(
      `/api/graph/nodes/${id}`),
  graphEdge: (id: string) =>
    request<import("./graph").GraphEdgeDetail>(
      `/api/graph/edges/${encodeURIComponent(id)}`),
  graphSearch: (q: string) =>
    request<{ results: import("./graph").GraphSearchResult[] }>(
      `/api/graph/search?q=${encodeURIComponent(q)}`),
};

export type {
  DocumentSummary,
  PageInspection,
  BlockOut,
  ChunkOut,
  EvidencePack,
  RetrievalResponse,
} from "./types";
