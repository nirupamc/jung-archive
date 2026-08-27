"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useReducer,
} from "react";
import { api } from "@/lib/api";
import type {
  AskResponse,
  DocumentSummary,
  EvidencePack,
  RetrievalMode,
  RetrievalResponse,
} from "@/lib/types";

export type Tab = "ask" | "document" | "structure" | "chunks" | "retrieval"
  | "evaluation" | "graph";

export interface WorkspaceState {
  documents: DocumentSummary[] | null;
  documentsError: string | null;
  documentId: string | null;
  page: number;
  selectedBlockIds: string[];
  selectedChunkId: string | null;
  tab: Tab;
  query: string;
  mode: RetrievalMode;
  topK: number;
  retrieval: RetrievalResponse | null;
  retrievalLoading: boolean;
  retrievalError: string | null;
  evidence: EvidencePack | null;
  evidenceLoading: boolean;
  evidenceError: string | null;
  ask: import("@/lib/types").AskResponse | null;
  askLoading: boolean;
  askError: string | null;
}

type Action =
  | { type: "documents_loaded"; docs: DocumentSummary[] }
  | { type: "documents_error"; error: string }
  | { type: "select_document"; id: string }
  | { type: "set_page"; page: number }
  | { type: "select_blocks"; ids: string[] }
  | { type: "select_chunk"; id: string | null }
  | { type: "set_tab"; tab: Tab }
  | { type: "set_query"; query: string }
  | { type: "set_mode"; mode: RetrievalMode }
  | { type: "set_top_k"; topK: number }
  | { type: "retrieval_start" }
  | { type: "retrieval_done"; result: RetrievalResponse }
  | { type: "retrieval_error"; error: string }
  | { type: "evidence_start" }
  | { type: "evidence_done"; pack: EvidencePack }
  | { type: "evidence_error"; error: string }
  | { type: "ask_start" }
  | { type: "ask_done"; result: import("@/lib/types").AskResponse }
  | { type: "ask_error"; error: string };

const initialState: WorkspaceState = {
  documents: null,
  documentsError: null,
  documentId: null,
  page: 1,
  selectedBlockIds: [],
  selectedChunkId: null,
  tab: "ask",
  query: "",
  mode: "hybrid",
  topK: 5,
  retrieval: null,
  retrievalLoading: false,
  retrievalError: null,
  evidence: null,
  evidenceLoading: false,
  evidenceError: null,
  ask: null,
  askLoading: false,
  askError: null,
};

function reducer(state: WorkspaceState, action: Action): WorkspaceState {
  switch (action.type) {
    case "documents_loaded":
      return { ...state, documents: action.docs, documentsError: null };
    case "documents_error":
      return { ...state, documentsError: action.error };
    case "select_document":
      return {
        ...state,
        documentId: action.id,
        page: 1,
        selectedBlockIds: [],
        selectedChunkId: null,
        retrieval: null,
        evidence: null,
        retrievalError: null,
        evidenceError: null,
        ask: null,
        askError: null,
      };
    case "set_page":
      return { ...state, page: Math.max(1, action.page) };
    case "select_blocks":
      return { ...state, selectedBlockIds: action.ids };
    case "select_chunk":
      return { ...state, selectedChunkId: action.id };
    case "set_tab":
      return { ...state, tab: action.tab };
    case "set_query":
      return { ...state, query: action.query };
    case "set_mode":
      return { ...state, mode: action.mode };
    case "set_top_k":
      return { ...state, topK: action.topK };
    case "retrieval_start":
      return { ...state, retrievalLoading: true, retrievalError: null };
    case "retrieval_done":
      return {
        ...state,
        retrievalLoading: false,
        retrieval: action.result,
      };
    case "retrieval_error":
      return { ...state, retrievalLoading: false, retrievalError: action.error };
    case "evidence_start":
      return { ...state, evidenceLoading: true, evidenceError: null };
    case "evidence_done":
      return { ...state, evidenceLoading: false, evidence: action.pack };
    case "evidence_error":
      return { ...state, evidenceLoading: false, evidenceError: action.error };
    case "ask_start":
      return { ...state, askLoading: true, askError: null };
    case "ask_done":
      return { ...state, askLoading: false, ask: action.result };
    case "ask_error":
      return { ...state, askLoading: false, askError: action.error };
    default:
      return state;
  }
}

export interface TraceTarget {
  pageNumbers: number[];
  blockIds: string[];
  chunkId?: string | null;
}

interface WorkspaceApi {
  state: WorkspaceState;
  dispatch: React.Dispatch<Action>;
  selectDocument: (id: string) => void;
  setPage: (page: number) => void;
  selectBlocks: (ids: string[]) => void;
  traceToSource: (t: TraceTarget) => void;
  runSearch: () => Promise<void>;
  assembleEvidence: (question: string) => Promise<void>;
  ask: () => Promise<void>;
}

const WorkspaceContext = createContext<WorkspaceApi | null>(null);

export function WorkspaceProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    api
      .documents()
      .then((docs) => {
        dispatch({ type: "documents_loaded", docs });
        if (docs.length > 0) {
          const preferred =
            docs.find((d) => d.title === "The Undiscovered Self") ?? docs[0];
          dispatch({ type: "select_document", id: preferred.document_id });
        }
      })
      .catch((e: Error) =>
        dispatch({ type: "documents_error", error: e.message }),
      );
  }, []);

  const traceToSource = useCallback((t: TraceTarget) => {
    if (t.chunkId) dispatch({ type: "select_chunk", id: t.chunkId });
    if (t.pageNumbers.length > 0) {
      dispatch({ type: "set_page", page: Math.min(...t.pageNumbers) });
    }
    dispatch({ type: "select_blocks", ids: t.blockIds });
    dispatch({ type: "set_tab", tab: "document" });
  }, []);

  const runSearch = useCallback(async () => {
    dispatch({ type: "retrieval_start" });
    try {
      const result = await api.search({
        query: state.query,
        mode: state.mode,
        top_k: state.topK,
      });
      dispatch({ type: "retrieval_done", result });
    } catch (e) {
      dispatch({
        type: "retrieval_error",
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }, [state.query, state.mode, state.topK]);

  const assembleEvidence = useCallback(
    async (question: string) => {
      dispatch({ type: "evidence_start" });
      try {
        const pack = await api.evidence({
          question,
          top_k: 8,
          max_tokens: 2500,
          max_items: 8,
        });
        dispatch({ type: "evidence_done", pack });
      } catch (e) {
        dispatch({
          type: "evidence_error",
          error: e instanceof Error ? e.message : String(e),
        });
      }
    },
    [],
  );

  const askCallback = useCallback(async () => {
    dispatch({ type: "ask_start" });
    try {
      const result = await api.ask({
        query: state.query,
        filters: {},
        generation: {},
      });
      dispatch({ type: "ask_done", result });
    } catch (e) {
      dispatch({
        type: "ask_error",
        error: e instanceof Error ? e.message : String(e),
      });
    }
  }, [state.query]);

  const value = useMemo<WorkspaceApi>(
    () => ({
      state,
      dispatch,
      selectDocument: (id: string) => dispatch({ type: "select_document", id }),
      setPage: (page: number) => dispatch({ type: "set_page", page }),
      selectBlocks: (ids: string[]) => dispatch({ type: "select_blocks", ids }),
      traceToSource,
      runSearch,
      assembleEvidence,
      ask: askCallback,
    }),
    [state, traceToSource, runSearch, assembleEvidence, askCallback],
  );

  return (
    <WorkspaceContext.Provider value={value}>
      {children}
    </WorkspaceContext.Provider>
  );
}

export function useWorkspace(): WorkspaceApi {
  const ctx = useContext(WorkspaceContext);
  if (!ctx) throw new Error("useWorkspace outside WorkspaceProvider");
  return ctx;
}
