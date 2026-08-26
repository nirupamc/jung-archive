# Jung Archive — Final Architecture (M7)

```mermaid
flowchart TD
    PDF["Source PDFs<br/>(primary/ + secondary/)"] --> INGEST["PDF Ingestion<br/>native / OCR routing"]
    REG["Document Registry<br/>config/document_metadata.json<br/>INCLUDE / REVIEW / EXCLUDE"] --> GATE{"index_status?"}
    PDF --> GATE
    GATE -- EXCLUDE/REVIEW --> STOP["never indexed"]
    GATE -- INCLUDE --> IR

    INGEST --> IR["Canonical IR<br/>pages · blocks · bboxes<br/>layout · reading order · typing"]
    IR --> CHUNKS["Structure-aware chunks<br/>provenance: chunk → blocks → page"]
    CHUNKS --> DENSE["Dense<br/>MiniLM + Chroma"]
    CHUNKS --> BM25["BM25<br/>rank-bm25, lexical-v1"]
    CHUNKS --> GRAPH["KNOWLEDGE GRAPH<br/>concepts · typed edges<br/>heuristic confidence · evidence"]

    DENSE --> RRF
    BM25 --> RRF["RRF Fusion (ranks only)"]
    RRF --> RERANK["Cross-encoder reranker<br/>(hybrid_rerank mode)"]
    RERANK --> EVIDENCE
    RRF --> PLAIN["plain hybrid mode"]
    GRAPH -- "evidence spans" --> EVIDENCE

    subgraph EVIDENCE["Evidence assembly"]
        CLEAN["cleanup (source-preserving)"]
        DEDUP["dedup + diversity"]
        BUDGET["token budget · S1..Sn ids"]
        CLEAN --> DEDUP --> BUDGET
    end

    EVIDENCE --> UI["Inspection frontend<br/>DOCUMENT · STRUCTURE · CHUNKS ·<br/>RETRIEVAL · EVALUATION · GRAPH"]
    EVAL["Evaluation Lab<br/>Hit@K · Recall@K · P@K · MRR · NDCG<br/>failure analysis · ablations"]
    CHUNKS --> EVAL
    EVIDENCE --> EVAL
    API["Read-only FastAPI<br/>documents · pages · chunks · pdf<br/>search · evidence · evaluation · graph"]
    UI --- API
    EVAL --- API
```

## Provenance invariant

Every trusted graph edge and every evidence item resolves to:

```text
graph node/edge
  → GraphEvidence (immutable span)
    → chunk artifact
      → source block IDs
        → page number(s)
          → original PDF region
```

Anything that cannot be traced is marked WEAK/UNVERIFIED or excluded.
```
