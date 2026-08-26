# Jung Archive

A document-intelligence and retrieval-inspection workstation for the Carl
Jung corpus — built to make every stage of processing **observable**,
**traceable**, and **measurable**: from PDF pages to parsed blocks,
chunks, hybrid retrieval, cross-encoder reranking, evidence assembly,
evaluation, and an evidence-backed knowledge graph.

This is deliberately *not* another "chat with your PDFs" demo. There is no
answer-generation step and no chatbot: the system's purpose is document
intelligence plus observable retrieval, with every claim connected to its
source.

---

## Why it exists

Most RAG demos hide everything between "PDF in" and "answer out". Jung
Archive exposes the machinery:

* how a PDF was classified, laid out, and typed into blocks
* how blocks became provenance-complete chunks
* how dense, lexical, and fused retrieval rank candidates
* whether reranking actually helps (it is measured, not assumed)
* how evidence packs are budgeted and cleaned
* which graph relationships are trustworthy — and what evidence supports them

## Architecture

```text
PDF
 ↓
classification / OCR routing
 ↓
layout + reading order + block typing      (canonical IR)
 ↓
structure-aware chunking                   (provenance-complete artifacts)
 ↓
┌──────────────┬──────────────┬─────────────────┐
│ DENSE        │ BM25         │ KNOWLEDGE GRAPH  │
│ (MiniLM +    │ (rank-bm25)  │ concepts/edges   │
│  Chroma)     │              │ w/ evidence      │
└──────┬───────┴──────┬───────┴────────┬────────┘
       ↓              │                │
   RRF FUSION         │                │
       ↓              │                │
 CROSS-ENCODER        │                │
 RERANKER             │                │
       ↓              ↓                ↓
    EVIDENCE PACK ←───────────────────┘
    cleanup · dedup · diversity · token budget
       ↓
 INSPECTION UI  +  EVALUATION LAB
```

Final architecture notes:

* Retrieval scores are never mixed across systems; fusion uses ranks only.
* Every trusted graph relationship carries immutable evidence linking
  `edge → evidence → chunk → block → page → PDF`.
* Heuristic graph confidence is labeled as a heuristic score, never a
  probability.
* Generation / answering is intentionally out of scope for M1–M7.

## The six views

| View | What it shows |
|---|---|
| **DOCUMENT** | rendered PDF page with canonical block-type overlays and a per-page inspector |
| **STRUCTURE** | the canonical M1 block flow; click any item to jump to its source |
| **CHUNKS** | M2 chunk browser with heading/page/token metadata and block-level tracing |
| **RETRIEVAL** | dense / BM25 / hybrid / hybrid+rerank pipelines with full score paths, plus budgeted evidence packs |
| **EVALUATION** | benchmark overview, run-vs-run deltas (regressions labeled), failure inspector |
| **GRAPH** | evidence-backed concept neighborhood, node/edge details, concept search, filters, source tracing |

The core interaction throughout is one click from any result, evidence
item, or graph relation to the highlighted region of the original PDF:
`result → chunk → blocks → page → PDF`.

## Measured benchmark (M6)

On the 30-question manually grounded benchmark of
*The Undiscovered Self* (ground truth set by reading the source PDF, never
from the retriever under test), identical dataset and K values:

| Mode | Hit@1 | Recall@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| Dense | 0.433 | 0.606 | 0.602 | 0.524 |
| BM25 | 0.567 | 0.711 | 0.685 | 0.648 |
| Hybrid | 0.500 | 0.631 | 0.654 | 0.567 |
| Hybrid + Reranker | **0.767** | **0.783** | **0.853** | **0.761** |

These numbers are scoped to this corpus and benchmark only — they are not
general claims. Full per-query results, failure analysis, and chunk-size
experiments live in `data/evaluation/`.

## Knowledge graph

Built deterministically by `python -m jung_archive.cli graph build` from a
curated seed vocabulary (aliases included) over canonical chunks:

* conservative relation rules only: sentence co-occurrence (+ optional
  explicit patterns like *part of*, *contrasts with*, *symbolizes*),
  same-block, same-chunk — nothing weaker
* heuristic confidence documented in code; edges are TRUSTED / WEAK /
  UNVERIFIED, and TRUSTED requires supporting evidence
* persisted as JSON under `data/graph/` with schema/vocabulary/extractor
  versions and corpus fingerprint; stale graphs are detected, not served

Current build on the verified corpus: **17 nodes · 84 edges (42 TRUSTED) ·
538 evidence spans**, every trusted edge traceable to chunks, blocks, and
PDF pages.

## Running locally (Windows / PowerShell)

Requires Python 3.10+, Node 18+.

```powershell
# backend
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e .[rerank,api,test]
python -m jung_archive.cli index "primary/The Undiscovered Self.pdf"
python -m uvicorn jung_archive.api.app:app --port 8000

# frontend (second terminal)
cd frontend
npm install
npm run dev          # http://localhost:3000
```

Optional extras:

* OCR: install Tesseract system-wide (`pip install -e .[ocr]`); missing
  Tesseract degrades gracefully — OCR_REQUIRED pages keep their class and
  record warnings instead of fabricated text.
* Reranker/graph models download once from Hugging Face into the local
  cache on first use (`pip install -e .[rerank]`).

Evaluation & graph commands:

```powershell
python create_benchmark.py
python -m jung_archive.cli eval retrieval --dataset data/evaluation/dataset.json
python -m jung_archive.cli eval compare <run_a> <run_b>
python -m jung_archive.cli eval chunksize --target-tokens 150
python -m jung_archive.cli graph build
```

## Tests

```powershell
python -m pytest -q          # backend (271 tests)
cd frontend
npx vitest run               # frontend (34 tests)
npm run lint ; npm run build
```

## Document identity policy

Folder location never grants trust. Unregistered documents are
`index_status=REVIEW`; explicit decisions live in
`config/document_metadata.json` (the Bookey "Aion" summary is registered
EXCLUDE so it can never enter the index as Jung-authored material).
Author/title filters are enforced against this registry.

## Limitations

* Single-document index (*The Undiscovered Self*) is fully verified;
  multi-document indexing works mechanically but is not yet evaluated.
* Section hierarchy is one heading level deep (M1 typing limitation).
* Page furniture inside chunk interiors is left untouched by evidence
  cleanup (conservative by design); boundary folios/running heads are removed.
* Graph relations are co-occurrence-based; they show association supported
  by text spans, not semantic certainty. WEAK/UNVERIFIED edges are shown
  but clearly marked.
* Persona does not appear as a node because this essay barely uses the term.
* Embedding-model comparison is implemented but not executed.
* Generation metrics: NOT RUN — no generation system exists in M1–M7.
* No browser-automation suite (Playwright) installed; interactive flows were
  verified via API/SSR/component tests instead.
