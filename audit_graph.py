"""M7 manual audit: full provenance chain for one trusted graph edge."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from jung_archive.graph.build import load_graph  # noqa: E402

g = load_graph()
names = {n.node_id: n.canonical_name for n in g.nodes}
e = next(e for e in g.edges
         if e.status == "TRUSTED"
         and "concept:shadow" in (e.source_node_id, e.target_node_id))
print("EDGE:", names[e.source_node_id], e.relationship_type,
      names[e.target_node_id])
print("confidence:", e.confidence, "| status:", e.status,
      "| evidence_count:", e.evidence_count)
ev_index = {x.evidence_id: x for x in g.evidence}
ev = ev_index[e.evidence_ids[0]]
print()
print("EVIDENCE", ev.evidence_id)
print("  chunk :", ev.chunk_id, "pages", ev.page_numbers)
print("  blocks:", ev.source_block_ids)
print("  span  :", ev.preview(200))

art = json.load(open(f"data/chunks/{ev.document_id}.json", encoding="utf-8"))
chunk = next(c for c in art["chunks"] if c["chunk_id"] == ev.chunk_id)
print()
print("CHUNK in artifact:", chunk["chunk_id"], "pages",
      chunk["page_numbers"], "blocks", chunk["source_block_ids"])

import fitz  # noqa: E402

doc = fitz.open(art["document"]["source_path"])
probe = " ".join(ev.evidence_text.split())[:40].lower()
for pno in ev.page_numbers:
    t = doc[pno - 1].get_text().replace("\n", " ").lower()
    print(f"PDF page {pno} contains span start:",
          probe[:30] in t, f"('{probe[:30]}...')")
