"""Test corpus ingestion for a single INCLUDE document."""
import sys
sys.path.insert(0, "src")

from jung_archive.ingestion.batch import ingest_batch
from jung_archive.corpus import discover_corpus
import json

# Just test with a small document first - The Undiscovered Self (already indexed)
# and then Man and His Symbols
docs = discover_corpus()
for d in docs:
    if d.index_status == "INCLUDE" and d.document_id:
        print(f"{d.document_id[:12]} {d.path} status={d.status} pages={d.page_count}")

# Try ingesting all INCLUDE documents
report = ingest_batch(
    sections=("primary", "secondary"),
    force_index=False,
    progress=lambda msg: print(f"  {msg}"),
)

print(json.dumps(report, indent=2))
