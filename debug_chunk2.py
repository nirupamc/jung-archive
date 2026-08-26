"""Test chunking on a document that was failing validation."""
import sys
sys.path.insert(0, "src")

from jung_archive.chunking.chunker import StructureAwareChunker
from jung_archive.chunking.validation import validate_chunks
from jung_archive.chunking.artifacts import load_chunk_artifact
from jung_archive.models.document import Document
from pathlib import Path
import json

doc_id = "183d18f3e973"  # Man and His Symbols
processed_path = Path("data/processed") / f"{doc_id}.json"

with open(processed_path, encoding="utf-8") as f:
    data = json.load(f)

document = Document.model_validate(data)

# Check what block types exist
from collections import Counter
block_types = Counter()
for page in document.pages:
    for blk in page.blocks:
        block_types[blk.block_type] += 1

print(f"Block types: {dict(block_types)}")

# Check header/footer/page_number blocks with text
non_content_with_text = 0
for page in document.pages:
    for blk in page.blocks:
        if blk.block_type in ("HEADER", "FOOTER", "PAGE_NUMBER"):
            if blk.text.strip():
                non_content_with_text += 1
print(f"Header/Footer/PageNumber blocks with text: {non_content_with_text}")

# Try chunking
chunker = StructureAwareChunker()
chunks = chunker.chunk_document(document)
print(f"\nChunks created: {len(chunks)}")

result = validate_chunks(chunks, document)
print(f"Validation OK: {result.ok}")
if not result.ok:
    print(f"Errors ({len(result.errors)}):")
    for e in result.errors[:10]:
        print(f"  {e}")
