import json
from pathlib import Path
from jung_archive.chunking.chunker import StructureAwareChunker
from jung_archive.chunking.validation import validate_chunks
from jung_archive.chunking.tokenizer import count_tokens
from jung_archive.models.document import Document
from jung_archive.chunking.tokenizer import count_tokens as ct

doc_id = "158b59f7a945"
processed_path = Path("data/processed") / f"{doc_id}.json"

with open(processed_path, encoding="utf-8") as f:
    data = json.load(f)

document = Document.model_validate(data)

chunker = StructureAwareChunker()
chunks = chunker.chunk_document(document)

# Find the specific failing chunk
bad_chunk = next((c for c in chunks if c.chunk_id == "158b59f7a945-c00066"), None)
if bad_chunk:
    print(f"Chunk: {bad_chunk.chunk_id}")
    print(f"  text: {bad_chunk.text[:200]}...")
    print(f"  source_block_ids: {bad_chunk.source_block_ids}")
    print(f"  text length: {len(bad_chunk.text)}")
    
    for bid in bad_chunk.source_block_ids:
        blk = next((b for p in document.pages for b in p.blocks if b.block_id == bid), None)
        if blk:
            frag = " ".join(blk.text.split())
            body = " ".join(bad_chunk.text.split())
            print(f"\n  Block {bid}:")
            print(f"    text: {blk.text[:100]}...")
            print(f"    frag_len: {len(frag)}")
            print(f"    frag[:40]: '{frag[:40]}'")
            print(f"    body starts: '{body[:80]}'")
            print(f"    body[-40:]: '{body[-40:]}'")
            print(f"    frag[:40] in body: {frag[:40] in body}")
            print(f"    body[-40:] in frag: {body[-40:] in frag}")
            print(f"    len(body) >= 40: {len(body) >= 40}")
        else:
            print(f"  Block {bid}: NOT FOUND")

# Also check what the validation result says
result = validate_chunks(chunks, document)
print(f"\nTotal validation errors: {len(result.errors)}")
print(f"Sample errors: {result.errors[:5]}")
