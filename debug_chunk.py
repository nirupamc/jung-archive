import traceback
from pathlib import Path
from jung_archive.chunking.chunker import StructureAwareChunker
from jung_archive.chunking.validation import validate_chunks
from jung_archive.chunking.tokenizer import count_tokens
from jung_archive.models.document import Document
import json

doc_id = "158b59f7a945"  # CW Vol 12
processed_path = Path("data/processed") / f"{doc_id}.json"

with open(processed_path, encoding="utf-8") as f:
    data = json.load(f)

document = Document.model_validate(data)

print(f"Document: {document.title}")
print(f"Pages: {document.page_count}")
print(f"Blocks per page (first 3):")
for p in document.pages[:3]:
    text_blocks = [b for b in p.blocks if b.text.strip()]
    print(f"  Page {p.page_number}: {len(p.blocks)} blocks, {len(text_blocks)} with text")

# Count total blocks and content blocks
total_blocks = 0
content_blocks = 0
empty_blocks = 0
for p in document.pages:
    for b in p.blocks:
        total_blocks += 1
        if b.text.strip():
            content_blocks += 1
        else:
            empty_blocks += 1
print(f"\nTotal blocks: {total_blocks}, content: {content_blocks}, empty: {empty_blocks}")

# Try chunking
try:
    chunker = StructureAwareChunker()
    chunks = chunker.chunk_document(document)
    print(f"\nChunks created: {len(chunks)}")
    
    # Validate
    result = validate_chunks(chunks, document)
    print(f"Validation OK: {result.ok}")
    if not result.ok:
        for err in result.errors[:10]:
            print(f"  ERROR: {err}")
        
        # Let's investigate the first error
        if result.errors:
            first_error = result.errors[0]
            print(f"\nAnalyzing first error: {first_error}")
            
            # Check if it's about text derivation
            if "text not derived" in first_error:
                for chunk in chunks[:3]:
                    print(f"\n  Chunk {chunk.chunk_id}:")
                    print(f"    text preview: {chunk.text[:100]}...")
                    print(f"    source_block_ids: {chunk.source_block_ids}")
                    for bid in chunk.source_block_ids[:5]:
                        blk = next((b for p in document.pages for b in p.blocks if b.block_id == bid), None)
                        if blk:
                            frag = " ".join(blk.text.split())
                            body = " ".join(chunk.text.split())
                            head_hit = frag[:40] in body if len(frag) >= 20 else True
                            tail_hit = len(body) >= 40 and body[-40:] in frag if len(frag) >= 20 else True
                            print(f"    block {bid}: frag_len={len(frag)} text_preview={frag[:60]}...")
                            if len(frag) >= 20:
                                print(f"    head_hit={head_hit} tail_hit={tail_hit}")
except Exception as e:
    print(f"ERROR: {e}")
    traceback.print_exc()
