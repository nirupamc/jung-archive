"""Manual M2 verification: chunk -> source blocks -> pages -> original PDF."""
import json
import sys

import fitz

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANON = "data/processed/381d2da4b68e.json"
ART = "data/chunks/381d2da4b68e.json"
PDF = "primary/The Undiscovered Self.pdf"


def norm(s):
    return " ".join(s.split())


def main(chunk_ids):
    canon = json.load(open(CANON, encoding="utf-8"))
    art = json.load(open(ART, encoding="utf-8"))
    blocks_by_id = {
        b["block_id"]: b
        for p in canon["pages"]
        for b in p.get("blocks", [])
    }
    doc = fitz.open(PDF)

    for cid in chunk_ids:
        chunk = next(c for c in art["chunks"] if c["chunk_id"] == cid)
        print("=" * 72)
        print(f"CHUNK {cid}")
        print(f"  pages={chunk['page_numbers']} tokens={chunk['token_count']} "
              f"heading_path={chunk['heading_path']}")
        print(f"  metadata={chunk['metadata']}")
        print("  --- CHUNK TEXT (first 220 chars) ---")
        print("   ", norm(chunk["text"])[:220])
        missing = []
        for bid in chunk["source_block_ids"]:
            blk = blocks_by_id.get(bid)
            if blk is None:
                missing.append(bid)
                continue
            frag = norm(blk["text"])
            contained = frag[:40] in norm(chunk["text"]) or \
                norm(chunk["text"])[-40:] in frag
            print(f"   block {bid} type={blk['block_type']:<10} "
                  f"derived={contained} :: {frag[:70]!r}")
        if missing:
            print("   MISSING BLOCKS:", missing)
        first_page = min(chunk["page_numbers"])
        raw = norm(doc[first_page - 1].get_text("text"))
        sample = norm(chunk["text"])[:60]
        hit = sample[:30] in raw
        print(f"  PDF page {first_page} contains chunk start: {hit}")
    doc.close()


if __name__ == "__main__":
    ids = sys.argv[1:] or [
        "381d2da4b68e-c00004",  # early, under heading (carried part-number case)
        "381d2da4b68e-c00039",  # ordinary body chunk
        "381d2da4b68e-c00006",  # split-window oversized block (unusual)
        "381d2da4b68e-c00210",  # last chunk
    ]
    main(ids)
