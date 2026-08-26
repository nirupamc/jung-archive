"""Manual verification helper: compare canonical JSON vs source PDF."""
import json
import sys

import fitz

PDF = "primary/Aion PDF.pdf"
CANON = sys.argv[1] if len(sys.argv) > 1 else "data/processed/c268c6e9dd9c.json"


def dump_page(pno):
    doc = fitz.open(PDF)
    page = doc[pno - 1]
    raw_text = page.get_text("text")
    native_chars = len([c for c in raw_text if not c.isspace()])
    imgs = page.get_images(full=True)
    doc.close()

    with open(CANON, encoding="utf-8") as f:
        canon = json.load(f)
    cp = canon["pages"][pno - 1]
    assert cp["page_number"] == pno, "page numbering mismatch"

    print("=" * 70)
    print(f"PAGE {pno}  ({cp['width']:.0f}x{cp['height']:.0f})")
    print(f"  classification={cp['classification']} conf={cp['classification_confidence']}")
    print(f"  reason={cp['reason']!r}")
    print(f"  layout={cp['layout']} conf={cp['layout_confidence']} reason={cp.get('layout_reason')!r}")
    print(f"  SOURCE: native_chars={native_chars} images={len(imgs)}")
    print(f"  CANON : blocks={len(cp['blocks'])} warnings={cp['warnings']}")
    ordered = sorted(cp["blocks"], key=lambda b: b["reading_order"])
    for b in ordered[:6]:
        t = " ".join(b["text"].split())[:60]
        bb = b["bbox"]
        print(
            f"   [{b['reading_order']:>2}] {b['block_type']:<10} "
            f"({bb['x0']:6.1f},{bb['y0']:6.1f},{bb['x1']:6.1f},{bb['y1']:6.1f}) {t!r}"
        )
    if len(ordered) > 6:
        last = ordered[-1]
        t = " ".join(last["text"].split())[:60]
        print(f"   ... [{last['reading_order']:>2}] {t!r}")
    raw_first = " ".join(raw_text.split())[:80]
    print(f"  RAW TEXT START: {raw_first!r}")


if __name__ == "__main__":
    pages = [int(a) for a in sys.argv[2:]] or [2, 31, 8, 47, 150]
    for p in pages:
        dump_page(p)
