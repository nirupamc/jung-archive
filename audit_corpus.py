"""Audit all candidate PDFs: path, embedded title/author, first-page text."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import fitz


def hashlib_sha(p):
    import hashlib

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


out = []
for section in ("primary", "secondary"):
    for pdf in sorted((Path(".") / section).glob("*.pdf")):
        try:
            doc = fitz.open(pdf)
            meta = doc.metadata or {}
            page_count = doc.page_count
            first = doc[0].get_text()[:220].replace("\n", " ")
            second = doc[min(2, page_count - 1)].get_text()[:160].replace("\n", " ")
            sha = hashlib_sha(pdf)
            out.append({
                "path": str(pdf),
                "section": section,
                "sha256": sha,
                "pages": page_count,
                "embedded_title": meta.get("title", ""),
                "embedded_author": meta.get("author", ""),
                "first_page": first,
                "page3": second,
            })
            doc.close()
        except Exception as e:
            out.append({"path": str(pdf), "error": str(e)})

print(json.dumps(out, indent=1, ensure_ascii=False))


def hashlib_sha(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
