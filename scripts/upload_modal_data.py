"""Upload precomputed Jung Archive runtime artifacts into the Modal Volume.

This transfers ONLY the read-only retrieval/index data the live API needs:
    data/chroma, data/bm25, data/chunks, data/graph, data/processed, data/evaluation

It deliberately EXCLUDES:
    primary/, secondary/   (source PDFs)
    .env                   (secrets)
    diagnostics/, experiments/, images/  (temp/diagnostic data)
    frontend/, src/, tests/  (code)

The operation is repeatable and idempotent (uses `modal volume put --force`).
It does NOT require local Modal app deploy; it only needs `modal` installed and
authenticated (`modal setup`).

Usage:
    python scripts/upload_modal_data.py
    python scripts/upload_modal_data.py --volume jung-archive-data
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# (local relative path, volume-relative remote path)
ARTIFACTS = [
    ("data/chroma", "chroma"),
    ("data/bm25", "bm25"),
    ("data/chunks", "chunks"),
    ("data/graph", "graph"),
    ("data/processed", "processed"),
    ("data/evaluation", "evaluation"),
]

EXCLUDED_HINT = (
    "Excluded by design: primary/, secondary/, .env, diagnostics/, "
    "experiments/, images/, frontend/, src/, tests/."
)


def run(cmd: list[str]) -> int:
    print("+ " + " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--volume", default="jung-archive-data",
        help="Name of the Modal Volume to upload into.")
    parser.add_argument(
        "--force", action="store_true", default=True,
        help="Overwrite existing files in the volume (default: on).")
    args = parser.parse_args()

    print(EXCLUDED_HINT)

    rc = 0
    for local_rel, remote_rel in ARTIFACTS:
        local = REPO_ROOT / local_rel
        if not local.exists():
            print(f"! skip (missing locally): {local_rel}")
            continue
        cmd = ["modal", "volume", "put"]
        if args.force:
            cmd.append("--force")
        cmd += [args.volume, str(local), remote_rel]
        code = run(cmd)
        if code != 0:
            print(f"!! failed to upload {local_rel} (exit {code})")
            rc = code
    if rc == 0:
        print("\nDone. Volume is ready for `modal deploy modal_app.py`.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
