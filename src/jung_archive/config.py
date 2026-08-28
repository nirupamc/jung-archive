"""Centralized runtime configuration for data paths.

Local development defaults to ``<repo>/data``. Container deployments (Modal,
Docker, CI) override with the ``JUNG_ARCHIVE_DATA_DIR`` environment variable so
the same application code locates persisted retrieval/index artifacts without
hardcoding deployment-specific paths.

This module must stay importable with no heavy side effects (no model loads,
no Chroma clients) so it can be imported by tests and by the Modal entrypoint.
"""
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_data_dir() -> Path:
    """Resolve the root directory containing all persisted runtime artifacts.

    Defaults to ``<repo>/data`` for local development. When
    ``JUNG_ARCHIVE_DATA_DIR`` is set (e.g. ``/data`` on Modal), every runtime
    artifact under it is resolved from that root, leaving local behavior intact.
    """
    env = os.environ.get("JUNG_ARCHIVE_DATA_DIR")
    if env:
        return Path(env)
    return REPO_ROOT / "data"


DATA_DIR = get_data_dir()
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
CHROMA_DIR = DATA_DIR / "chroma"
BM25_DIR = DATA_DIR / "bm25"
GRAPH_DIR = DATA_DIR / "graph"
EVAL_DIR = DATA_DIR / "evaluation"
