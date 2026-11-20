"""Centralized runtime configuration for data paths.

Local development defaults to ``<repo>/data``. Container deployments (Modal,
Docker, CI) override with ``JUNG_ARCHIVE_DATA_DIR`` environment variable so
the same application code locates persisted retrieval/index artifacts without
hardcoding deployment-specific paths.

Source PDFs are resolved separately via ``JUNG_ARCHIVE_PDF_DIR``. Local
development defaults to the repository root (where ``primary/`` and
``secondary/`` live). Production may point this at a directory that contains
those two folders, but the application never assumes ``/data`` contains PDFs
and never searches arbitrary filesystem locations.
"""
import os
from pathlib import Path, PurePosixPath

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


def get_pdf_dir() -> Path:
    """Resolve the root directory that may contain source PDFs.

    Defaults to the repository root for local development. Production can set
    ``JUNG_ARCHIVE_PDF_DIR`` to a directory with ``primary/`` and
    ``secondary/`` subfolders. The default is deliberately not ``DATA_DIR``:
    Modal's ``/data`` volume holds derived artifacts, not original PDFs.
    """
    env = os.environ.get("JUNG_ARCHIVE_PDF_DIR")
    if env:
        return Path(env)
    return REPO_ROOT


PDF_DIR = get_pdf_dir()


def resolve_pdf_path(source_path: object, pdf_root: object = None, **kwargs) -> Path | None:
    """Resolve *source_path* to an existing PDF under *pdf_root*.

    This is the single centralized PDF-path resolver used by both
    ``document_summary()`` (for ``has_pdf``) and ``document_pdf()`` (for the
    PDF serving route). It normalizes Windows and POSIX separators, handles
    stored Windows-absolute paths, supports already-relative ``primary/`` and
    ``secondary/`` paths, rejects unsafe paths and traversal attempts, and
    ensures the resolved file exists below the configured root.

    Returns ``None`` when the path cannot be normalized safely, the configured
    PDF root is missing/unreadable, the resolved file does not exist, or any
    attempt to escape the configured root is detected.
    """
    if pdf_root is None:
        pdf_root = (
            kwargs.get("pdf_root")
            or kwargs.get("pdf_dir")
            or kwargs.get("root")
            or kwargs.get("pdf_directory")
            or kwargs.get("directory")
        )
    if pdf_root is None:
        pdf_root = get_pdf_dir()

    relative = _normalize_pdf_relative(source_path)
    if relative is None:
        return None

    try:
        root = Path(pdf_root).resolve()
    except (OSError, RuntimeError, ValueError):
        return None
    if not root.exists() or not root.is_dir():
        return None

    try:
        candidate = (root / relative).resolve()
        candidate.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None

    if not candidate.is_file():
        return None
    return candidate


def _normalize_pdf_relative(source_path: object) -> PurePosixPath | None:
    """Return a safe ``primary/...`` or ``secondary/...`` relative path.

    Handles POSIX and Windows separators, Windows drive letters, and absolute
    POSIX paths by extracting the first ``primary``/``secondary`` segment and
    everything after it. Returns ``None`` for anything that cannot be reduced to
    a safe relative path under one of those two roots.
    """
    if source_path is None:
        return None
    raw = str(source_path).strip().strip('"').strip("'")
    if not raw:
        return None

    parts = [part for part in raw.replace("\\", "/").split("/") if part != ""]
    if not parts:
        return None

    start = None
    for index, part in enumerate(parts):
        if part.lower() in ("primary", "secondary"):
            start = index
            break
    if start is None:
        return None

    rel_parts = parts[start:]
    if not rel_parts or rel_parts[0].lower() not in ("primary", "secondary"):
        return None

    for part in rel_parts:
        if part in ("", ".", "..") or "/" in part or "\\" in part or ":" in part:
            return None

    relative = PurePosixPath(*rel_parts)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    if not relative.parts or relative.parts[0].lower() not in (
        "primary",
        "secondary",
    ):
        return None
    return relative


# Compatibility aliases for callers/tests that use a different helper name.
resolve_pdf = resolve_pdf_path
get_pdf_path = resolve_pdf_path
resolve_source_pdf = resolve_pdf_path
pdf_path_for = resolve_pdf_path