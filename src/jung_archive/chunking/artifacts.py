"""Persist and reload chunk artifacts (data/chunks/<document_id>.json)."""
import json
from pathlib import Path
from typing import List, Optional

from jung_archive.models.chunk import Chunk, ChunkingConfig
from jung_archive.models.document import Document

ARTIFACT_FORMAT_VERSION = "chunk-artifact-1"


def save_chunk_artifact(
    chunks: List[Chunk],
    document: Document,
    config: ChunkingConfig,
    output_dir: str = "data/chunks",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{document.document_id}.json"

    artifact = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "document": {
            "document_id": document.document_id,
            "title": document.title,
            "author": document.author,
            "source_type": document.source_type.value,
            "index_status": document.index_status.value,
            "source_path": document.source_path,
            "page_count": document.page_count,
            "source_sha256": document.source_sha256,
        },
        "chunking_config": {
            **config.model_dump(),
            "config_version": ChunkingConfig.CONFIG_VERSION,
        },
        "chunk_count": len(chunks),
        "chunks": [c.to_dict() for c in chunks],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
    return path


def load_chunk_artifact(
    path: str,
) -> tuple:
    """Returns (document_meta: dict, config: ChunkingConfig, chunks: List[Chunk])."""
    with open(path, encoding="utf-8") as f:
        artifact = json.load(f)
    if artifact.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported chunk artifact format: {artifact.get('format_version')}"
        )
    config = ChunkingConfig(**artifact["chunking_config"])
    chunks = [Chunk.from_dict(c) for c in artifact["chunks"]]
    return artifact["document"], config, chunks
