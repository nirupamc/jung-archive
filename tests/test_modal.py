"""Modal deployment preparation tests.

These are cheap, side-effect-free checks that the deployment wiring is correct
and that the data-root configuration behaves as required (local default vs
JUNG_ARCHIVE_DATA_DIR override) without importing heavy models or building
the Modal image.
"""
import importlib
import os

import pytest


def test_config_default_data_dir(monkeypatch):
    monkeypatch.delenv("JUNG_ARCHIVE_DATA_DIR", raising=False)
    import jung_archive.config as cfg

    importlib.reload(cfg)
    assert cfg.get_data_dir() == cfg.REPO_ROOT / "data"


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("JUNG_ARCHIVE_DATA_DIR", "/data")
    import jung_archive.config as cfg

    importlib.reload(cfg)
    assert cfg.get_data_dir() == __import__("pathlib").Path("/data")
    assert cfg.CHROMA_DIR == __import__("pathlib").Path("/data/chroma")
    assert cfg.CHUNKS_DIR == __import__("pathlib").Path("/data/chunks")
    # Restore default module state so later imports of the app see repo/data.
    monkeypatch.delenv("JUNG_ARCHIVE_DATA_DIR", raising=False)
    importlib.reload(cfg)


def test_modal_entrypoint_imports():
    """Importing the Modal entrypoint must not download models or fail."""
    import modal

    import modal_app  # noqa: F401

    assert isinstance(modal_app.app, modal.App)
    # fastapi_app is decorated with @app.function + @modal.asgi_app()
    assert isinstance(modal_app.fastapi_app, modal.Function)


def test_modal_volume_and_secret_referenced():
    import modal_app

    # Volume + secret are referenced by the function definition.
    assert modal_app.volume is not None
    assert modal_app.secrets


def test_cors_origin_parsing(monkeypatch):
    monkeypatch.setenv("CORS_ORIGINS", "https://jung-archive.vercel.app")
    from jung_archive.api.app import _cors_origins

    origins = _cors_origins()
    assert "https://jung-archive.vercel.app" in origins
    assert "http://localhost:3000" in origins
    # Wildcard is never the default.
    assert "*" not in origins


def test_health_no_model_load():
    """/api/health must be lightweight: no model/index initialization."""
    from fastapi.testclient import TestClient

    from jung_archive.api.app import app

    client = TestClient(app)
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_section_from_absolute_windows_source_path():
    """Regression: chunk artifacts store absolute Windows source_path values.

    Before the fix, ``document_summary`` derived ``section`` by naively
    splitting the path, yielding 'D:' for absolute paths. The Library sidebar
    only renders PRIMARY/SECONDARY lanes, so those documents vanished on Modal
    (where corpus discovery can't run). Section must come from source_type (or
    a path scan), never from the drive letter.
    """
    import json
    from pathlib import Path

    from jung_archive.api.app import _section_from_source, document_summary

    # Helper behavior with absolute Windows paths.
    assert (
        _section_from_source(
            "UNKNOWN",
            r"D:\protofolo projectzzz\jung-archive\primary\vol-12.pdf",
        )
        == "PRIMARY"
    )
    assert (
        _section_from_source(
            "UNKNOWN",
            r"D:\protofolo projectzzz\jung-archive\secondary\foo.pdf",
        )
        == "SECONDARY"
    )
    # Relative paths still work.
    assert _section_from_source("UNKNOWN", "PRIMARY\\bar.pdf") == "PRIMARY"
    # source_type wins when present.
    assert _section_from_source("SECONDARY", None) == "SECONDARY"

    # End-to-end: at least one real chunk artifact uses an absolute Windows
    # source_path; its summary must be bucketed into a rendered lane.
    chunks_dir = Path("data/chunks")
    if chunks_dir.exists():
        for f in sorted(chunks_dir.glob("*.json")):
            doc = json.loads(f.read_text(encoding="utf-8")).get("document", {})
            sp = doc.get("source_path", "")
            if ":\\" in sp:  # absolute Windows path
                doc_id = f.stem
                s = document_summary(doc_id)
                assert s.section in ("PRIMARY", "SECONDARY"), (
                    f"{doc_id} -> {s.section!r} from {sp!r}"
                )
                break
