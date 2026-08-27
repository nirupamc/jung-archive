"""Targeted backend tests for the ASK / generation layer.

These tests avoid importing torch / sentence-transformers by mocking
the heavy retrieval, reranking, and evidence components.
"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests
from fastapi.testclient import TestClient

from jung_archive.api.app import app
from jung_archive.generation.citations import (
    Citation,
    citation_validation_warnings,
    validate_citations,
)
from jung_archive.generation.prompt import build_ask_prompt
from jung_archive.generation.provider import (
    GenerationError,
    OpenAICompatibleProvider,
)


# ----------------------------------------------------------------------
# Prompt assembly
# ----------------------------------------------------------------------

class DummyItem:
    evidence_id = "S1"
    text = "Apart from agglomerations of huge masses of people."
    clean_text = "Apart from agglomerations of huge masses of people."
    page_numbers = [18, 19]
    source_block_ids = ["p0018-b002"]
    heading_path = ["Carl Gustav Jung", "THE PLIGHT OF THE INDIVIDUAL"]
    source_type = "PRIMARY"
    author = "Carl Gustav Jung"
    title = "The Undiscovered Self"
    section_id = None
    token_count = 30
    scores = {}

    def pages_display(self) -> str:
        lo, hi = min(self.page_numbers), max(self.page_numbers)
        return str(lo) if lo == hi else f"{lo}-{hi}"


class DummyPack:
    question = "mass psychology?"
    items = [DummyItem()]
    tokens_used = 30
    max_evidence_tokens = 2500
    max_evidence_items = 8
    candidates_considered = 8
    suppressed_duplicates = []
    suppressed_diversity = []
    skipped_oversized = []
    warnings = []


def test_build_ask_prompt_contains_evidence_and_instruction():
    prompt = build_ask_prompt("mass psychology?", DummyPack())
    assert "mass psychology?" in prompt
    assert "[S1]" in prompt
    assert "The Undiscovered Self" in prompt
    assert "apart from agglomerations" in prompt.lower()
    assert "Do not invent facts" in prompt or "only from the evidence" in prompt.lower()


def test_build_ask_prompt_does_not_leak_secrets():
    prompt = build_ask_prompt("q", DummyPack())
    assert "api_key" not in prompt.lower()
    assert "GENERATION_API_KEY" not in prompt


# ----------------------------------------------------------------------
# Citation validation
# ----------------------------------------------------------------------

def test_validate_citations_valid():
    pack = DummyPack()
    citations = validate_citations("Jung describes the Self [S1].", pack)
    assert len(citations) == 1
    assert citations[0].evidence_id == "S1"
    assert citations[0].status == "valid"
    assert citations[0].id == "[S1]"


def test_validate_citations_unknown():
    pack = DummyPack()
    citations = validate_citations("Something about [S99].", pack)
    assert len(citations) == 1
    assert citations[0].evidence_id == "S99"
    assert citations[0].status == "unknown"
    assert "S99 not in evidence pack" in (citations[0].note or "")


def test_validate_citations_no_citations_when_none_present():
    pack = DummyPack()
    citations = validate_citations("No citations here.", pack)
    assert citations == []


def test_citation_validation_warnings_unknown():
    pack = DummyPack()
    warnings = citation_validation_warnings("Reference [S99] and [S2].", pack)
    assert any("unknown citation" in w for w in warnings)
    ids = [w for w in warnings if "unknown citation" in w][0]
    assert "S99" in ids
    assert "S2" in ids


def test_citation_validation_warnings_missing_citations():
    pack = DummyPack()
    warnings = citation_validation_warnings("An answer with no citations.", pack)
    assert any("no citations despite available evidence" in w for w in warnings)


# ----------------------------------------------------------------------
# OpenAICompatibleProvider
# ----------------------------------------------------------------------

def test_provider_defaults_from_env(monkeypatch):
    monkeypatch.delenv("GENERATION_PROVIDER", raising=False)
    monkeypatch.delenv("GENERATION_BASE_URL", raising=False)
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    monkeypatch.delenv("GENERATION_API_KEY", raising=False)
    monkeypatch.delenv("GENERATION_TIMEOUT", raising=False)

    p = OpenAICompatibleProvider()
    assert p.base_url == "https://integrate.api.nvidia.com/v1"
    assert p.model == "meta/llama-3.2-11b-vision-instruct"
    assert p.api_key == ""
    assert p.timeout == 60


def test_provider_local_detection(monkeypatch):
    monkeypatch.setenv("GENERATION_BASE_URL", "http://127.0.0.1:8080/v1")
    p = OpenAICompatibleProvider()
    assert p.is_local is True

    monkeypatch.setenv("GENERATION_BASE_URL", "http://localhost:8080/v1")
    p = OpenAICompatibleProvider()
    assert p.is_local is True

    monkeypatch.setenv("GENERATION_BASE_URL", "http://remote.example.com/v1")
    p = OpenAICompatibleProvider()
    assert p.is_local is False


def test_provider_generate_success(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "test-model")
    monkeypatch.setenv("GENERATION_BASE_URL", "http://127.0.0.1:8080/v1")
    monkeypatch.delenv("GENERATION_API_KEY", raising=False)

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "model": "test-model",
        "choices": [{"message": {"content": "Answer [S1]."}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    with patch("jung_archive.generation.provider.requests.post", return_value=fake_response) as mock_post:
        p = OpenAICompatibleProvider()
        result = p.generate("prompt")
        assert result.text == "Answer [S1]."
        assert result.model == "test-model"
        assert result.provider == "openai_compatible"
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["json"]["model"] == "test-model"
        assert "Authorization" not in call_kwargs["headers"]


def test_provider_nim_request_shape(monkeypatch):
    """NVIDIA NIM is just another OpenAI-compatible endpoint.

    Confirms the exact request shape the provider emits for a NIM config:
    POST <base_url>/chat/completions, Bearer auth, the NIM model id, and a
    system+user message pair. No NIM-specific coupling in the code.
    """
    monkeypatch.setenv("GENERATION_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("GENERATION_MODEL", "meta/llama-3.2-11b-vision-instruct")
    monkeypatch.setenv("GENERATION_API_KEY", "dummy-nim-key")

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "model": "meta/llama-3.2-11b-vision-instruct",
        "choices": [{"message": {"content": "Answer [S1]."}}],
    }

    with patch("jung_archive.generation.provider.requests.post", return_value=fake_response) as mock_post:
        p = OpenAICompatibleProvider()
        assert p.is_local is False
        result = p.generate("evidence prompt")
        args, kwargs = mock_post.call_args
        assert args[0] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer dummy-nim-key"
        assert kwargs["json"]["model"] == "meta/llama-3.2-11b-vision-instruct"
        assert kwargs["json"]["temperature"] == 0.2
        assert kwargs["json"]["top_p"] == 0.7
        assert kwargs["json"]["max_tokens"] == 1200
        msgs = kwargs["json"]["messages"]
        assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
        assert result.text == "Answer [S1]."
        assert result.provider == "openai_compatible"


def test_provider_nim_model_default_regression(monkeypatch):
    """Regression: the safe NVIDIA NIM default must be the current model.

    Covers: env override wins, absent env falls back to the expected default,
    and the NIM base URL is classified REMOTE (never LOCAL).
    """
    # 1. Explicit env value always wins over the default.
    monkeypatch.setenv("GENERATION_BASE_URL", "https://integrate.api.nvidia.com/v1")
    monkeypatch.setenv("GENERATION_MODEL", "some-user-model")
    p_env = OpenAICompatibleProvider()
    assert p_env.model == "some-user-model"
    assert p_env.is_local is False

    # 2. Absent env falls back to the current NIM default.
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    p_default = OpenAICompatibleProvider()
    assert p_default.model == "meta/llama-3.2-11b-vision-instruct"
    assert p_default.is_local is False


def test_provider_generate_with_api_key(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "test-model")
    monkeypatch.setenv("GENERATION_API_KEY", "secret-key")
    monkeypatch.setenv("GENERATION_BASE_URL", "http://127.0.0.1:8080/v1")

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "model": "test-model",
        "choices": [{"message": {"content": "Answer."}}],
    }

    with patch("jung_archive.generation.provider.requests.post", return_value=fake_response) as mock_post:
        p = OpenAICompatibleProvider()
        p.generate("prompt")
        headers = mock_post.call_args[1]["headers"]
        assert headers["Authorization"] == "Bearer secret-key"


def test_provider_generate_connection_error(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "test-model")
    with patch("jung_archive.generation.provider.requests.post", side_effect=requests.ConnectionError("boom")):
        p = OpenAICompatibleProvider()
        with pytest.raises(GenerationError, match="cannot reach generation endpoint"):
            p.generate("prompt")


def test_provider_generate_timeout(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "test-model")
    with patch("jung_archive.generation.provider.requests.post", side_effect=requests.Timeout("timed out")):
        p = OpenAICompatibleProvider()
        with pytest.raises(GenerationError, match="timed out"):
            p.generate("prompt")


def test_provider_generate_401(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "test-model")
    fake_response = MagicMock()
    fake_response.status_code = 401
    fake_response.raise_for_status.side_effect = Exception("401")
    with patch("jung_archive.generation.provider.requests.post", return_value=fake_response):
        p = OpenAICompatibleProvider()
        with pytest.raises(GenerationError, match="credentials"):
            p.generate("prompt")


def test_provider_generate_missing_model(monkeypatch):
    # Explicitly empty model must still raise; the NIM model is only a
    # default and can be overridden (including to an empty value).
    monkeypatch.setenv("GENERATION_MODEL", "")
    p = OpenAICompatibleProvider()
    with pytest.raises(GenerationError, match="GENERATION_MODEL is not set"):
        p.generate("prompt")


# ----------------------------------------------------------------------
# /api/ask endpoint — unit tests with mocked service
# ----------------------------------------------------------------------

def test_ask_rejects_empty_query():
    client = TestClient(app)
    r = client.post("/api/ask", json={"query": "   ", "filters": {}, "generation": {}})
    assert r.status_code == 400
    body = r.json()
    assert "query must be non-empty" in body["detail"]


def test_ask_success_mocked():
    """Ask pipeline happy path with mocked retrieval / provider."""
    from jung_archive.api.app import run_ask
    from jung_archive.generation.service import AskResponse, CitationOut

    mock_response = AskResponse(
        answer="Jung describes the Self as totality [S1].",
        citations=[CitationOut(id="[S1]", evidence_id="S1", status="valid")],
        evidence_pack={"items": [], "tokens_used": 100, "max_evidence_tokens": 2500, "max_evidence_items": 8, "candidates_considered": 8, "suppressed_duplicates": [], "suppressed_diversity": [], "skipped_oversized": [], "warnings": []},
        provider="openai_compatible",
        model="test-model",
        local_or_remote="LOCAL",
        retrieval_metadata={"mode": "hybrid", "top_k": 20, "latency_ms": 50, "results": 5, "warnings": []},
        warnings=[],
    )

    mock_service = MagicMock()
    mock_service.ask.return_value = mock_response

    mock_vi = MagicMock()
    mock_bm25 = MagicMock()
    mock_reranker = MagicMock()

    with patch("jung_archive.api.app.get_services", return_value=(mock_vi, mock_bm25, mock_reranker)), \
         patch("jung_archive.generation.AskService", return_value=mock_service), \
         patch("jung_archive.generation.OpenAICompatibleProvider", return_value=MagicMock()):
        from jung_archive.api.ask_schemas import AskRequest
        req = AskRequest(query="test", filters={}, generation={})
        result = run_ask(req)

    assert result.answer == "Jung describes the Self as totality [S1]."
    assert result.local_or_remote == "LOCAL"
    assert result.provider == "openai_compatible"


def test_ask_endpoint_502_on_failure():
    """If run_ask raises, endpoint returns 502."""
    client = TestClient(app)
    with patch("jung_archive.api.app.run_ask", side_effect=RuntimeError("provider down")):
        r = client.post("/api/ask", json={"query": "test", "filters": {}, "generation": {}})
    assert r.status_code == 502
    body = r.json()
    assert "ask failed" in body["detail"]


# ----------------------------------------------------------------------
# .env bootstrap regression (NVIDIA 401 fix)
# ----------------------------------------------------------------------

def test_provider_dotenv_bootstrap_present():
    """Regression: generation provider must auto-load .env via python-dotenv.

    Ensures the 401 fix is not regressed — the backend must call
    load_dotenv() at import time so `uvicorn jung_archive.api.app:app`
    sees GENERATION_API_KEY without --env-file.
    """
    import pathlib

    import jung_archive.generation.provider as mod

    source = pathlib.Path(mod.__file__).read_text(encoding="utf-8")
    assert "load_dotenv" in source, "provider.py must call load_dotenv()"
    assert "from dotenv import load_dotenv" in source
    # Must not use override=True — explicit OS env must win over .env.
    assert "override=True" not in source


def test_dotenv_values_and_explicit_override(tmp_path, monkeypatch):
    """Temporary .env is loaded and explicit process env overrides it.

    Does not use the real repo .env and does not call NVIDIA.
    """
    from dotenv import load_dotenv

    env_file = tmp_path / ".env.test"
    env_file.write_text(
        "GENERATION_MODEL=env-model-from-dotenv\n"
        "GENERATION_BASE_URL=https://env.example.com/v1\n"
        "GENERATION_API_KEY=env-key-from-dotenv\n"
    )

    # Start clean — provider should fall back to defaults if no env at all,
    # but after load_dotenv it must see the file's values.
    for key in ("GENERATION_MODEL", "GENERATION_BASE_URL", "GENERATION_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    load_dotenv(dotenv_path=str(env_file), override=False)

    p = OpenAICompatibleProvider()
    assert p.model == "env-model-from-dotenv"
    assert p.base_url == "https://env.example.com/v1"
    assert p.api_key == "env-key-from-dotenv"
    assert p.is_local is False

    # Explicit process env must win over .env (override=False).
    monkeypatch.setenv("GENERATION_MODEL", "explicit-model")
    monkeypatch.setenv("GENERATION_API_KEY", "explicit-key")
    # Re-load same file — with override=False the explicit values must survive.
    load_dotenv(dotenv_path=str(env_file), override=False)

    p2 = OpenAICompatibleProvider()
    assert p2.model == "explicit-model"
    assert p2.api_key == "explicit-key"
    # base_url was not overridden explicitly, so .env value remains
    assert p2.base_url == "https://env.example.com/v1"
