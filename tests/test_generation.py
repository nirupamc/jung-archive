"""Targeted backend tests for the ASK / generation layer.

These tests avoid importing torch / sentence-transformers by mocking
the heavy retrieval, reranking, and evidence components.
"""
from __future__ import annotations

import contextlib
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
    DEFAULT_GENERATION_MODEL,
    GenerationConfig,
    GenerationError,
    OpenAICompatibleProvider,
    resolve_generation_config,
)
from jung_archive.generation.smoke import run_live_generation_smoke_test


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
    assert p.model == "nvidia/nemotron-3.5-bolt-e4b"
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
    monkeypatch.setenv("GENERATION_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    monkeypatch.setenv("GENERATION_API_KEY", "dummy-nim-key")

    fake_response = MagicMock()
    fake_response.status_code = 200
    fake_response.json.return_value = {
        "model": "nvidia/nemotron-3.5-bolt-e4b",
        "choices": [{"message": {"content": "Answer [S1]."}}],
    }

    with patch("jung_archive.generation.provider.requests.post", return_value=fake_response) as mock_post:
        p = OpenAICompatibleProvider()
        assert p.is_local is False
        result = p.generate("evidence prompt")
        args, kwargs = mock_post.call_args
        assert args[0] == "https://integrate.api.nvidia.com/v1/chat/completions"
        assert kwargs["headers"]["Authorization"] == "Bearer dummy-nim-key"
        assert kwargs["json"]["model"] == "nvidia/nemotron-3.5-bolt-e4b"
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

    # 2. Absent env falls back to the configured NIM default.
    monkeypatch.delenv("GENERATION_MODEL", raising=False)
    p_default = OpenAICompatibleProvider()
    assert p_default.model == "nvidia/nemotron-3.5-bolt-e4b"
    assert p_default.is_local is False
    assert DEFAULT_GENERATION_MODEL == "nvidia/nemotron-3.5-bolt-e4b"


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


def test_provider_generate_410_gone(monkeypatch):
    monkeypatch.setenv("GENERATION_MODEL", "test-model")
    fake_response = MagicMock()
    fake_response.status_code = 410
    fake_response.raise_for_status.side_effect = Exception("410")
    with patch("jung_archive.generation.provider.requests.post", return_value=fake_response):
        p = OpenAICompatibleProvider()
        with pytest.raises(GenerationError, match="410"):
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


# ----------------------------------------------------------------------
# Non-streaming request contract
# ----------------------------------------------------------------------

def test_provider_request_is_non_streaming_and_complete(monkeypatch):
    """The API returns one complete answer, so no `stream` flag is sent."""
    monkeypatch.setenv("GENERATION_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    monkeypatch.setenv("GENERATION_API_KEY", "dummy")
    monkeypatch.setenv("GENERATION_BASE_URL", "https://integrate.api.nvidia.com/v1")

    fake = MagicMock()
    fake.status_code = 200
    fake.json.return_value = {
        "model": "nvidia/nemotron-3.5-bolt-e4b",
        "choices": [{"message": {"content": "One complete answer [S1]."}}],
    }

    with patch("jung_archive.generation.provider.requests.post", return_value=fake) as post:
        result = OpenAICompatibleProvider().generate("evidence prompt")

    payload = post.call_args[1]["json"]
    assert "stream" not in payload
    assert set(payload) == {"model", "messages", "temperature", "top_p", "max_tokens"}
    assert isinstance(result.text, str) and result.text.strip()


# ----------------------------------------------------------------------
# Generation smoke test — staged diagnostics against the real endpoint
# ----------------------------------------------------------------------

def _config(**overrides):
    """Build a smoke-test config without touching .env or the real env."""
    base = dict(
        provider="openai_compatible",
        base_url="https://integrate.api.nvidia.com/v1",
        model=DEFAULT_GENERATION_MODEL,
        api_key="dummy-key",
        timeout=60,
        config_source="test",
    )
    base.update(overrides)
    return GenerationConfig(**base)


class FakeResp:
    def __init__(self, status_code=200, body=None, json_raises=False):
        self.status_code = status_code
        self._body = body if body is not None else {}
        self._json_raises = json_raises

    def json(self):
        if self._json_raises:
            raise ValueError("not json")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)


def _catalogue(*model_ids):
    return FakeResp(200, {"data": [{"id": m} for m in model_ids]})


def _ok_body(model, content="Jung distinguishes the Self from the ego."):
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}}],
        "usage": {"total_tokens": 12},
    }


class _HTTPMocks:
    """Context manager exposing the patched smoke HTTP call mocks."""

    def __init__(self, get_mock, post_mock):
        self.get = get_mock
        self.post = post_mock
        self._stack = contextlib.ExitStack()

    def __enter__(self):
        self._stack.__enter__()
        return self

    def __exit__(self, *exc):
        return self._stack.__exit__(*exc)


def _patch_http(get=None, post=None):
    """Patch both smoke HTTP calls so tests never touch the network."""
    stack = contextlib.ExitStack()
    stack.__enter__()
    get_mock = stack.enter_context(
        patch(
            "jung_archive.generation.smoke.requests.get",
            return_value=(
                get if get is not None else _catalogue(DEFAULT_GENERATION_MODEL)
            ),
        )
    )
    post_mock = stack.enter_context(
        patch(
            "jung_archive.generation.smoke.requests.post",
            return_value=post if post is not None else FakeResp(200, {}),
        )
    )
    mocks = _HTTPMocks(get_mock, post_mock)
    mocks._stack = stack
    return mocks


def test_smoke_errors_when_credentials_missing():
    """Missing credentials fail immediately, without any HTTP call."""
    with _patch_http() as mocks:
        report = run_live_generation_smoke_test(_config(api_key=""))

    assert report["status"] == "error"
    assert "GENERATION_API_KEY" in report["error"]
    assert report["stages"]["credentials_present"] is False
    mocks.get.assert_not_called()
    mocks.post.assert_not_called()


def test_smoke_uses_provider_configuration_resolution(monkeypatch):
    """Smoke config and provider config come from the same resolver."""
    monkeypatch.setenv("GENERATION_MODEL", "nvidia/nemotron-3.5-bolt-e4b")
    monkeypatch.setenv("GENERATION_BASE_URL", "https://integrate.api.nvidia.com/v1/")
    monkeypatch.setenv("GENERATION_API_KEY", "env-configured-key")
    monkeypatch.setenv("GENERATION_TIMEOUT", "42")

    config = resolve_generation_config()
    provider = OpenAICompatibleProvider()

    assert config.model == provider.model == "nvidia/nemotron-3.5-bolt-e4b"
    assert config.base_url == provider.base_url == "https://integrate.api.nvidia.com/v1"
    assert config.api_key == provider.api_key == "env-configured-key"
    assert config.timeout == provider.timeout == 42


def test_smoke_requests_the_candidate_model_and_reports_every_stage():
    sent = {}

    def fake_post(url, **kwargs):
        sent["url"] = url
        sent["model"] = kwargs["json"]["model"]
        sent["auth"] = kwargs["headers"].get("Authorization")
        sent["payload_keys"] = set(kwargs["json"])
        return FakeResp(200, _ok_body("nvidia/nemotron-3.5-bolt-e4b"))

    with _patch_http(
        get=_catalogue("nvidia/nemotron-3.5-bolt-e4b"),
        post=FakeResp(200, _ok_body("nvidia/nemotron-3.5-bolt-e4b")),
    ):
        report = run_live_generation_smoke_test(_config())

    assert report["status"] == "ok"
    assert report["requested_model"] == "nvidia/nemotron-3.5-bolt-e4b"
    assert report["stages"] == {
        "credentials_present": True,
        "catalogue_checked": True,
        "catalogue_status_code": 200,
        "catalogue_size": 1,
        "model_listed": True,
        "http_status_code": 200,
        "auth_succeeded": True,
        "model_accepted": True,
        "assistant_content": True,
        "model_echo_matches": True,
    }
    assert report["assistant_text"]
    assert DEFAULT_GENERATION_MODEL == "nvidia/nemotron-3.5-bolt-e4b"


def test_smoke_catalogue_miss_is_advisory_not_fatal():
    """A catalogue miss must be reported, not turned into success or failure."""
    with _patch_http(
        get=_catalogue("some/other-model"),
        post=FakeResp(200, _ok_body(DEFAULT_GENERATION_MODEL)),
    ):
        report = run_live_generation_smoke_test(_config())

    assert report["status"] == "ok"
    assert report["stages"]["model_listed"] is False
    assert any("catalogue" in note for note in report["notes"])


def test_smoke_reports_catalogue_auth_failure_separately():
    with _patch_http(
        get=FakeResp(401, {"error": "unauthorized"}),
        post=FakeResp(200, _ok_body(DEFAULT_GENERATION_MODEL)),
    ):
        report = run_live_generation_smoke_test(_config())

    assert report["status"] == "ok"
    assert report["stages"]["catalogue_status_code"] == 401
    assert "model_listed" not in report["stages"]


@pytest.mark.parametrize(
    ("status_code", "expected"),
    [
        (401, "authentication failed"),
        (403, "forbidden"),
        (404, "not found"),
        (410, "no longer served"),
        (429, "rate limited"),
        (503, "server error"),
    ],
)
def test_smoke_reports_unsuccessful_http_responses(status_code, expected):
    with _patch_http(post=FakeResp(status_code, {"error": "nope"})):
        report = run_live_generation_smoke_test(_config())

    assert report["status"] == "error"
    assert expected in report["error"].lower()
    assert str(status_code) in report["error"]
    assert report["requested_model"] == DEFAULT_GENERATION_MODEL
    assert report["stages"]["http_status_code"] == status_code
    assert report["stages"]["auth_succeeded"] is (status_code not in (401, 403))
    assert report["stages"].get("assistant_content") is None


def test_smoke_errors_on_empty_assistant_content():
    with _patch_http(post=FakeResp(200, _ok_body(DEFAULT_GENERATION_MODEL, "   "))):
        report = run_live_generation_smoke_test(_config())

    assert report["status"] == "error"
    assert "empty assistant content" in report["error"]
    assert report["stages"]["assistant_content"] is False


def test_smoke_errors_on_non_json_body():
    with _patch_http(post=FakeResp(200, json_raises=True)):
        report = run_live_generation_smoke_test(_config())

    assert report["status"] == "error"
    assert "non-JSON" in report["error"]


def test_smoke_errors_when_served_model_differs_from_requested():
    """A response from some other model must not be reported as success."""
    with _patch_http(post=FakeResp(200, _ok_body("meta/llama-3.3-70b-instruct"))):
        report = run_live_generation_smoke_test(_config())

    assert report["status"] == "error"
    assert "meta/llama-3.3-70b-instruct" in report["error"]
    assert DEFAULT_GENERATION_MODEL in report["error"]
    assert report["stages"]["model_echo_matches"] is False


def test_smoke_success_report_never_contains_the_key_and_flags_connectivity():
    with _patch_http(post=FakeResp(200, _ok_body(DEFAULT_GENERATION_MODEL))):
        report = run_live_generation_smoke_test(_config(api_key="super-secret-key"))

    assert report["status"] == "ok"
    assert report["stages"]["model_echo_matches"] is True
    assert report["assistant_content_length"] > 0
    assert "does not prove genuine NVIDIA connectivity" in report["connectivity_note"]
    serialized = json.dumps(report)
    assert "super-secret-key" not in serialized
    assert "Authorization" not in serialized


def test_smoke_cli_exits_nonzero_on_failure(monkeypatch):
    import jung_archive.generation.smoke as smoke_mod

    monkeypatch.setattr(
        smoke_mod,
        "run_live_generation_smoke_test",
        lambda config=None: {"status": "error", "error": "boom"},
    )
    assert smoke_mod.main() == 1


def test_smoke_cli_exits_zero_on_success(monkeypatch):
    import jung_archive.generation.smoke as smoke_mod

    monkeypatch.setattr(
        smoke_mod,
        "run_live_generation_smoke_test",
        lambda config=None: {"status": "ok"},
    )
    assert smoke_mod.main() == 0


# ----------------------------------------------------------------------
# EvidencePack grounding through AskService (no retrieval changes)
# ----------------------------------------------------------------------

def _real_pack():
    from jung_archive.evidence.models import EvidenceItem, EvidencePack
    from jung_archive.models.document import SourceType

    item = EvidenceItem(
        evidence_id="S1",
        chunk_id="chunk-1",
        document_id="doc-1",
        text="Apart from agglomerations of huge masses of people.",
        clean_text="Apart from agglomerations of huge masses of people.",
        page_numbers=[18, 19],
        source_block_ids=["p0018-b002"],
        heading_path=["Carl Gustav Jung", "THE PLIGHT OF THE INDIVIDUAL"],
        source_type=SourceType.PRIMARY,
        author="Carl Gustav Jung",
        title="The Undiscovered Self",
        token_count=30,
    )
    return EvidencePack(
        question="mass psychology?",
        items=[item],
        tokens_used=30,
        max_evidence_tokens=2500,
        max_evidence_items=8,
        candidates_considered=1,
    )


def _ask_service(answer_text, pack=None):
    """AskService with mocked retrieval/reranking but a real EvidencePack."""
    from jung_archive.generation.service import AskService
    from jung_archive.reranking.base import PairConstructionReport, Reranker

    pack = pack or _real_pack()

    class StubReranker(Reranker):
        """Satisfies the reranker interface without loading a cross-encoder."""

        def score_pairs(self, query, documents):
            return [1.0 for _ in documents], PairConstructionReport()

    class FakeProvider:
        provider_name = "openai_compatible"
        model = "nvidia/nemotron-3.5-bolt-e4b"
        is_local = False

        def __init__(self):
            self.prompts = []

        def generate(self, prompt, **kwargs):
            from jung_archive.generation.provider import GenerationResult

            self.prompts.append(prompt)
            self.kwargs = kwargs
            return GenerationResult(
                text=answer_text, model=self.model, provider=self.provider_name
            )

    provider = FakeProvider()
    service = AskService(MagicMock(), MagicMock(), StubReranker(), provider)

    retrieval = MagicMock()
    retrieval.mode = "hybrid"
    retrieval.top_k = 20
    retrieval.latency_ms = 12
    retrieval.results = [MagicMock()]
    retrieval.warnings = []
    service.retriever.search = MagicMock(return_value=retrieval)
    service.pipeline.reranker.rank_results = MagicMock(return_value=([MagicMock()], {}))
    service.evidence_assembler.assemble = MagicMock(return_value=pack)
    return service, provider


def test_ask_generation_consumes_evidence_pack_and_preserves_citations():
    service, provider = _ask_service("The Self is the totality of the psyche [S1].")

    response = service.ask("mass psychology?")

    assert len(provider.prompts) == 1
    prompt = provider.prompts[0]
    assert "EVIDENCE:" in prompt
    assert "[S1]" in prompt
    assert "The Undiscovered Self" in prompt
    assert "agglomerations of huge masses" in prompt
    assert "QUESTION: mass psychology?" in prompt
    assert "Use ONLY the EVIDENCE" in prompt

    assert response.answer == "The Self is the totality of the psyche [S1]."
    assert [c.evidence_id for c in response.citations] == ["S1"]
    assert response.citations[0].status == "valid"
    assert response.evidence_pack["items"][0]["evidence_id"] == "S1"
    assert response.local_or_remote == "REMOTE"


def test_ask_flags_invented_citations_and_never_silently_accepts_them():
    service, _ = _ask_service("Jung says so [S9].")

    response = service.ask("mass psychology?")

    assert response.citations[0].evidence_id == "S9"
    assert response.citations[0].status == "unknown"
    assert any("unknown citation" in w for w in response.warnings)


def test_ask_admits_insufficient_evidence_without_calling_generation():
    service, provider = _ask_service("unused")

    from jung_archive.evidence.models import EvidencePack

    empty = EvidencePack(question="mass psychology?", items=[])
    service.evidence_assembler.assemble = MagicMock(return_value=empty)

    response = service.ask("mass psychology?")

    assert provider.prompts == []
    assert "don't have enough evidence" in response.answer
    assert response.citations == []
