"""Provider-neutral LLM generation layer.

All generation providers read configuration from environment variables
only. No keys or secrets are ever exposed to frontend code.

Environment variables (OpenAI-compatible provider):
    GENERATION_PROVIDER   = openai_compatible (default)
    GENERATION_BASE_URL   = https://integrate.api.nvidia.com/v1 (default)
    GENERATION_MODEL      = nvidia/nemotron-3.5-bolt-e4b (default)
    GENERATION_API_KEY    = required for hosted providers; sent as Bearer
    GENERATION_TIMEOUT    = 60 (seconds, default)

NVIDIA NIM is just another OpenAI-compatible endpoint: the defaults above
target it, but any OpenAI-compatible server (local llama.cpp, LM Studio,
OpenRouter, ...) works by overriding the env vars. LOCAL/REMOTE is derived
from the base URL only (127.0.0.1 / localhost / 0.0.0.0 => LOCAL).

Request shape (verified against NVIDIA NIM chat completions):
    POST {base_url}/chat/completions
    {model, messages, temperature, top_p, max_tokens}

Streaming is deliberately not used: /api/ask returns one complete response,
so the provider sends a single non-streaming request (`stream` is never set).
NVIDIA-specific extras (chat_template_kwargs / enable_thinking) are not sent:
they are model-specific reasoning switches, not portable OpenAI-compatible
options, and the configured model does not require them.
"""
from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

# Load .env at import time so GENERATION_* vars are available when
# OpenAICompatibleProvider is instantiated via `uvicorn jung_archive.api.app:app`.
# Uses python-dotenv's default (override=False) so explicit OS / PowerShell
# env vars continue to take precedence over .env.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


DEFAULT_GENERATION_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Default only applies when GENERATION_MODEL is not set; production overrides
# it via the Modal secret / .env.
DEFAULT_GENERATION_MODEL = "nvidia/nemotron-3.5-bolt-e4b"


@dataclass(frozen=True)
class GenerationConfig:
    """Resolved generation configuration, shared by provider and smoke test."""

    provider: str
    base_url: str
    model: str
    api_key: str
    timeout: int
    config_source: str

    @property
    def is_local(self) -> bool:
        base = self.base_url.lower()
        return any(local in base for local in ["127.0.0.1", "localhost", "0.0.0.0"])


def resolve_generation_config() -> GenerationConfig:
    """Resolve generation settings the same way for provider and smoke test.

    Reads ``.env`` (without overriding real process/OS variables) and then the
    ``GENERATION_*`` environment variables. This is the single source of truth
    so a smoke test cannot silently run against a different model than the
    production provider.
    """
    source = "process environment (.env is loaded at provider import)"

    raw_timeout = os.environ.get("GENERATION_TIMEOUT", "60")
    try:
        timeout = int(raw_timeout)
    except (TypeError, ValueError):
        timeout = 60

    return GenerationConfig(
        provider=os.environ.get("GENERATION_PROVIDER", "openai_compatible")
        or "openai_compatible",
        base_url=os.environ.get(
            "GENERATION_BASE_URL", DEFAULT_GENERATION_BASE_URL
        ).rstrip("/"),
        model=os.environ.get("GENERATION_MODEL", DEFAULT_GENERATION_MODEL),
        api_key=os.environ.get("GENERATION_API_KEY", ""),
        timeout=timeout,
        config_source=source,
    )


class GenerationError(RuntimeError):
    """Raised when the generation provider cannot fulfil a request."""


class GenerationResult:
    """Structured output from a single generation call."""

    def __init__(
        self,
        text: str,
        model: str,
        provider: str,
        usage: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.model = model
        self.provider = provider
        self.usage = usage or {}


class GenerationProvider(ABC):
    """Interface every generation backend must implement."""

    provider_name: str

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        """Return generated text for *prompt*."""


class OpenAICompatibleProvider(GenerationProvider):
    """OpenAI-compatible chat completion provider.

    Supports local llama.cpp OpenAI-compatible servers, LM Studio,
    OpenRouter, and any OpenAI-compatible endpoint.

    Configuration via environment variables only. Never expose API
    keys to frontend code.
    """

    provider_name = "openai_compatible"

    def __init__(self):
        config = resolve_generation_config()
        self.base_url = config.base_url
        self.model = config.model
        self.api_key = config.api_key
        self.timeout = config.timeout
        self._is_local = self._detect_local()

    def _detect_local(self) -> bool:
        base = self.base_url.lower()
        return any(local in base for local in ["127.0.0.1", "localhost", "0.0.0.0"])

    @property
    def is_local(self) -> bool:
        return self._detect_local()

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        if not self.model:
            raise GenerationError("GENERATION_MODEL is not set")

        messages = kwargs.get("messages")
        if messages is None:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a careful research assistant answering questions "
                        "about Jung's works. Use ONLY the supplied EVIDENCE below. "
                        "Do not invent facts outside the evidence. Cite claims "
                        "using the bracketed IDs from the evidence, e.g. [S1], [S3]. "
                        "Do not invent citation IDs. If the evidence is insufficient, "
                        "say so explicitly. Do not claim certainty beyond the evidence."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.2),
            "top_p": kwargs.get("top_p", 0.7),
            "max_tokens": kwargs.get("max_tokens", 1200),
        }

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        started = time.perf_counter()
        try:
            resp = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=self.timeout,
            )
        except requests.ConnectionError as exc:
            raise GenerationError(
                f"cannot reach generation endpoint at {self.base_url}: {exc}"
            ) from exc
        except requests.Timeout as exc:
            raise GenerationError(
                f"generation request timed out after {self.timeout}s"
            ) from exc

        if resp.status_code == 401:
            raise GenerationError("generation endpoint rejected credentials (401)")
        if resp.status_code == 403:
            raise GenerationError("generation endpoint forbidden (403)")
        if resp.status_code == 404:
            raise GenerationError(
                f"model '{self.model}' not found at {self.base_url} (404)"
            )
        if resp.status_code == 410:
            raise GenerationError(
                f"model '{self.model}' is no longer available at {self.base_url} (410)"
            )
        if resp.status_code == 429:
            raise GenerationError("generation endpoint rate-limited (429)")
        if resp.status_code >= 500:
            raise GenerationError(
                f"generation endpoint server error ({resp.status_code})"
            )
        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise GenerationError(f"generation HTTP error: {exc}") from exc

        try:
            body = resp.json()
        except ValueError as exc:
            raise GenerationError("generation endpoint returned non-JSON body") from exc

        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GenerationError(
                "malformed generation response: missing choices[0].message.content"
            ) from exc

        if not isinstance(text, str) or not text.strip():
            raise GenerationError(
                "malformed generation response: empty assistant content"
            )

        usage = body.get("usage", {})
        return GenerationResult(
            text=text.strip(),
            model=body.get("model", self.model),
            provider=self.provider_name,
            usage=usage,
        )
