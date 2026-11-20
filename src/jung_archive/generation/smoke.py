"""Live NVIDIA NIM generation smoke test (real endpoint, no mocks).

Checks, in order and reporting each stage separately:

    1. credentials present       — GENERATION_API_KEY resolved, never printed
    2. catalogue                 — GET {base_url}/models, is the model listed?
    3. authentication            — did the endpoint accept the credentials?
    4. model accepted            — POST chat/completions for the model
    5. assistant content         — non-empty assistant text returned
    6. model echo                — did the response name the requested model?

Stages 1 and 4-6 are required for success. A catalogue miss alone is not fatal
(catalogues are paginated/stale), but HTTP success with no matching model echo,
or with empty content, is reported as an error.

Configuration is resolved through
``jung_archive.generation.provider.resolve_generation_config`` so the smoke
test can never run against a different model than production generation.

A successful HTTP status is not proof of genuine NVIDIA connectivity:
outbound HTTPS may be intercepted by a proxy that fabricates responses. The
report says so explicitly instead of claiming verified connectivity.

Run from the repository root:
    python -m jung_archive.generation.smoke

Exit status: 0 only on success, 1 for any error (including missing credentials).
"""

from __future__ import annotations

import json
import sys
import time
from typing import Any, Dict, Optional

import requests

from jung_archive.generation.provider import (
    GenerationConfig,
    resolve_generation_config,
)

CONNECTIVITY_NOTE = (
    "HTTP success with a matching model echo and non-empty assistant content is "
    "the strongest signal available from this machine, but it does not prove "
    "genuine NVIDIA connectivity: outbound HTTPS can be intercepted by a proxy "
    "that fabricates responses. Confirm from an environment with direct network "
    "access and a real API key."
)


def _probe_catalogue(config: GenerationConfig, headers: Dict[str, str]) -> Dict[str, Any]:
    """Report whether the endpoint's model catalogue lists the configured model."""
    try:
        resp = requests.get(
            f"{config.base_url}/models", headers=headers, timeout=config.timeout
        )
    except (requests.ConnectionError, requests.Timeout) as exc:
        return {"catalogue_checked": False, "catalogue_error": str(exc)}

    result: Dict[str, Any] = {
        "catalogue_checked": True,
        "catalogue_status_code": resp.status_code,
    }
    if resp.status_code != 200:
        return result
    try:
        ids = [m["id"] for m in resp.json().get("data", []) if isinstance(m, dict)]
    except (ValueError, TypeError, AttributeError):
        result["catalogue_error"] = "catalogue body was not a model list"
        return result
    result["catalogue_size"] = len(ids)
    result["model_listed"] = config.model in ids
    return result


def run_live_generation_smoke_test(
    config: Optional[GenerationConfig] = None,
) -> Dict[str, Any]:
    """Send a minimal real chat-completion request and validate the response.

    Returns a dict with status "ok" or "error" plus a ``stages`` map. Never
    logs, prints, or returns the raw GENERATION_API_KEY.
    """
    # Same resolution path as OpenAICompatibleProvider: .env then process env.
    config = config or resolve_generation_config()
    model = config.model
    base_url = config.base_url

    def fail(message: str, stages: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "status": "error",
            "error": message,
            "requested_model": model,
            "base_url": base_url,
            "config_source": config.config_source,
            "stages": stages,
        }
        report.update(extra)
        return report

    stages: Dict[str, Any] = {"credentials_present": bool(config.api_key)}

    if not config.api_key:
        return fail(
            "GENERATION_API_KEY is not set; cannot authenticate to "
            f"{base_url}. Set it in the process environment or in .env "
            "before running the smoke test.",
            stages,
        )

    if not model:
        return fail("GENERATION_MODEL is not set; refusing to guess a model", stages)

    headers: Dict[str, str] = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    catalogue = _probe_catalogue(config, headers)
    stages.update(catalogue)

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Reply with exactly one short sentence."},
            {"role": "user", "content": "Reply with one short sentence."},
        ],
        "temperature": 0.2,
        "top_p": 0.7,
        "max_tokens": 30,
    }

    started = time.perf_counter()
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=config.timeout,
        )
    except requests.ConnectionError as exc:
        return fail(f"cannot reach generation endpoint at {base_url}: {exc}", stages)
    except requests.Timeout:
        return fail(f"generation request timed out after {config.timeout}s", stages)

    elapsed = time.perf_counter() - started
    stages["http_status_code"] = resp.status_code
    stages["auth_succeeded"] = resp.status_code not in (401, 403)
    shared = {
        "stages": stages,
        "latency_seconds": round(elapsed, 3),
    }

    if resp.status_code == 401:
        return fail(
            f"authentication failed (401) for model '{model}' at {base_url}: "
            "GENERATION_API_KEY was rejected",
            **shared,
        )
    if resp.status_code == 403:
        return fail(
            f"access forbidden (403) for model '{model}' at {base_url}: the "
            "key is valid but is not entitled to this model",
            **shared,
        )
    if resp.status_code == 404:
        return fail(
            f"model '{model}' or the completions route was not found (404) at "
            f"{base_url}; check GENERATION_MODEL",
            **shared,
        )
    if resp.status_code == 410:
        return fail(
            f"model '{model}' is no longer served (410 Gone) at {base_url}; "
            "choose a currently hosted NVIDIA NIM model",
            **shared,
        )
    if resp.status_code == 429:
        return fail(
            f"rate limited (429) by {base_url} for model '{model}'; retry later",
            **shared,
        )
    if resp.status_code >= 500:
        return fail(
            f"server error ({resp.status_code}) from {base_url} for model "
            f"'{model}'; the provider is unhealthy",
            **shared,
        )

    stages["model_accepted"] = resp.status_code == 200

    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        return fail(f"generation HTTP error for model '{model}': {exc}", **shared)

    try:
        body = resp.json()
    except ValueError:
        return fail(
            f"endpoint at {base_url} returned a non-JSON body for model '{model}'",
            **shared,
        )

    try:
        text = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return fail(
            "malformed generation response: missing choices[0].message.content",
            **shared,
        )

    if not isinstance(text, str) or not text.strip():
        stages["assistant_content"] = False
        return fail("generation response had empty assistant content", **shared)

    stages["assistant_content"] = True

    reported_model = body.get("model")
    if isinstance(reported_model, str) and reported_model.strip():
        model_confirmed = reported_model.strip().lower() == model.strip().lower()
    else:
        model_confirmed = False
    stages["model_echo_matches"] = model_confirmed

    if not model_confirmed:
        return fail(
            f"requested model '{model}' but the endpoint reported "
            f"'{reported_model}'. The response did not come from the intended "
            "model (a proxy or an outdated configuration is likely).",
            reported_model=reported_model,
            assistant_content_length=len(text.strip()),
            **shared,
        )

    notes = []
    if catalogue.get("model_listed") is False:
        notes.append(
            "model was not present in the endpoint's catalogue but answered a "
            "chat completion request; treat the catalogue as advisory"
        )

    return {
        "status": "ok",
        "requested_model": model,
        "reported_model": reported_model,
        "provider": "openai_compatible",
        "base_url": base_url,
        "config_source": config.config_source,
        "assistant_text": text.strip(),
        "assistant_content_length": len(text.strip()),
        "latency_seconds": round(elapsed, 3),
        "usage": body.get("usage"),
        "stages": stages,
        "notes": notes,
        "connectivity_note": CONNECTIVITY_NOTE,
    }


def main() -> int:
    """CLI entry point: print a JSON report and exit non-zero on failure."""
    # Only the presence of the key is reported, never its value.
    report = run_live_generation_smoke_test()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report.get("status") == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
