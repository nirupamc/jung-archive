"""Generation layer (provider-neutral).

Re-exports the public surface so callers can do:
    from jung_archive.generation import AskService, OpenAICompatibleProvider
"""
from jung_archive.generation.provider import (
    GenerationError,
    GenerationProvider,
    GenerationResult,
    OpenAICompatibleProvider,
)
from jung_archive.generation.service import AskService
from jung_archive.generation.citations import Citation, validate_citations

__all__ = [
    "GenerationProvider",
    "GenerationResult",
    "GenerationError",
    "OpenAICompatibleProvider",
    "AskService",
    "Citation",
    "validate_citations",
]
