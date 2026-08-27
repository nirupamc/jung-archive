"""API contracts for the ASK endpoint."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class AskRequest(BaseModel):
    query: str
    filters: Dict[str, Any] = {}
    generation: Dict[str, Any] = {}


class CitationOut(BaseModel):
    id: str
    evidence_id: str
    status: str
    note: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    citations: List[CitationOut]
    evidence_pack: Dict[str, Any]
    provider: str
    model: str
    local_or_remote: str
    retrieval_metadata: Dict[str, Any]
    warnings: List[str]
