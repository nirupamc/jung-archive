"""ASK service — retrieval → rerank → evidence → generation → citation validation.

Reuses the existing retrieval, reranking, and evidence components.
No new retrieval stack is introduced.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from jung_archive.evidence.assembler import EvidenceAssembler, EvidenceConfig
from jung_archive.evidence.models import EvidencePack
from jung_archive.generation.citations import Citation, citation_validation_warnings, validate_citations
from jung_archive.generation.provider import (
    GenerationError,
    GenerationProvider,
    GenerationResult,
    OpenAICompatibleProvider,
)
from jung_archive.retrieval.hybrid import HybridRetriever, HybridRetrieverConfig
from jung_archive.retrieval.pipeline import RerankingPipeline, RerankingPipelineConfig
from jung_archive.retrieval.results import RetrievalResponse


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


class AskService:
    """Orchestrate the full ASK pipeline."""

    def __init__(
        self,
        vector_index,
        bm25,
        reranker,
        provider: Optional[GenerationProvider] = None,
    ):
        self.retriever = HybridRetriever(
            vector_index,
            bm25,
            HybridRetrieverConfig(
                dense_candidate_k=30,
                bm25_candidate_k=30,
                rrf_k=60,
                final_top_k=20,
                mode="hybrid",
            ),
        )
        self.pipeline = RerankingPipeline(
            vector_index,
            bm25,
            reranker,
            RerankingPipelineConfig(
                dense_candidate_k=30,
                bm25_candidate_k=30,
                rrf_k=60,
                fusion_candidate_k=20,
                rerank_top_k=10,
                mode="hybrid_rerank",
            ),
        )
        self.gen_provider = provider or OpenAICompatibleProvider()
        self.evidence_assembler = EvidenceAssembler(
            EvidenceConfig(max_evidence_tokens=2500, max_evidence_items=8)
        )

    def ask(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        generation: Optional[Dict[str, Any]] = None,
    ) -> AskResponse:
        filters = filters or {}
        generation = generation or {}
        warnings: List[str] = []

        # 1. Hybrid retrieval
        retrieval: RetrievalResponse = self.retriever.search(
            query, top_k=20, filters=filters, mode="hybrid"
        )
        retrieval_metadata: Dict[str, Any] = {
            "mode": retrieval.mode,
            "top_k": retrieval.top_k,
            "latency_ms": retrieval.latency_ms,
            "results": len(retrieval.results),
            "warnings": list(retrieval.warnings),
        }
        warnings.extend(retrieval.warnings)

        no_eligible = any(
            "no eligible documents" in w.lower() for w in retrieval.warnings
        )
        if no_eligible:
            return AskResponse(
                answer="I don't have enough evidence in the indexed archive to answer this reliably.",
                citations=[],
                evidence_pack={},
                provider=self.gen_provider.provider_name,
                model=getattr(self.gen_provider, "model", ""),
                local_or_remote="LOCAL" if self.gen_provider.is_local else "REMOTE",
                retrieval_metadata=retrieval_metadata,
                warnings=warnings,
            )

        # 2. Rerank
        if not retrieval.results:
            return AskResponse(
                answer="I don't have enough evidence in the indexed archive to answer this reliably.",
                citations=[],
                evidence_pack={},
                provider=self.gen_provider.provider_name,
                model=getattr(self.gen_provider, "model", ""),
                local_or_remote="LOCAL" if self.gen_provider.is_local else "REMOTE",
                retrieval_metadata=retrieval_metadata,
                warnings=warnings + ["no retrieval results"],
            )

        try:
            ranked, _ = self.pipeline.reranker.rank_results(query, retrieval.results)
        except Exception as exc:
            return AskResponse(
                answer=f"Reranking failed: {exc}",
                citations=[],
                evidence_pack={},
                provider=self.gen_provider.provider_name,
                model=getattr(self.gen_provider, "model", ""),
                local_or_remote="LOCAL" if self.gen_provider.is_local else "REMOTE",
                retrieval_metadata=retrieval_metadata,
                warnings=warnings + [f"reranker failed: {exc}"],
            )

        # 3. Evidence assembly
        pack = self.evidence_assembler.assemble(query, ranked)

        if not pack.items:
            return AskResponse(
                answer="I don't have enough evidence in the indexed archive to answer this reliably.",
                citations=[],
                evidence_pack=pack.model_dump(mode="json"),
                provider=self.gen_provider.provider_name,
                model=getattr(self.gen_provider, "model", ""),
                local_or_remote="LOCAL" if self.gen_provider.is_local else "REMOTE",
                retrieval_metadata=retrieval_metadata,
                warnings=warnings + pack.warnings + ["no evidence items selected"],
            )

        # 4. Build prompt and generate
        from jung_archive.generation.prompt import build_ask_prompt

        prompt = build_ask_prompt(query, pack)
        try:
            result: GenerationResult = self.gen_provider.generate(prompt, **generation)
        except GenerationError as exc:
            return AskResponse(
                answer=f"Generation failed: {exc}",
                citations=[],
                evidence_pack=pack.model_dump(mode="json"),
                provider=self.gen_provider.provider_name,
                model=getattr(self.gen_provider, "model", ""),
                local_or_remote="LOCAL" if self.gen_provider.is_local else "REMOTE",
                retrieval_metadata=retrieval_metadata,
                warnings=warnings + pack.warnings + [f"generation failed: {exc}"],
            )

        # 5. Validate citations
        citations = validate_citations(result.text, pack)
        warnings.extend(pack.warnings)
        warnings.extend(citation_validation_warnings(result.text, pack))

        if not self.gen_provider.is_local:
            warnings.append(
                "generation provider is REMOTE; corpus evidence is being sent off-machine"
            )

        return AskResponse(
            answer=result.text,
            citations=[
                CitationOut(
                    id=c.id,
                    evidence_id=c.evidence_id,
                    status=c.status,
                    note=c.note,
                )
                for c in citations
            ],
            evidence_pack=pack.model_dump(mode="json"),
            provider=result.provider,
            model=result.model,
            local_or_remote="LOCAL" if self.gen_provider.is_local else "REMOTE",
            retrieval_metadata=retrieval_metadata,
            warnings=warnings,
        )
