"""Local cross-encoder reranker (M4).

Backend: sentence-transformers CrossEncoder over a lightweight MS MARCO
miniLM cross-encoder — small enough for CPU, Apache-2.0 licensed,
SentenceTransformers-compatible, fully local (no remote API).

Model is loaded lazily once per instance and reused (warm reranking).
All failures raise RerankerUnavailableError; there is no silent
fallback and no fabricated scores.
"""
import threading
from typing import List, Tuple

from jung_archive.reranking.base import (
    PairConstructionReport,
    Reranker,
    RerankerUnavailableError,
    build_safe_pairs,
)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LocalCrossEncoderReranker(Reranker):
    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        device: str = "cpu",
        batch_size: int = 16,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model_max_length = 512  # corrected from the tokenizer at load
        self._model = None
        self._tokenizer = None
        self._load_lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    try:
                        from sentence_transformers import CrossEncoder
                    except ImportError as e:
                        raise RerankerUnavailableError(
                            f"sentence-transformers not installed: {e}") from e
                    try:
                        self._model = CrossEncoder(
                            self.model_name,
                            device=self.device,
                            max_length=self.model_max_length,
                        )
                    except Exception as e:
                        raise RerankerUnavailableError(
                            f"failed to initialize cross-encoder "
                            f"{self.model_name!r}: {e}") from e
                    try:
                        tok = getattr(self._model, "tokenizer", None)
                        if tok is not None:
                            self._tokenizer = tok
                            reported = getattr(
                                tok, "model_max_length", None)
                            if isinstance(reported, int) and \
                                    0 < reported < 100_000:
                                self.model_max_length = reported
                    except Exception as e:  # tokenizer introspection only
                        raise RerankerUnavailableError(
                            f"reranker tokenizer unavailable: {e}") from e
        return self._model

    def score_pairs(self, query: str, documents: List[str]) -> \
            Tuple[List[float], PairConstructionReport]:
        if not documents:
            return [], PairConstructionReport(
                model_max_length=self.model_max_length)
        model = self._ensure_model()
        pairs, report = build_safe_pairs(
            query,
            documents,
            model_max_length=self.model_max_length,
            tokenizer=self._tokenizer,
        )
        try:
            raw = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
        except Exception as e:
            raise RerankerUnavailableError(
                f"cross-encoder scoring failed: {e}") from e
        try:
            scores = [float(s) for s in raw]
        except (TypeError, ValueError) as e:
            raise RerankerUnavailableError(
                f"cross-encoder returned non-numeric scores: {e}") from e
        if len(scores) != len(pairs):
            raise RerankerUnavailableError(
                f"cross-encoder returned {len(scores)} scores for "
                f"{len(pairs)} pairs")
        return scores, report
