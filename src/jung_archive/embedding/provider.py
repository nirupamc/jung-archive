"""
Provider-neutral embedding interface with a local SentenceTransformer
implementation. No remote APIs; model weights are cached locally.
"""
import threading
from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingProvider(ABC):
    """Interface every embedding backend must implement."""

    model_name: str
    dimension: int
    normalized: bool

    @abstractmethod
    def embed(self, texts: List[str]) -> np.ndarray:
        """Embed a batch of texts -> (n_texts, dimension) float array."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]


class LocalSentenceTransformerProvider(EmbeddingProvider):
    """Local CPU-friendly provider backed by sentence-transformers.

    The model is loaded lazily once per instance and reused across calls.
    """

    _load_lock = threading.Lock()

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
        batch_size: int = 32,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalized = normalize
        self._model = None
        self.dimension = None  # known after first load/encode

    def _ensure_model(self):
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_name, device=self.device
                    )
                    self.dimension = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension or 0), dtype=np.float32)
        self._ensure_model()
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalized,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        self.dimension = vectors.shape[1]
        return vectors
