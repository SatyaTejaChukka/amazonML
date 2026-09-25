"""Optional embedding extension point.

The baseline intentionally has no model download or network dependency. A future
provider can implement ``EmbeddingProvider`` and contribute ANN candidates without
changing the matcher output contract.
"""

from __future__ import annotations

from typing import Dict, List, Protocol, Tuple

from .normalize import NormalizedRecord


class EmbeddingProvider(Protocol):
    def score_records(self, query: NormalizedRecord, records: Dict[str, NormalizedRecord]) -> Dict[str, float]:
        ...


class NullEmbeddingProvider:
    def score_records(self, query: NormalizedRecord, records: Dict[str, NormalizedRecord]) -> Dict[str, float]:
        return {}


class SentenceTransformerProvider:
    """GPU-backed semantic reranker for already-blocked candidate pairs.

    It deliberately does not build a 10M-record embedding index. The CPU blocker
    controls recall, while this model is used only on the small candidate set.
    """

    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 16):
        import torch
        from sentence_transformers import SentenceTransformer

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    @staticmethod
    def _text(record: NormalizedRecord) -> str:
        return " | ".join(
            part for part in (
                record.business_name,
                record.business_address,
                record.country,
            ) if part
        )

    def score_records(self, query: NormalizedRecord, records: Dict[str, NormalizedRecord]) -> Dict[str, float]:
        if not records:
            return {}
        import numpy as np

        query_embedding = self.model.encode(
            [self._text(query)],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        ids = list(records)
        candidate_embeddings = self.model.encode(
            [self._text(records[entity_id]) for entity_id in ids],
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        scores = np.asarray(candidate_embeddings) @ np.asarray(query_embedding)
        return {entity_id: float(score) for entity_id, score in zip(ids, scores)}


def build_provider(name: str = "none", device: str = "auto", batch_size: int = 16) -> EmbeddingProvider:
    if name in ("", "none", None):
        return NullEmbeddingProvider()
    return SentenceTransformerProvider(name, device=device, batch_size=batch_size)
