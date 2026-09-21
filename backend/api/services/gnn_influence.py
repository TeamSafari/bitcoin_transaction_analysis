"""GNN edge influence calculator using GraphSAGE embeddings."""

from __future__ import annotations

import numpy as np

from backend.api.services.data_loader import DataLoader


class GNNInfluenceCalculator:
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._cache: dict[str, float] = {}

    def calculate_edge_influence(self, source_id: str, target_id: str) -> float:
        cache_key = f"{source_id}_{target_id}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            src_emb = self.loader.graphsage_embeddings.loc[source_id].values
            tgt_emb = self.loader.graphsage_embeddings.loc[target_id].values
            cosine_sim = self._cosine_similarity(src_emb, tgt_emb)
        except KeyError:
            cosine_sim = 0.5

        try:
            edges = self.loader.graph_edges
            mask = (edges["source_wallet_id"].astype(str) == str(source_id)) & (
                edges["target_wallet_id"].astype(str) == str(target_id)
            )
            tx_count = int(edges[mask]["transaction_count"].values[0]) if mask.any() else 1
            freq_score = min(1.0, np.log1p(tx_count) / np.log1p(100))
        except Exception:
            freq_score = 0.1

        influence = round(max(0.01, min(1.0, 0.6 * cosine_sim + 0.4 * freq_score)), 4)
        self._cache[cache_key] = influence
        return influence

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        n1, n2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        return float(dot / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.5
