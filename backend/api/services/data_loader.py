"""Load pipeline output artifacts for API read endpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.config import OUTPUT_DIR


class DataLoader:
    """Loads and caches CSV artifacts from a job output directory."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self._cache: dict[str, pd.DataFrame] = {}

    def load_csv(self, relative_path: str) -> pd.DataFrame:
        if relative_path in self._cache:
            return self._cache[relative_path]

        full_path = self.data_root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Artifact not found: {full_path}")

        df = pd.read_csv(full_path)
        self._cache[relative_path] = df
        return df

    @property
    def graph_edges(self) -> pd.DataFrame:
        return self.load_csv("graphs/graph_edges.csv")

    @property
    def graph_features(self) -> pd.DataFrame:
        return self.load_csv("graphs/graph_features.csv").set_index("wallet_id")

    @property
    def wallet_features(self) -> pd.DataFrame:
        return self.load_csv("features/wallet_features.csv").set_index("wallet_id")

    @property
    def graphsage_embeddings(self) -> pd.DataFrame:
        return self.load_csv("graphs/graphsage_embeddings.csv").set_index("wallet_id")

    @property
    def deterministic_scores(self) -> pd.DataFrame:
        return self.load_csv("models/deterministic_scores.csv").set_index("wallet_id")

    @property
    def fusion_scores(self) -> pd.DataFrame:
        return self.load_csv("models/fusion_scores.csv").set_index("wallet_id")

    @property
    def risk_predictions(self) -> pd.DataFrame:
        return self.load_csv("models/risk_model_predictions.csv").set_index("wallet_id")

    def transactions_for_job(self, raw_dir: Path | None) -> pd.DataFrame:
        candidates = []
        if raw_dir:
            candidates.append(Path(raw_dir) / "transactions.csv")
        candidates.append(self.data_root.parent / "raw" / "transactions.csv")
        candidates.append(OUTPUT_DIR / "raw" / "transactions.csv")

        for path in candidates:
            if path.exists():
                return pd.read_csv(path)
        raise FileNotFoundError("transactions.csv not found for job.")
