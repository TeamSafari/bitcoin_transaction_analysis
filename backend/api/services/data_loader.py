"""Load pipeline output artifacts for API read endpoints."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backend.config import OUTPUT_DIR
from backend.database import get_result


class DataLoader:
    """Loads and caches CSV artifacts from a job output directory."""

    def __init__(self, data_root: Path, job_id: str | None = None, db_path: Path | None = None):
        self.data_root = Path(data_root)
        self.job_id = job_id
        self.db_path = db_path
        self._cache: dict[str, pd.DataFrame] = {}

    def load_result(self, artifact_name: str) -> pd.DataFrame:
        if not self.job_id:
            raise FileNotFoundError(f"Database artifact unavailable without a job ID: {artifact_name}")
        records = get_result(self.job_id, artifact_name, db_path=self.db_path)
        if records is None:
            raise FileNotFoundError(f"Database artifact not found: {artifact_name}")
        return pd.DataFrame(records)

    def load_csv(self, relative_path: str) -> pd.DataFrame:
        if relative_path in self._cache:
            return self._cache[relative_path]

        full_path = self.data_root / relative_path
        if not full_path.exists():
            raise FileNotFoundError(f"Artifact not found: {full_path}")

        df = pd.read_csv(full_path)
        self._cache[relative_path] = df
        return df

    def _load_artifact(self, artifact_name: str, relative_path: str) -> pd.DataFrame:
        if self.job_id:
            if artifact_name in self._cache:
                return self._cache[artifact_name]
            df = self.load_result(artifact_name)
            self._cache[artifact_name] = df
            return df
        return self.load_csv(relative_path)

    @property
    def graph_edges(self) -> pd.DataFrame:
        return self._load_artifact("graph_edges", "graphs/graph_edges.csv")

    @property
    def graph_features(self) -> pd.DataFrame:
        return self._load_artifact("graph_features", "graphs/graph_features.csv").set_index("wallet_id")

    @property
    def wallet_features(self) -> pd.DataFrame:
        return self._load_artifact("wallet_features", "features/wallet_features.csv").set_index("wallet_id")

    @property
    def graphsage_embeddings(self) -> pd.DataFrame:
        return self._load_artifact("graphsage_embeddings", "graphs/graphsage_embeddings.csv").set_index("wallet_id")

    @property
    def deterministic_scores(self) -> pd.DataFrame:
        return self._load_artifact("deterministic_scores", "models/deterministic_scores.csv").set_index("wallet_id")

    @property
    def fusion_scores(self) -> pd.DataFrame:
        return self._load_artifact("fusion_scores", "models/fusion_scores.csv").set_index("wallet_id")

    @property
    def risk_predictions(self) -> pd.DataFrame:
        return self._load_artifact("risk_model_predictions", "models/risk_model_predictions.csv").set_index("wallet_id")

    @property
    def alerts(self) -> list[dict]:
        if self.job_id:
            return get_result(self.job_id, "alerts", db_path=self.db_path) or []
        import json

        with open(self.data_root / "alerts" / "alerts.json", encoding="utf-8") as f:
            return json.load(f)

    @property
    def shap_contributions(self) -> pd.DataFrame:
        return self._load_artifact(
            "shap_contributions", "explainability/top_feature_contributions.csv"
        )

    def transactions_for_job(self, raw_dir: Path | None) -> pd.DataFrame:
        candidates = []
        if raw_dir:
            candidates.append(Path(raw_dir) / "transactions.csv")
        candidates.append(self.data_root.parent / "raw" / "transactions.csv")
        candidates.append(OUTPUT_DIR / "raw" / "transactions.csv")

        for path in candidates:
            if path.exists():
                return pd.read_csv(path)
        if self.job_id:
            records = get_result(self.job_id, "datasets/transactions", db_path=self.db_path)
            if records is not None:
                return pd.DataFrame(records)
        raise FileNotFoundError("transactions.csv not found for job.")
