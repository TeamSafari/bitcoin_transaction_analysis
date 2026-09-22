"""Pipeline execution context shared across orchestrator stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd


ProgressCallback = Callable[[str, float], None]


@dataclass
class PipelineContext:
    """Mutable state carrier for a single pipeline run."""

    raw_dir: Path
    output_dir: Path
    artifact_dir: Path
    job_id: Optional[str] = None
    progress_callback: Optional[ProgressCallback] = None

    # Ingestion
    datasets: Dict[str, pd.DataFrame] = field(default_factory=dict)
    identity_resolution: Optional[pd.DataFrame] = None

    # Features & graph
    wallet_features: Optional[pd.DataFrame] = None
    scaled_features: Optional[pd.DataFrame] = None
    feature_names: List[str] = field(default_factory=list)
    graph_edges: Optional[pd.DataFrame] = None
    graph_features: Optional[pd.DataFrame] = None
    embeddings: Optional[pd.DataFrame] = None

    # Detectors
    isolation_scores: Optional[pd.DataFrame] = None
    autoencoder_scores: Optional[pd.DataFrame] = None
    deterministic_scores: Optional[pd.DataFrame] = None

    # Fusion & scoring
    fused_features: Optional[pd.DataFrame] = None
    risk_predictions: Optional[pd.DataFrame] = None
    fusion_scores: Optional[pd.DataFrame] = None
    routing_decisions: Optional[pd.DataFrame] = None

    # Explainability & alerts
    shap_contributions: Optional[pd.DataFrame] = None
    shap_values: Optional[pd.DataFrame] = None
    global_feature_importance: Optional[pd.DataFrame] = None
    deterministic_statistics: Dict[str, Any] = field(default_factory=dict)
    fusion_metadata: Dict[str, Any] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def features_dir(self) -> Path:
        return self.output_dir / "features"

    @property
    def graphs_dir(self) -> Path:
        return self.output_dir / "graphs"

    @property
    def models_dir(self) -> Path:
        return self.output_dir / "models"

    @property
    def explainability_dir(self) -> Path:
        return self.output_dir / "explainability"

    @property
    def alerts_dir(self) -> Path:
        return self.output_dir / "alerts"

    @property
    def reports_dir(self) -> Path:
        return self.output_dir / "reports"

    @property
    def identity_dir(self) -> Path:
        return self.output_dir / "identity"

    def ensure_dirs(self) -> None:
        for d in (
            self.features_dir,
            self.graphs_dir,
            self.models_dir,
            self.explainability_dir,
            self.alerts_dir,
            self.reports_dir,
            self.identity_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def report_progress(self, stage: str, fraction: float) -> None:
        if self.progress_callback:
            self.progress_callback(stage, fraction)
