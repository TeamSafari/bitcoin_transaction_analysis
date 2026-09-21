"""
Bitcoin Transaction Forensics — Pipeline Orchestrator

End-to-end workflow:
  1. Data Ingestion
  2. Identity Resolution
  3. Graph Construction
  4. Feature Engineering
  5. Rule Engine (deterministic)
  6. GNN Engine (GraphSAGE)
  7. Anomaly Detectors (Isolation Forest + Autoencoder)
  8. Feature Fusion
  9. Risk Model Inference
 10. Fusion & Scoring (StatisticalRiskEngine)
 11. Explainability (SHAP)
 12. Alert Generation
"""

from __future__ import annotations

import backend.bootstrap  # noqa: F401

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from backend.config import ARTIFACT_DIR, RAW_DATA_DIR
from backend.pipeline.context import PipelineContext, ProgressCallback
from backend.pipeline import stages

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Orchestrator:
    """Coordinates sequential execution of all forensic pipeline stages."""

    STAGE_SEQUENCE = [
        ("Data Ingestion", stages.stage_ingestion),
        ("Identity Resolution", stages.stage_identity_resolution),
        ("Graph Construction", stages.stage_graph_construction),
        ("Feature Engineering", stages.stage_feature_engineering),
        ("Rule Engine", stages.stage_rule_engine),
        ("GNN Engine", stages.stage_gnn_engine),
        ("Isolation Forest", stages.stage_isolation_forest),
        ("Autoencoder", stages.stage_autoencoder),
        ("Feature Fusion", stages.stage_feature_fusion),
        ("Risk Model Inference", stages.stage_risk_model),
        ("Fusion & Scoring", stages.stage_fusion_scoring),
        ("Explainability", stages.stage_explainability),
        ("Alert Generation", stages.stage_alerts),
    ]

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        artifact_dir: Optional[Path] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ):
        self.ctx = PipelineContext(
            raw_dir=Path(raw_dir),
            output_dir=Path(output_dir),
            artifact_dir=Path(artifact_dir) if artifact_dir else ARTIFACT_DIR,
            progress_callback=progress_callback,
        )

    def execute(self) -> Dict[str, Any]:
        start = time.time()
        self.ctx.ensure_dirs()
        stage_reports: Dict[str, Any] = {}

        logger.info(
            "Starting pipeline: raw=%s output=%s",
            self.ctx.raw_dir,
            self.ctx.output_dir,
        )

        for stage_name, stage_fn in self.STAGE_SEQUENCE:
            logger.info("Running stage: %s", stage_name)
            stage_reports[stage_name] = stage_fn(self.ctx)

        elapsed = time.time() - start
        self.ctx.report_progress("Pipeline Completed", 1.0)

        alerts_summary = stage_reports.get("Alert Generation", {})
        ingestion_summary = stage_reports.get("Data Ingestion", {})
        graph_summary = stage_reports.get("Graph Construction", {})

        manifest = {
            "status": "success",
            "execution_time_seconds": round(elapsed, 2),
            "stages_completed": [name for name, _ in self.STAGE_SEQUENCE],
            "dataset": {
                "wallet_count": ingestion_summary.get("wallet_count", 0),
                "transaction_count": ingestion_summary.get("transaction_count", 0),
                "edge_count": graph_summary.get("edge_count", 0),
            },
            "identity": stage_reports.get("Identity Resolution", {}),
            "features": stage_reports.get("Feature Engineering", {}),
            "fusion": stage_reports.get("Fusion & Scoring", {}),
            "alerts_summary": alerts_summary,
            "stage_reports": stage_reports,
        }

        manifest_path = self.ctx.reports_dir / "pipeline_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info("Pipeline completed in %.2fs", elapsed)
        return manifest


def run_pipeline(
    raw_dir: str | Path,
    output_dir: str | Path,
    artifact_dir: Optional[str | Path] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    """Run the full forensic pipeline and return the execution manifest."""
    orchestrator = Orchestrator(
        raw_dir=Path(raw_dir),
        output_dir=Path(output_dir),
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        progress_callback=progress_callback,
    )
    return orchestrator.execute()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Bitcoin forensic pipeline.")
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DATA_DIR))
    parser.add_argument("--output-dir", type=str, default=str(PROJECT_ROOT / "outputs"))
    parser.add_argument("--artifact-dir", type=str, default=str(ARTIFACT_DIR))
    args = parser.parse_args()

    try:
        manifest = run_pipeline(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            artifact_dir=args.artifact_dir,
        )
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION SUMMARY")
        print("=" * 60)
        print(json.dumps(manifest, indent=2))
    except Exception as exc:
        print(f"\n[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
