"""SHAP-based wallet risk explainability for the investigator console."""

from __future__ import annotations

import json
from typing import Any, Dict, List

import pandas as pd

from backend.api.services.data_loader import DataLoader


class ExplainabilityEngine:
    FEATURE_DESCRIPTIONS = {
        "pagerank": "Importance in transaction network (higher = more central/intermediary)",
        "betweenness_centrality": "Frequency of appearing on shortest paths between wallets",
        "out_degree": "Number of unique outgoing transaction recipients",
        "in_degree": "Number of unique incoming transaction sources",
        "fan_out": "Distribution spread (many recipients per transaction)",
        "fan_in": "Collection spread (many sources per transaction)",
        "tx_count": "Total transaction count (activity volume)",
        "unique_asns": "Diversity of network providers used",
        "off_hour_transaction_ratio": "Percentage of transactions outside business hours",
    }

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def explain_wallet_risk(self, wallet_id: str) -> Dict[str, Any]:
        factors: List[Dict[str, Any]] = []
        try:
            shap_df = self.loader.shap_contributions
            w_shap = shap_df[shap_df["wallet_id"].astype(str) == str(wallet_id)]
            for _, row in w_shap.iterrows():
                fname = str(row["feature"])
                shap_val = float(row.get("shap_value", 0.0))
                factors.append(
                    {
                        "feature": fname,
                        "feature_name": fname,
                        "shap_value": round(shap_val, 4),
                        "shap_impact": round(min(1.0, max(0.0, abs(shap_val) / 5)), 4),
                        "raw_value": round(float(row.get("feature_value", 0.0)), 4),
                        "direction": str(row.get("direction", "increases_risk")),
                        "description": self.FEATURE_DESCRIPTIONS.get(fname, f"Statistical impact in {fname}"),
                        "percentile_rank": round(float(min(100.0, max(0.0, abs(shap_val) * 20))), 2),
                    }
                )
        except FileNotFoundError:
            pass

        try:
            det_score = float(self.loader.deterministic_scores.loc[wallet_id]["deterministic_score"])
        except KeyError:
            det_score = None

        try:
            risk_score = float(self.loader.risk_predictions.loc[wallet_id]["risk_probability"])
        except KeyError:
            risk_score = 0.5

        try:
            fusion_score = float(
                self.loader.fusion_scores.loc[wallet_id]["statistical_aggregate_score"]
            )
        except KeyError:
            fusion_score = risk_score

        if not factors:
            factors = self._fallback_from_deterministic(wallet_id)

        overall = (risk_score + (fusion_score or risk_score)) / 2

        return {
            "wallet_id": wallet_id,
            "overall_risk_score": round(min(1.0, max(0.0, overall)), 4),
            "model_used": "XGBoost + SHAP + Statistical Fusion",
            "top_risk_factors": factors[:10],
            "deterministic_score": round(det_score, 4) if det_score is not None else None,
            "fusion_score": round(fusion_score, 4) if fusion_score is not None else None,
        }

    def _fallback_from_deterministic(self, wallet_id: str) -> List[Dict[str, Any]]:
        try:
            row = self.loader.deterministic_scores.loc[wallet_id]
            evidence = json.loads(str(row.get("deterministic_evidence", "[]")).replace('""', '"'))
        except Exception:
            return []

        factors = []
        for item in evidence[:5]:
            score = float(item.get("empirical_anomaly_score", 0.5))
            fname = str(item.get("feature", "unknown"))
            factors.append(
                {
                    "feature": fname,
                    "feature_name": fname,
                    "shap_value": round(score * 2.0, 4),
                    "shap_impact": round(score, 4),
                    "raw_value": round(float(item.get("value", 0.0)), 4),
                    "direction": "increases_risk",
                    "description": self.FEATURE_DESCRIPTIONS.get(fname, f"Statistical anomaly in {fname}"),
                    "percentile_rank": round(score * 100, 2),
                }
            )
        return factors
