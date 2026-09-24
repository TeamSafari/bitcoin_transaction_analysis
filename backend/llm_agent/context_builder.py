"""
Build a compact, token-efficient prompt from pipeline outputs.

Reads data from the DataLoader and assembles the four context slices
(risk verdict, SHAP factors, behavioral profile, anomaly signals)
into a single prompt string for the LLM.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

import pandas as pd

from backend.api.services.data_loader import DataLoader
from backend.llm_agent.config import (
    BEHAVIORAL_FEATURES,
    FEATURE_DISPLAY_NAMES,
    SHAP_EXCLUDE_PREFIXES,
    TOP_DET_EVIDENCE_COUNT,
    TOP_SHAP_COUNT,
)
from backend.llm_agent.prompts import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sats_to_btc(sats: float) -> str:
    """Convert satoshis to BTC with 4 decimal places."""
    btc = sats / 1e8
    if btc >= 1.0:
        return f"{btc:.4f} BTC"
    elif btc >= 0.001:
        return f"{btc:.6f} BTC"
    return f"{sats:.0f} sats"


def _seconds_to_human(seconds: float) -> str:
    """Convert seconds to a human-readable duration."""
    days = seconds / 86400
    if days >= 1:
        return f"{days:.1f} days"
    hours = seconds / 3600
    if hours >= 1:
        return f"{hours:.1f} hours"
    return f"{seconds:.0f} seconds"


def _display_name(feature: str) -> str:
    """Get human-readable display name for a feature."""
    return FEATURE_DISPLAY_NAMES.get(feature, feature.replace("_", " ").title())


def _format_feature_value(feature: str, value: float) -> str:
    """Format a feature value based on its type."""
    if "sats" in feature:
        return _sats_to_btc(value)
    if "duration" in feature or "seconds" in feature:
        return _seconds_to_human(value)
    if isinstance(value, float):
        if abs(value) < 0.01 and value != 0:
            return f"{value:.6f}"
        return f"{value:.4f}"
    return str(value)


# ---------------------------------------------------------------------------
# Slice builders
# ---------------------------------------------------------------------------

def _build_risk_verdict(wallet_id: str, loader: DataLoader) -> Dict[str, str]:
    """Slice 1: Risk scores and severity."""
    result: Dict[str, str] = {
        "risk_probability": "N/A",
        "severity": "N/A",
        "risk_level": "N/A",
        "fusion_score": "N/A",
        "deterministic_score": "N/A",
        "routing_action": "N/A",
    }

    # Risk predictions
    try:
        risk_row = loader.risk_predictions.loc[wallet_id]
        result["risk_probability"] = f"{float(risk_row['risk_probability']):.3f}"
    except (KeyError, FileNotFoundError):
        pass

    # Fusion scores
    try:
        fusion_row = loader.fusion_scores.loc[wallet_id]
        result["fusion_score"] = f"{float(fusion_row['statistical_aggregate_score']):.4f}"
        result["risk_level"] = str(fusion_row.get("risk_level", "N/A"))
    except (KeyError, FileNotFoundError):
        pass

    # Alerts (for severity and deterministic score)
    try:
        alerts = loader.alerts
        for alert in alerts:
            if str(alert.get("wallet_id")) == str(wallet_id):
                result["severity"] = str(alert.get("severity", "N/A"))
                det = alert.get("deterministic_score")
                if det is not None:
                    result["deterministic_score"] = f"{float(det):.4f}"
                break
    except (FileNotFoundError, Exception):
        pass

    # Routing decisions
    try:
        routing_df = loader._load_artifact(
            "routing_decisions", "models/routing_decisions.csv"
        )
        routing_row = routing_df[routing_df["wallet_id"] == wallet_id]
        if not routing_row.empty:
            result["routing_action"] = str(routing_row.iloc[0]["routing_action"])
    except (KeyError, FileNotFoundError):
        pass

    return result


def _build_shap_factors(wallet_id: str, loader: DataLoader) -> str:
    """Slice 2: Top SHAP factors, excluding opaque features."""
    try:
        shap_df = loader.shap_contributions
        w_shap = shap_df[shap_df["wallet_id"].astype(str) == str(wallet_id)].copy()
    except (FileNotFoundError, KeyError):
        return "No SHAP data available."

    if w_shap.empty:
        return "No SHAP data available for this wallet."

    # Filter out excluded prefixes (e.g., graphsage_dim_*)
    for prefix in SHAP_EXCLUDE_PREFIXES:
        w_shap = w_shap[~w_shap["feature"].str.startswith(prefix)]

    # Sort by absolute SHAP value and take top N
    w_shap = w_shap.sort_values("absolute_shap_value", ascending=False).head(
        TOP_SHAP_COUNT
    )

    lines = []
    for i, (_, row) in enumerate(w_shap.iterrows(), 1):
        feat = str(row["feature"])
        val = float(row.get("feature_value", 0))
        shap_val = float(row.get("shap_value", 0))
        direction = str(row.get("direction", "unknown"))

        display = _display_name(feat)
        formatted_val = _format_feature_value(feat, val)
        sign = "+" if shap_val > 0 else ""
        lines.append(
            f"{i}. {display}: value={formatted_val}, "
            f"impact={sign}{shap_val:.3f} ({direction})"
        )

    return "\n".join(lines)


def _build_behavior_summary(wallet_id: str, loader: DataLoader) -> str:
    """Slice 3: Key behavioral features from wallet_features.csv."""
    try:
        features_df = loader.wallet_features
        row = features_df.loc[wallet_id]
    except (KeyError, FileNotFoundError):
        return "No behavioral data available."

    parts = []
    for feat in BEHAVIORAL_FEATURES:
        try:
            val = float(row[feat])
            display = _display_name(feat)
            formatted = _format_feature_value(feat, val)
            parts.append(f"{display}={formatted}")
        except (KeyError, ValueError, TypeError):
            continue

    return ", ".join(parts) if parts else "No behavioral data available."


def _build_anomaly_signals(wallet_id: str, loader: DataLoader) -> str:
    """Slice 4: Anomaly detector outputs and deterministic evidence."""
    parts = []

    # Isolation Forest
    try:
        iso_df = loader._load_artifact(
            "isolation_forest_scores", "models/isolation_forest_scores.csv"
        ).set_index("wallet_id")
        iso_row = iso_df.loc[wallet_id]
        score = float(iso_row["isolation_forest_anomaly_score"])
        flag = int(iso_row["isolation_forest_flag"])
        parts.append(f"IF_score={score:.4f} (flagged={'yes' if flag else 'no'})")
    except (KeyError, FileNotFoundError):
        pass

    # Autoencoder
    try:
        ae_df = loader._load_artifact(
            "autoencoder_scores", "models/autoencoder_scores.csv"
        ).set_index("wallet_id")
        ae_row = ae_df.loc[wallet_id]
        ae_err = float(ae_row["autoencoder_reconstruction_error"])
        parts.append(f"AE_error={ae_err:.4f}")
    except (KeyError, FileNotFoundError):
        pass

    # Deterministic evidence (top items)
    try:
        det_df = loader.deterministic_scores
        det_row = det_df.loc[wallet_id]
        evidence_raw = str(det_row.get("deterministic_evidence", "[]"))
        evidence = json.loads(evidence_raw.replace('""', '"'))

        if evidence:
            det_parts = []
            for item in evidence[:TOP_DET_EVIDENCE_COUNT]:
                feat = str(item.get("feature", "?"))
                val = item.get("value", 0)
                pct = float(item.get("empirical_anomaly_score", 0)) * 100
                det_parts.append(f"{_display_name(feat)}={val} ({pct:.0f}th %ile)")
            if det_parts:
                parts.append("Deterministic: " + ", ".join(det_parts))
    except (KeyError, FileNotFoundError, json.JSONDecodeError):
        pass

    return "; ".join(parts) if parts else "No anomaly signal data available."


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_context(wallet_id: str, loader: DataLoader) -> tuple[str, str]:
    """
    Assemble system prompt and user prompt for the LLM.

    Returns:
        (system_prompt, user_prompt) tuple.
    """
    # Slice 1: Risk verdict
    risk = _build_risk_verdict(wallet_id, loader)

    # Slice 2: SHAP factors
    shap_text = _build_shap_factors(wallet_id, loader)

    # Slice 3: Behavioral profile
    behavior_text = _build_behavior_summary(wallet_id, loader)

    # Slice 4: Anomaly signals
    anomaly_text = _build_anomaly_signals(wallet_id, loader)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        wallet_id=wallet_id,
        risk_probability=risk["risk_probability"],
        severity=risk["severity"],
        risk_level=risk["risk_level"],
        routing_action=risk["routing_action"],
        fusion_score=risk["fusion_score"],
        deterministic_score=risk["deterministic_score"],
        shap_factors=shap_text,
        behavior_summary=behavior_text,
        anomaly_signals=anomaly_text,
    )

    return SYSTEM_PROMPT, user_prompt
