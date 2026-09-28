"""
Individual pipeline stage implementations.

Each function operates on a PipelineContext and writes artifacts to output_dir.
"""

from __future__ import annotations


import logging
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch_geometric.nn import SAGEConv

from backend.config import HOLD_THRESHOLD, REVIEW_THRESHOLD
from backend.feature_engineering.correlation_features import build_correlation_features
from backend.feature_engineering.network_features import build_network_features
from backend.feature_engineering.temporal_features import build_temporal_features
from backend.feature_engineering.transaction_features import build_transaction_features
from backend.graph.graph_engine import build_graph_edges
from backend.graph.graph_features import calculate_graph_features
from backend.graph.networkx_graph import build_wallet_graph
from backend.ingestion.identity_resolution import resolve_identities
from backend.ingestion.ingestion_pipeline import ingestion_report, run_ingestion, run_ingestion_from_db
from backend.pipeline.context import PipelineContext
from backend.risk.deterministic_engine import DeterministicRiskEngine
from backend.risk.risk_engine import StatisticalRiskEngine

logger = logging.getLogger(__name__)


class GraphSAGEEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class Autoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim_1: int = 64,
        hidden_dim_2: int = 32,
        latent_dim: int = 16,
    ):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        return self.decoder(latent), latent


# ---------------------------------------------------------------------------
# Stage 1: Data Ingestion
# ---------------------------------------------------------------------------

def stage_ingestion(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Data Ingestion", 0.05)
    ctx.datasets = (
        run_ingestion_from_db(ctx.job_id)
        if ctx.job_id
        else run_ingestion(ctx.raw_dir)
    )
    report = ingestion_report(ctx.datasets)
    logger.info(
        "Ingested %s wallets, %s transactions",
        report["wallet_count"],
        report["transaction_count"],
    )
    return report


# ---------------------------------------------------------------------------
# Stage 2: Identity Resolution
# ---------------------------------------------------------------------------

def stage_identity_resolution(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Identity Resolution", 0.10)
    ctx.datasets, ctx.identity_resolution = resolve_identities(ctx.datasets)

    dup_count = int(ctx.identity_resolution["duplicate_entity_flag"].sum())
    linked = int(ctx.identity_resolution["resolved_entity_id"].notna().sum())
    return {
        "wallets_linked_to_entities": linked,
        "duplicate_entity_flags": dup_count,
    }


# ---------------------------------------------------------------------------
# Stage 3: Graph Construction
# ---------------------------------------------------------------------------

def stage_graph_construction(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Graph Construction", 0.20)
    data = ctx.datasets
    transactions = data["transactions"]
    tx_inputs = data["transaction_inputs"]
    tx_outputs = data["transaction_outputs"]
    all_wallets = data["wallets"]["wallet_id"].astype(str).tolist()

    ctx.graph_edges = build_graph_edges(transactions, tx_inputs, tx_outputs)

    graph = build_wallet_graph(ctx.graph_edges, all_wallets)
    ctx.graph_features = calculate_graph_features(graph)

    return {"edge_count": len(ctx.graph_edges), "node_count": len(all_wallets)}


# ---------------------------------------------------------------------------
# Stage 4: Feature Engineering
# ---------------------------------------------------------------------------

def stage_feature_engineering(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Feature Engineering", 0.30)
    data = ctx.datasets
    wallets = data["wallets"].copy()
    wallets["wallet_id"] = wallets["wallet_id"].astype(str)

    tx_feats = build_transaction_features(
        wallets, data["transactions"], data["transaction_inputs"], data["transaction_outputs"]
    )
    temp_feats = build_temporal_features(data["transactions"], wallets)
    net_feats = build_network_features(data["network_observations"], data["ip_metadata"])
    corr_feats = build_correlation_features(data["transactions"], data["network_observations"])

    merged = pd.DataFrame({"wallet_id": wallets["wallet_id"].astype(str)})
    for frame in [tx_feats, temp_feats, net_feats, corr_feats]:
        if frame is not None and not frame.empty:
            f = frame.copy()
            f["wallet_id"] = f["wallet_id"].astype(str)
            f = f.drop_duplicates(subset=["wallet_id"])
            merged = merged.merge(f, on="wallet_id", how="left")

    if ctx.identity_resolution is not None:
        id_feats = ctx.identity_resolution[
            ["wallet_id", "unique_ip_count", "unique_asn_count", "entity_link_confidence"]
        ].copy()
        id_feats["wallet_id"] = id_feats["wallet_id"].astype(str)
        merged = merged.merge(id_feats, on="wallet_id", how="left")

    feature_cols = [c for c in merged.columns if c != "wallet_id"]
    for col in feature_cols:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    merged[feature_cols] = (
        merged[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
    )

    ctx.wallet_features = merged

    scaler_file = ctx.artifact_dir / "feature_scaler.pkl"
    if scaler_file.exists():
        scaler = joblib.load(scaler_file)
        if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(feature_cols):
            scaled_values = scaler.transform(merged[feature_cols])
        else:
            scaled_values = StandardScaler().fit_transform(merged[feature_cols])
    else:
        scaler = StandardScaler().fit(merged[feature_cols])
        scaled_values = scaler.transform(merged[feature_cols])

    ctx.scaled_features = pd.DataFrame(scaled_values, columns=feature_cols)
    ctx.scaled_features.insert(0, "wallet_id", merged["wallet_id"])
    ctx.feature_names = feature_cols

    return {"feature_count": len(feature_cols)}


# ---------------------------------------------------------------------------
# Stage 5: Rule Engine (Deterministic)
# ---------------------------------------------------------------------------

def stage_rule_engine(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Rule Engine", 0.42)
    engine = DeterministicRiskEngine()
    data_to_score = ctx.wallet_features.merge(ctx.graph_features, on="wallet_id", how="left")
    ctx.deterministic_scores = engine.run(data_to_score)
    ctx.deterministic_statistics = engine.feature_statistics

    flagged = int((ctx.deterministic_scores["deterministic_score"] >= REVIEW_THRESHOLD).sum())
    return {"deterministic_flagged_wallets": flagged}


# ---------------------------------------------------------------------------
# Stage 6: GNN Engine (GraphSAGE)
# ---------------------------------------------------------------------------

def stage_gnn_engine(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("GNN Engine", 0.52)
    model_file = ctx.artifact_dir / "graphsage.pt"
    if not model_file.exists():
        raise FileNotFoundError(f"GraphSAGE checkpoint missing: {model_file}")

    checkpoint = torch.load(model_file, weights_only=False, map_location="cpu")
    input_dim = checkpoint.get("input_dim", 52)
    hidden_dim = checkpoint.get("hidden_dim", 64)
    embedding_dim = checkpoint.get("embedding_dim", 32)
    node_feature_cols = checkpoint.get("node_feature_columns", [])

    feat_df = ctx.scaled_features.copy().set_index("wallet_id")
    if node_feature_cols:
        for col in node_feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0
        x_data = feat_df[node_feature_cols].values
    else:
        x_data = feat_df.values

    if x_data.shape[1] != input_dim:
        if x_data.shape[1] < input_dim:
            pad = np.zeros((x_data.shape[0], input_dim - x_data.shape[1]))
            x_data = np.hstack([x_data, pad])
        else:
            x_data = x_data[:, :input_dim]

    wallet_ids = ctx.scaled_features["wallet_id"].astype(str).tolist()
    wallet_to_idx = {wid: i for i, wid in enumerate(wallet_ids)}

    src_indices, dst_indices = [], []
    for _, row in ctx.graph_edges.iterrows():
        src, dst = str(row["source_wallet_id"]), str(row["target_wallet_id"])
        if src in wallet_to_idx and dst in wallet_to_idx:
            src_indices.append(wallet_to_idx[src])
            dst_indices.append(wallet_to_idx[dst])

    if not src_indices:
        src_indices = dst_indices = list(range(len(wallet_ids)))

    edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
    x_tensor = torch.tensor(x_data, dtype=torch.float32)

    model = GraphSAGEEncoder(input_dim, hidden_dim, embedding_dim)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        embeddings = model(x_tensor, edge_index).cpu().numpy()

    emb_cols = [f"graphsage_dim_{i}" for i in range(embedding_dim)]
    ctx.embeddings = pd.DataFrame(embeddings, columns=emb_cols)
    ctx.embeddings.insert(0, "wallet_id", wallet_ids)

    return {"embedding_dimension": embedding_dim}


# ---------------------------------------------------------------------------
# Stage 7: Anomaly Detectors (Isolation Forest + Autoencoder)
# ---------------------------------------------------------------------------

def _merge_detector_features(ctx: PipelineContext) -> pd.DataFrame:
    merged = ctx.wallet_features.merge(ctx.graph_features, on="wallet_id", how="left", suffixes=("", "_graph"))
    return merged.merge(ctx.embeddings, on="wallet_id", how="left", suffixes=("", "_graphsage"))


def stage_isolation_forest(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Isolation Forest", 0.62)
    model_file = ctx.artifact_dir / "isolation_forest.pkl"
    scaler_file = ctx.artifact_dir / "isolation_forest_scaler.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Isolation Forest artifact missing: {model_file}")

    artifact = joblib.load(model_file)
    if isinstance(artifact, dict):
        model = artifact["model"]
        feature_cols = artifact.get("feature_columns", [])
        scaler = artifact.get("scaler")
    else:
        model, feature_cols, scaler = artifact, [], None

    if scaler is None and scaler_file.exists():
        scaler = joblib.load(scaler_file)

    merged = _merge_detector_features(ctx)
    wallet_ids = merged["wallet_id"].astype(str).tolist()
    x_data = _align_feature_matrix(merged, feature_cols)
    x_data = np.nan_to_num(x_data, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is not None:
        try:
            x_data = scaler.transform(x_data)
        except Exception:
            pass

    raw_scores = -model.decision_function(x_data)
    minimum, maximum = raw_scores.min(), raw_scores.max()
    norm = (
        (raw_scores - minimum) / (maximum - minimum + 1e-9)
        if maximum > minimum
        else np.zeros(len(raw_scores))
    )

    ctx.isolation_scores = pd.DataFrame(
        {
            "wallet_id": wallet_ids,
            "isolation_forest_raw_score": raw_scores,
            "isolation_forest_anomaly_score": norm,
            "isolation_forest_flag": (model.predict(x_data) == -1).astype(int),
        }
    )
    return {"anomaly_flags": int(ctx.isolation_scores["isolation_forest_flag"].sum())}


def stage_autoencoder(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Autoencoder", 0.70)
    model_file = ctx.artifact_dir / "autoencoder.pt"
    scaler_file = ctx.artifact_dir / "autoencoder_scaler.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Autoencoder artifact missing: {model_file}")

    checkpoint = torch.load(model_file, weights_only=False, map_location="cpu")
    input_dim = checkpoint.get("input_dim", 94)
    feature_cols = checkpoint.get("feature_columns", [])

    scaler = None
    if scaler_file.exists():
        sc_art = joblib.load(scaler_file)
        scaler = sc_art.get("scaler") if isinstance(sc_art, dict) else sc_art

    merged = _merge_detector_features(ctx)
    wallet_ids = merged["wallet_id"].astype(str).tolist()
    x_data = _align_feature_matrix(merged, feature_cols)
    x_data = np.nan_to_num(x_data, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is not None:
        try:
            x_data = scaler.transform(x_data)
        except Exception:
            pass

    if x_data.shape[1] != input_dim:
        if x_data.shape[1] < input_dim:
            pad = np.zeros((x_data.shape[0], input_dim - x_data.shape[1]))
            x_data = np.hstack([x_data, pad])
        else:
            x_data = x_data[:, :input_dim]

    model = Autoencoder(
        input_dim,
        checkpoint.get("hidden_dim_1", 64),
        checkpoint.get("hidden_dim_2", 32),
        checkpoint.get("latent_dim", 16),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x_tensor = torch.tensor(x_data, dtype=torch.float32)
        recon, _ = model(x_tensor)
        recon_error = torch.mean((recon - x_tensor) ** 2, dim=1).cpu().numpy()

    min_err, max_err = recon_error.min(), recon_error.max()
    norm = (
        (recon_error - min_err) / (max_err - min_err + 1e-9)
        if max_err > min_err
        else np.zeros_like(recon_error)
    )

    ctx.autoencoder_scores = pd.DataFrame(
        {
            "wallet_id": wallet_ids,
            "autoencoder_reconstruction_error": recon_error,
            "autoencoder_anomaly_score": norm,
        }
    )
    return {"mean_reconstruction_error": float(recon_error.mean())}


def _align_feature_matrix(merged: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    if feature_cols:
        for col in feature_cols:
            if col not in merged.columns:
                merged[col] = 0.0
        return merged[feature_cols].values
    num_cols = [c for c in merged.columns if c not in ("wallet_id", "community_id")]
    return merged[num_cols].values


# ---------------------------------------------------------------------------
# Stage 8: Feature Fusion
# ---------------------------------------------------------------------------

def stage_feature_fusion(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Feature Fusion", 0.76)
    fused = ctx.wallet_features.copy()
    for df in (
        ctx.graph_features,
        ctx.embeddings,
        ctx.isolation_scores,
        ctx.autoencoder_scores,
        ctx.deterministic_scores,
    ):
        if df is not None and not df.empty:
            fused = fused.merge(df, on="wallet_id", how="left")

    cols_to_save = [c for c in fused.columns if c != "deterministic_evidence"]
    ctx.fused_features = fused
    return {"fused_feature_count": len(cols_to_save) - 1}


# ---------------------------------------------------------------------------
# Stage 9: Risk Model Inference
# ---------------------------------------------------------------------------

def stage_risk_model(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Risk Model Inference", 0.82)
    model_file = ctx.artifact_dir / "risk_model.pkl"
    if not model_file.exists():
        raise FileNotFoundError(f"Risk model artifact missing: {model_file}")

    artifact = joblib.load(model_file)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    feature_cols = artifact.get("feature_columns", []) if isinstance(artifact, dict) else []
    threshold = artifact.get("decision_threshold", 0.5) if isinstance(artifact, dict) else 0.5

    x_data, wallet_ids = _prepare_risk_matrix(ctx.fused_features, feature_cols)

    if hasattr(model, "set_params"):
        try:
            model.set_params(n_jobs=1)
        except Exception:
            pass

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x_data)[:, 1]
    else:
        probs = model.predict(x_data).astype(float)

    ctx.risk_predictions = pd.DataFrame(
        {
            "wallet_id": wallet_ids,
            "risk_probability": probs,
            "risk_prediction": (probs >= threshold).astype(int),
        }
    )
    return {"mean_risk_probability": float(probs.mean())}


def _prepare_risk_matrix(
    fused_features: pd.DataFrame, feature_cols: List[str]
) -> Tuple[np.ndarray, List[str]]:
    wallet_ids = fused_features["wallet_id"].astype(str).tolist()
    if feature_cols:
        col_data = {}
        for col in feature_cols:
            if col in fused_features.columns:
                col_data[col] = pd.to_numeric(fused_features[col], errors="coerce").fillna(0.0)
            else:
                col_data[col] = np.zeros(len(fused_features))
        x_data = pd.DataFrame(col_data, index=fused_features.index)[feature_cols].values
    else:
        drop_cols = ["wallet_id", "label", "deterministic_evidence"]
        num_cols = [c for c in fused_features.columns if c not in drop_cols]
        x_data = fused_features[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).values
    return x_data, wallet_ids


# ---------------------------------------------------------------------------
# Stage 10: Fusion & Scoring (StatisticalRiskEngine)
# ---------------------------------------------------------------------------

def stage_fusion_scoring(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Fusion & Scoring", 0.88)
    score_input = ctx.risk_predictions.merge(ctx.deterministic_scores, on="wallet_id", how="left")
    score_input = score_input.merge(ctx.isolation_scores, on="wallet_id", how="left")
    score_input = score_input.merge(ctx.autoencoder_scores, on="wallet_id", how="left")

    score_input = score_input.rename(
        columns={
            "risk_probability": "risk_model_probability",
            "autoencoder_reconstruction_error": "autoencoder_reconstruction_error",
        }
    )

    engine = StatisticalRiskEngine()
    ctx.fusion_scores = engine.run(score_input)
    ctx.fusion_metadata = {
        "engine_type": "pure_statistical",
        "source_weights": engine.source_weights,
    }

    # Apply configurable review/hold thresholds to final routing
    routing = []
    for _, row in ctx.fusion_scores.iterrows():
        prob = float(row.get("statistical_aggregate_score", 0.0))
        if prob >= HOLD_THRESHOLD:
            action = "HOLD"
        elif prob >= REVIEW_THRESHOLD:
            action = "REVIEW"
        else:
            action = "CLEAR"
        routing.append({"wallet_id": str(row["wallet_id"]), "routing_action": action})

    routing_df = pd.DataFrame(routing)
    ctx.routing_decisions = routing_df

    return {
        "source_weights": engine.source_weights,
        "hold_count": int((routing_df["routing_action"] == "HOLD").sum()),
        "review_count": int((routing_df["routing_action"] == "REVIEW").sum()),
    }


# ---------------------------------------------------------------------------
# Stage 11: Explainability (SHAP)
# ---------------------------------------------------------------------------

def stage_explainability(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Explainability", 0.93)
    model_file = ctx.artifact_dir / "risk_model.pkl"
    artifact = joblib.load(model_file)
    model = artifact["model"] if isinstance(artifact, dict) else artifact
    feature_cols = artifact.get("feature_columns", []) if isinstance(artifact, dict) else []

    wallet_ids = ctx.fused_features["wallet_id"].astype(str).tolist()
    x_matrix = _build_feature_matrix(ctx.fused_features, feature_cols)

    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_matrix)
        shap_matrix = shap_values[1] if isinstance(shap_values, list) and len(shap_values) == 2 else shap_values
    except Exception as exc:
        logger.warning("SHAP TreeExplainer fallback: %s", exc)
        importances = getattr(model, "feature_importances_", np.ones(len(feature_cols)))
        shap_matrix = x_matrix.values * importances

    shap_df = pd.DataFrame(shap_matrix, columns=feature_cols)
    shap_df.insert(0, "wallet_id", wallet_ids)
    ctx.shap_values = shap_df

    global_imp = np.abs(shap_matrix).mean(axis=0)
    ctx.global_feature_importance = pd.DataFrame({"feature": feature_cols, "mean_absolute_shap": global_imp}).sort_values(
        "mean_absolute_shap", ascending=False
    )

    top_rows = []
    for i, wid in enumerate(wallet_ids):
        w_shap = shap_matrix[i]
        w_vals = x_matrix.iloc[i].values
        top_indices = np.argsort(np.abs(w_shap))[::-1][:5]
        for rank, idx in enumerate(top_indices, 1):
            top_rows.append(
                {
                    "wallet_id": wid,
                    "rank": rank,
                    "feature": feature_cols[idx],
                    "feature_value": float(w_vals[idx]),
                    "shap_value": float(w_shap[idx]),
                    "absolute_shap_value": float(abs(w_shap[idx])),
                    "direction": "increases_risk" if w_shap[idx] > 0 else "decreases_risk",
                }
            )

    ctx.shap_contributions = pd.DataFrame(top_rows)
    return {"wallets_explained": len(wallet_ids)}


def _build_feature_matrix(fused_features: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    if feature_cols:
        col_data = {}
        for col in feature_cols:
            if col in fused_features.columns:
                col_data[col] = pd.to_numeric(fused_features[col], errors="coerce").fillna(0.0)
            else:
                col_data[col] = np.zeros(len(fused_features))
        return pd.DataFrame(col_data, index=fused_features.index)[feature_cols]
    drop_cols = ["wallet_id", "label", "deterministic_evidence"]
    cols = [c for c in fused_features.columns if c not in drop_cols]
    return fused_features[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)


# ---------------------------------------------------------------------------
# Stage 12: Alert Generation
# ---------------------------------------------------------------------------

def stage_alerts(ctx: PipelineContext) -> Dict[str, Any]:
    ctx.report_progress("Alert Generation", 0.97)
    merged = ctx.risk_predictions.merge(ctx.deterministic_scores, on="wallet_id", how="left")
    merged = merged.merge(ctx.isolation_scores, on="wallet_id", how="left")
    merged = merged.merge(ctx.autoencoder_scores, on="wallet_id", how="left")
    if ctx.fusion_scores is not None:
        merged = merged.merge(
            ctx.fusion_scores[["wallet_id", "statistical_aggregate_score", "risk_level"]],
            on="wallet_id",
            how="left",
        )

    top_by_wallet: Dict[str, list] = {}
    for wid, group in ctx.shap_contributions.groupby("wallet_id"):
        top_by_wallet[str(wid)] = group.to_dict(orient="records")

    alerts = []
    for _, row in merged.iterrows():
        wid = str(row["wallet_id"])
        risk_prob = float(row.get("risk_probability", 0.0))
        det_score = float(row.get("deterministic_score", 0.0))
        fusion_score = float(row.get("statistical_aggregate_score", risk_prob))

        if risk_prob >= HOLD_THRESHOLD:
            severity = "HIGH"
        elif risk_prob >= REVIEW_THRESHOLD:
            severity = "MEDIUM"
        else:
            severity = "LOW"

        alerts.append(
            {
                "wallet_id": wid,
                "severity": severity,
                "risk_probability": round(risk_prob, 4),
                "fusion_score": round(fusion_score, 4),
                "risk_prediction": int(row.get("risk_prediction", 0)),
                "deterministic_score": round(det_score, 4),
                "isolation_forest_score": round(float(row.get("isolation_forest_anomaly_score", 0.0)), 4),
                "autoencoder_score": round(float(row.get("autoencoder_anomaly_score", 0.0)), 4),
                "risk_level": str(row.get("risk_level", severity)),
                "top_shap_factors": top_by_wallet.get(wid, [])[:5],
            }
        )

    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    alerts.sort(key=lambda a: (severity_order.get(a["severity"], 3), -a["risk_probability"]))
    ctx.alerts = alerts

    high = sum(1 for a in alerts if a["severity"] == "HIGH")
    med = sum(1 for a in alerts if a["severity"] == "MEDIUM")
    return {
        "total_alerts": len(alerts),
        "high_risk_count": high,
        "medium_risk_count": med,
        "low_risk_count": len(alerts) - high - med,
    }
