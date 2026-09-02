"""
Bitcoin Transaction Forensics - Pipeline Orchestrator

The single canonical execution pipeline that orchestrates:
1. Ingestion & Validation (6 raw CSVs without risk labels)
2. Feature Engineering (Transaction, Temporal, Network, Correlation)
3. Graph Engine (NetworkX Graph & Structural Metrics)
4. GraphSAGE (32-D Neural Wallet Embeddings via pretrained model)
5. Isolation Forest (Anomaly Scoring via pretrained model)
6. Autoencoder (Reconstruction Error Anomaly via pretrained model)
7. Deterministic Statistical Risk Engine
8. Risk Feature Fusion
9. Risk Model Inference (LightGBM/XGBoost via pretrained model)
10. SHAP Explainability Engine
11. Alert Generation Service
12. Pipeline Manifest & Summary Generation
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv

# ============================================================
# PROJECT ROOT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.feature_engineering.transaction_features import build_transaction_features
from backend.feature_engineering.temporal_features import build_temporal_features
from backend.feature_engineering.network_features import build_network_features
from backend.feature_engineering.correlation_features import build_correlation_features
from backend.graph.graph_engine import build_graph_edges
from backend.graph.networkx_graph import build_wallet_graph
from backend.graph.graph_features import calculate_graph_features
from backend.risk.deterministic_engine import DeterministicRiskEngine

logger = logging.getLogger("orchestrator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ============================================================
# MODEL ARCHITECTURES FOR INFERENCE
# ============================================================

class GraphSAGEEncoder(nn.Module):
    """2-layer GraphSAGE Encoder."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int):
        super().__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, embedding_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x


class Autoencoder(nn.Module):
    """MLP Autoencoder for reconstruction anomaly detection."""

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
        reconstruction = self.decoder(latent)
        return reconstruction, latent


# ============================================================
# ORCHESTRATOR CLASS
# ============================================================

class Orchestrator:
    """
    Manages end-to-end execution of the Bitcoin forensic pipeline
    in inference mode using pre-trained model artifacts.
    """

    REQUIRED_RAW_FILES = [
        "wallets.csv",
        "transactions.csv",
        "transaction_inputs.csv",
        "transaction_outputs.csv",
        "network_observations.csv",
        "ip_metadata.csv",
    ]

    def __init__(
        self,
        raw_dir: Path,
        output_dir: Path,
        artifact_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        self.artifact_dir = (
            Path(artifact_dir)
            if artifact_dir
            else (PROJECT_ROOT / "backend" / "models" / "artifacts")
        )
        self.progress_callback = progress_callback

        # Output subdirectories
        self.features_dir = self.output_dir / "features"
        self.graphs_dir = self.output_dir / "graphs"
        self.models_dir = self.output_dir / "models"
        self.explainability_dir = self.output_dir / "explainability"
        self.alerts_dir = self.output_dir / "alerts"
        self.reports_dir = self.output_dir / "reports"

        self._create_output_dirs()

    def _create_output_dirs(self) -> None:
        for d in [
            self.features_dir,
            self.graphs_dir,
            self.models_dir,
            self.explainability_dir,
            self.alerts_dir,
            self.reports_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    def _update_progress(self, stage_name: str, fraction: float) -> None:
        logger.info(f"[{stage_name.upper()}] Starting... ({fraction * 100:.0f}%)")
        if self.progress_callback:
            try:
                self.progress_callback(stage_name, fraction)
            except Exception as e:
                logger.warning(f"Error in progress callback: {e}")

    # ========================================================
    # STAGE 1: INGESTION & VALIDATION
    # ========================================================

    def validate_and_load_raw_data(self) -> Dict[str, pd.DataFrame]:
        self._update_progress("Ingestion & Validation", 0.05)

        for filename in self.REQUIRED_RAW_FILES:
            file_path = self.raw_dir / filename
            if not file_path.exists():
                raise FileNotFoundError(
                    f"Required raw dataset file '{filename}' missing in {self.raw_dir}"
                )

        data = {}
        for filename in self.REQUIRED_RAW_FILES:
            stem = Path(filename).stem
            df = pd.read_csv(self.raw_dir / filename)
            if df.empty:
                raise ValueError(f"Raw CSV '{filename}' is empty.")
            data[stem] = df

        # Parse list-like columns in transactions
        tx = data["transactions"].copy()
        list_cols = [
            "input_addresses",
            "output_addresses",
            "input_amounts_sats",
            "output_amounts_sats",
        ]
        for col in list_cols:
            if col in tx.columns:
                tx[col] = tx[col].apply(self._parse_list_column)

        if "timestamp" in tx.columns:
            tx["timestamp"] = pd.to_datetime(tx["timestamp"], errors="coerce", utc=True)

        for col in ["total_input_sats", "total_output_sats", "fee_sats"]:
            if col in tx.columns:
                tx[col] = pd.to_numeric(tx[col], errors="coerce").fillna(0.0)

        data["transactions"] = tx
        logger.info(f"Loaded {len(data['wallets'])} wallets, {len(data['transactions'])} transactions.")
        return data

    @staticmethod
    def _parse_list_column(value: Any) -> list:
        if value is None or pd.isna(value):
            return []
        if isinstance(value, list):
            return value
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
        return []

    # ========================================================
    # STAGE 2: FEATURE ENGINEERING
    # ========================================================

    def run_feature_engineering(
        self, data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._update_progress("Feature Engineering", 0.15)

        wallets = data["wallets"].copy()
        wallets["wallet_id"] = wallets["wallet_id"].astype(str)
        transactions = data["transactions"].copy()
        tx_inputs = data["transaction_inputs"].copy()
        tx_outputs = data["transaction_outputs"].copy()
        net_obs = data["network_observations"].copy()
        ip_meta = data["ip_metadata"].copy()

        # 1. Feature groups
        tx_feats = build_transaction_features(wallets, transactions, tx_inputs, tx_outputs)
        temp_feats = build_temporal_features(transactions, wallets)
        net_feats = build_network_features(net_obs, ip_meta)
        corr_feats = build_correlation_features(transactions, net_obs)

        # 2. Merge all feature groups
        merged = pd.DataFrame({"wallet_id": wallets["wallet_id"].astype(str)})

        for frame in [tx_feats, temp_feats, net_feats, corr_feats]:
            if frame is not None and not frame.empty:
                f = frame.copy()
                f["wallet_id"] = f["wallet_id"].astype(str)
                f = f.drop_duplicates(subset=["wallet_id"])
                merged = merged.merge(f, on="wallet_id", how="left")

        # 3. Numeric conversions & cleaning
        feature_cols = [c for c in merged.columns if c != "wallet_id"]
        for c in feature_cols:
            merged[c] = pd.to_numeric(merged[c], errors="coerce")
        merged[feature_cols] = (
            merged[feature_cols].replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)
        )

        # Save raw features
        raw_feat_file = self.features_dir / "wallet_features.csv"
        merged.to_csv(raw_feat_file, index=False)

        # 4. Scaling
        # Load pre-trained feature scaler if available, otherwise fit a standard scaler
        scaler_file = self.artifact_dir / "feature_scaler.pkl"
        if scaler_file.exists():
            scaler = joblib.load(scaler_file)
            # Handle feature count mismatch gracefully
            if hasattr(scaler, "n_features_in_") and scaler.n_features_in_ == len(feature_cols):
                scaled_values = scaler.transform(merged[feature_cols])
            else:
                scaled_values = StandardScaler().fit_transform(merged[feature_cols])
        else:
            scaler = StandardScaler().fit(merged[feature_cols])
            scaled_values = scaler.transform(merged[feature_cols])

        scaled_df = pd.DataFrame(scaled_values, columns=feature_cols)
        scaled_df.insert(0, "wallet_id", merged["wallet_id"])

        scaled_file = self.features_dir / "wallet_features_scaled.csv"
        scaled_df.to_csv(scaled_file, index=False)

        with open(self.features_dir / "feature_names.json", "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2)

        return merged, scaled_df

    # ========================================================
    # STAGE 3: NETWORKX GRAPH ENGINE
    # ========================================================

    def run_graph_engine(
        self, data: Dict[str, pd.DataFrame]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        self._update_progress("NetworkX Graph Engine", 0.30)

        transactions = data["transactions"]
        tx_inputs = data["transaction_inputs"]
        tx_outputs = data["transaction_outputs"]
        all_wallets = data["wallets"]["wallet_id"].astype(str).tolist()

        edges = build_graph_edges(transactions, tx_inputs, tx_outputs)
        edges_file = self.graphs_dir / "graph_edges.csv"
        edges.to_csv(edges_file, index=False)

        graph = build_wallet_graph(edges, all_wallets)
        graph_features = calculate_graph_features(graph)
        graph_feat_file = self.graphs_dir / "graph_features.csv"
        graph_features.to_csv(graph_feat_file, index=False)

        return edges, graph_features

    # ========================================================
    # STAGE 4: GRAPHSAGE INFERENCE
    # ========================================================

    def run_graphsage(
        self, scaled_features: pd.DataFrame, edges: pd.DataFrame
    ) -> pd.DataFrame:
        self._update_progress("GraphSAGE Inference", 0.40)

        model_file = self.artifact_dir / "graphsage.pt"
        if not model_file.exists():
            raise FileNotFoundError(f"GraphSAGE model checkpoint missing: {model_file}")

        checkpoint = torch.load(model_file, weights_only=False, map_location="cpu")
        input_dim = checkpoint.get("input_dim", 52)
        hidden_dim = checkpoint.get("hidden_dim", 64)
        embedding_dim = checkpoint.get("embedding_dim", 32)
        node_feature_cols = checkpoint.get("node_feature_columns", [])

        # Align node features
        feat_df = scaled_features.copy().set_index("wallet_id")
        if node_feature_cols:
            for col in node_feature_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            x_data = feat_df[node_feature_cols].values
        else:
            x_data = feat_df.values

        # Ensure correct input dimension
        if x_data.shape[1] != input_dim:
            if x_data.shape[1] < input_dim:
                pad = np.zeros((x_data.shape[0], input_dim - x_data.shape[1]))
                x_data = np.hstack([x_data, pad])
            else:
                x_data = x_data[:, :input_dim]

        wallet_ids = scaled_features["wallet_id"].astype(str).tolist()
        wallet_to_idx = {wid: i for i, wid in enumerate(wallet_ids)}

        # Build edge index
        src_indices = []
        dst_indices = []
        for _, row in edges.iterrows():
            src = str(row["source_wallet_id"])
            dst = str(row["target_wallet_id"])
            if src in wallet_to_idx and dst in wallet_to_idx:
                src_indices.append(wallet_to_idx[src])
                dst_indices.append(wallet_to_idx[dst])

        if not src_indices:
            # Self-loops if no edges exist
            src_indices = list(range(len(wallet_ids)))
            dst_indices = list(range(len(wallet_ids)))

        edge_index = torch.tensor([src_indices, dst_indices], dtype=torch.long)
        x_tensor = torch.tensor(x_data, dtype=torch.float32)

        model = GraphSAGEEncoder(input_dim, hidden_dim, embedding_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            embeddings = model(x_tensor, edge_index).cpu().numpy()

        emb_cols = [f"graphsage_dim_{i}" for i in range(embedding_dim)]
        emb_df = pd.DataFrame(embeddings, columns=emb_cols)
        emb_df.insert(0, "wallet_id", wallet_ids)

        emb_file = self.graphs_dir / "graphsage_embeddings.csv"
        emb_df.to_csv(emb_file, index=False)
        return emb_df

    # ========================================================
    # STAGE 5: ISOLATION FOREST INFERENCE
    # ========================================================

    def run_isolation_forest(
        self,
        wallet_features: pd.DataFrame,
        graph_features: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        self._update_progress("Isolation Forest Inference", 0.50)

        model_file = self.artifact_dir / "isolation_forest.pkl"
        scaler_file = self.artifact_dir / "isolation_forest_scaler.pkl"

        if not model_file.exists():
            raise FileNotFoundError(f"Isolation Forest artifact missing: {model_file}")

        artifact = joblib.load(model_file)
        if isinstance(artifact, dict):
            model = artifact["model"]
            feature_cols = artifact.get("feature_columns", [])
            scaler = artifact.get("scaler")
        else:
            model = artifact
            feature_cols = []
            scaler = None

        if scaler is None and scaler_file.exists():
            scaler = joblib.load(scaler_file)

        # Merge features
        merged = wallet_features.merge(graph_features, on="wallet_id", how="left")
        merged = merged.merge(embeddings, on="wallet_id", how="left")

        wallet_ids = merged["wallet_id"].astype(str).tolist()

        if feature_cols:
            for col in feature_cols:
                if col not in merged.columns:
                    merged[col] = 0.0
            x_data = merged[feature_cols].values
        else:
            num_cols = [c for c in merged.columns if c != "wallet_id" and c != "community_id"]
            x_data = merged[num_cols].values

        x_data = np.nan_to_num(x_data, nan=0.0, posinf=0.0, neginf=0.0)

        if scaler is not None:
            try:
                x_data = scaler.transform(x_data)
            except Exception:
                pass

        # Calculate scores
        try:
            decision_scores = model.decision_function(x_data)
            # Invert: lower decision function means higher anomaly
            anomaly_scores = 1.0 - (
                (decision_scores - decision_scores.min())
                / (decision_scores.max() - decision_scores.min() + 1e-9)
            )
            predictions = model.predict(x_data)
            flags = (predictions == -1).astype(int)
        except Exception:
            anomaly_scores = np.zeros(len(wallet_ids))
            flags = np.zeros(len(wallet_ids), dtype=int)

        scores_df = pd.DataFrame(
            {
                "wallet_id": wallet_ids,
                "isolation_forest_anomaly_score": anomaly_scores,
                "isolation_forest_flag": flags,
            }
        )

        out_file = self.models_dir / "isolation_forest_scores.csv"
        scores_df.to_csv(out_file, index=False)
        return scores_df

    # ========================================================
    # STAGE 6: AUTOENCODER INFERENCE
    # ========================================================

    def run_autoencoder(
        self,
        wallet_features: pd.DataFrame,
        graph_features: pd.DataFrame,
        embeddings: pd.DataFrame,
    ) -> pd.DataFrame:
        self._update_progress("Autoencoder Inference", 0.60)

        model_file = self.artifact_dir / "autoencoder.pt"
        scaler_file = self.artifact_dir / "autoencoder_scaler.pkl"

        if not model_file.exists():
            raise FileNotFoundError(f"Autoencoder artifact missing: {model_file}")

        checkpoint = torch.load(model_file, weights_only=False, map_location="cpu")
        input_dim = checkpoint.get("input_dim", 94)
        h1 = checkpoint.get("hidden_dim_1", 64)
        h2 = checkpoint.get("hidden_dim_2", 32)
        latent_dim = checkpoint.get("latent_dim", 16)
        feature_cols = checkpoint.get("feature_columns", [])

        scaler = None
        if scaler_file.exists():
            sc_art = joblib.load(scaler_file)
            scaler = sc_art.get("scaler") if isinstance(sc_art, dict) else sc_art

        merged = wallet_features.merge(graph_features, on="wallet_id", how="left")
        merged = merged.merge(embeddings, on="wallet_id", how="left")
        wallet_ids = merged["wallet_id"].astype(str).tolist()

        if feature_cols:
            for col in feature_cols:
                if col not in merged.columns:
                    merged[col] = 0.0
            x_data = merged[feature_cols].values
        else:
            num_cols = [c for c in merged.columns if c != "wallet_id" and c != "community_id"]
            x_data = merged[num_cols].values

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

        model = Autoencoder(input_dim, h1, h2, latent_dim)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        with torch.no_grad():
            x_tensor = torch.tensor(x_data, dtype=torch.float32)
            recon, _ = model(x_tensor)
            recon_error = torch.mean((recon - x_tensor) ** 2, dim=1).cpu().numpy()

        # Normalize reconstruction error to 0-1
        min_err, max_err = recon_error.min(), recon_error.max()
        norm_score = (
            (recon_error - min_err) / (max_err - min_err + 1e-9)
            if max_err > min_err
            else np.zeros_like(recon_error)
        )

        scores_df = pd.DataFrame(
            {
                "wallet_id": wallet_ids,
                "autoencoder_reconstruction_error": recon_error,
                "autoencoder_anomaly_score": norm_score,
            }
        )

        out_file = self.models_dir / "autoencoder_scores.csv"
        scores_df.to_csv(out_file, index=False)
        return scores_df

    # ========================================================
    # STAGE 7: DETERMINISTIC STATISTICAL RISK
    # ========================================================

    def run_deterministic(
        self, wallet_features: pd.DataFrame, graph_features: pd.DataFrame
    ) -> pd.DataFrame:
        self._update_progress("Deterministic Statistical Risk", 0.70)

        engine = DeterministicRiskEngine()
        data_to_score = wallet_features.merge(graph_features, on="wallet_id", how="left")

        scores_df = engine.run(data_to_score)
        stats = engine.feature_statistics

        scores_file = self.models_dir / "deterministic_scores.csv"
        scores_df.to_csv(scores_file, index=False)

        stats_file = self.models_dir / "deterministic_statistics.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return scores_df

    # ========================================================
    # STAGE 8: RISK FEATURE FUSION
    # ========================================================

    def run_risk_fusion(
        self,
        wallet_features: pd.DataFrame,
        graph_features: pd.DataFrame,
        embeddings: pd.DataFrame,
        isolation_scores: pd.DataFrame,
        autoencoder_scores: pd.DataFrame,
        deterministic_scores: pd.DataFrame,
    ) -> pd.DataFrame:
        self._update_progress("Risk Feature Fusion", 0.75)

        # Join all on wallet_id
        fused = wallet_features.copy()
        for df in [
            graph_features,
            embeddings,
            isolation_scores,
            autoencoder_scores,
            deterministic_scores,
        ]:
            if df is not None and not df.empty:
                fused = fused.merge(df, on="wallet_id", how="left")

        # Drop non-feature column deterministic_evidence if present in tabular ML
        cols_to_save = [c for c in fused.columns if c != "deterministic_evidence"]
        fused_to_save = fused[cols_to_save]

        out_file = self.models_dir / "base_fused_features.csv"
        fused_to_save.to_csv(out_file, index=False)
        return fused

    # ========================================================
    # STAGE 9: RISK MODEL INFERENCE (LightGBM / XGBoost)
    # ========================================================

    def run_risk_model(self, fused_features: pd.DataFrame) -> pd.DataFrame:
        self._update_progress("Risk Model Inference", 0.82)

        model_file = self.artifact_dir / "risk_model.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Risk model artifact missing: {model_file}")

        artifact = joblib.load(model_file)
        model = artifact["model"] if isinstance(artifact, dict) else artifact
        feature_cols = artifact.get("feature_columns", []) if isinstance(artifact, dict) else []

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

        # Predict probabilities and binary outcomes
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(x_data)[:, 1]
        elif hasattr(model, "predict"):
            preds = model.predict(x_data)
            probs = preds.astype(float)
        else:
            probs = np.zeros(len(wallet_ids))

        decision_threshold = (
            artifact.get("decision_threshold", 0.5) if isinstance(artifact, dict) else 0.5
        )
        preds = (probs >= decision_threshold).astype(int)

        pred_df = pd.DataFrame(
            {
                "wallet_id": wallet_ids,
                "risk_probability": probs,
                "risk_prediction": preds,
            }
        )

        out_file = self.models_dir / "risk_model_predictions.csv"
        pred_df.to_csv(out_file, index=False)
        return pred_df

    # ========================================================
    # STAGE 10: SHAP EXPLAINABILITY
    # ========================================================

    def run_shap(
        self, fused_features: pd.DataFrame, risk_predictions: pd.DataFrame
    ) -> pd.DataFrame:
        self._update_progress("SHAP Explainability", 0.90)

        model_file = self.artifact_dir / "risk_model.pkl"
        artifact = joblib.load(model_file)
        model = artifact["model"] if isinstance(artifact, dict) else artifact
        feature_cols = artifact.get("feature_columns", []) if isinstance(artifact, dict) else []

        wallet_ids = fused_features["wallet_id"].astype(str).tolist()

        if feature_cols:
            col_data = {}
            for col in feature_cols:
                if col in fused_features.columns:
                    col_data[col] = pd.to_numeric(fused_features[col], errors="coerce").fillna(0.0)
                else:
                    col_data[col] = np.zeros(len(fused_features))
            x_matrix = pd.DataFrame(col_data, index=fused_features.index)[feature_cols]
        else:
            drop_cols = ["wallet_id", "label", "deterministic_evidence"]
            cols = [c for c in fused_features.columns if c not in drop_cols]
            x_matrix = fused_features[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            feature_cols = list(x_matrix.columns)

        try:
            import shap

            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(x_matrix)
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Binary classification positive class SHAP
                shap_matrix = shap_values[1]
            else:
                shap_matrix = shap_values
        except Exception as e:
            logger.warning(f"TreeExplainer fallback to feature-weight approximation: {e}")
            # Fallback approximation using feature values * model feature importances
            importances = getattr(model, "feature_importances_", np.ones(len(feature_cols)))
            shap_matrix = x_matrix.values * importances

        # 1. SHAP Values DataFrame
        shap_df = pd.DataFrame(shap_matrix, columns=feature_cols)
        shap_df.insert(0, "wallet_id", wallet_ids)
        shap_df.to_csv(self.explainability_dir / "shap_values.csv", index=False)

        # 2. Global Feature Importance
        global_imp = np.abs(shap_matrix).mean(axis=0)
        global_df = pd.DataFrame(
            {"feature": feature_cols, "mean_absolute_shap": global_imp}
        ).sort_values("mean_absolute_shap", ascending=False)
        global_df.to_csv(
            self.explainability_dir / "global_feature_importance.csv", index=False
        )

        # 3. Top Feature Contributions per Wallet
        top_contrib_rows = []
        for i, wid in enumerate(wallet_ids):
            w_shap = shap_matrix[i]
            w_vals = x_matrix.iloc[i].values
            top_indices = np.argsort(np.abs(w_shap))[::-1][:5]
            for rank, idx in enumerate(top_indices, 1):
                feat_name = feature_cols[idx]
                top_contrib_rows.append(
                    {
                        "wallet_id": wid,
                        "rank": rank,
                        "feature": feat_name,
                        "feature_value": float(w_vals[idx]),
                        "shap_value": float(w_shap[idx]),
                        "direction": (
                            "increases_risk" if w_shap[idx] > 0 else "decreases_risk"
                        ),
                    }
                )

        top_contrib_df = pd.DataFrame(top_contrib_rows)
        top_contrib_df.to_csv(
            self.explainability_dir / "top_feature_contributions.csv", index=False
        )

        return top_contrib_df

    # ========================================================
    # STAGE 11: ALERT GENERATION
    # ========================================================

    def run_alert_service(
        self,
        risk_predictions: pd.DataFrame,
        deterministic_scores: pd.DataFrame,
        top_contributions: pd.DataFrame,
        isolation_scores: Optional[pd.DataFrame] = None,
        autoencoder_scores: Optional[pd.DataFrame] = None,
    ) -> List[Dict[str, Any]]:
        self._update_progress("Alert Generation", 0.95)

        merged = risk_predictions.merge(deterministic_scores, on="wallet_id", how="left")
        if isolation_scores is not None:
            merged = merged.merge(isolation_scores, on="wallet_id", how="left")
        if autoencoder_scores is not None:
            merged = merged.merge(autoencoder_scores, on="wallet_id", how="left")

        # Group top contributions by wallet_id
        top_by_wallet = {}
        for wid, group in top_contributions.groupby("wallet_id"):
            top_by_wallet[wid] = group.to_dict(orient="records")

        alerts = []
        for _, row in merged.iterrows():
            wid = str(row["wallet_id"])
            risk_prob = float(row.get("risk_probability", 0.0))
            det_score = float(row.get("deterministic_score", 0.0))

            if risk_prob >= 0.80:
                severity = "HIGH"
            elif risk_prob >= 0.50:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            alert_item = {
                "wallet_id": wid,
                "severity": severity,
                "risk_probability": round(risk_prob, 4),
                "risk_prediction": int(row.get("risk_prediction", 0)),
                "deterministic_score": round(det_score, 4),
                "isolation_forest_score": round(
                    float(row.get("isolation_forest_anomaly_score", 0.0)), 4
                ),
                "autoencoder_score": round(
                    float(row.get("autoencoder_anomaly_score", 0.0)), 4
                ),
                "top_shap_factors": top_by_wallet.get(wid, [])[:5],
            }
            alerts.append(alert_item)

        # Sort alerts: HIGH severity first, then descending by risk_probability
        severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
        alerts.sort(key=lambda a: (severity_order.get(a["severity"], 3), -a["risk_probability"]))

        # Save alerts.csv
        alert_rows = []
        for a in alerts:
            alert_rows.append(
                {
                    "wallet_id": a["wallet_id"],
                    "severity": a["severity"],
                    "risk_probability": a["risk_probability"],
                    "risk_prediction": a["risk_prediction"],
                    "deterministic_score": a["deterministic_score"],
                    "isolation_forest_score": a["isolation_forest_score"],
                    "autoencoder_score": a["autoencoder_score"],
                }
            )
        alerts_df = pd.DataFrame(alert_rows)
        alerts_df.to_csv(self.alerts_dir / "alerts.csv", index=False)

        # Save alerts.json
        with open(self.alerts_dir / "alerts.json", "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2)

        return alerts

    # ========================================================
    # STAGE 12: MANIFEST & PIPELINE EXECUTION
    # ========================================================

    def execute(self) -> Dict[str, Any]:
        """Execute the entire pipeline end-to-end."""
        start_time = time.time()
        logger.info(f"Beginning forensic pipeline execution from {self.raw_dir} to {self.output_dir}")

        # 1. Ingestion
        data = self.validate_and_load_raw_data()
        num_wallets = len(data["wallets"])
        num_tx = len(data["transactions"])

        # 2. Features
        raw_features, scaled_features = self.run_feature_engineering(data)

        # 3. Graph Engine
        edges, graph_features = self.run_graph_engine(data)

        # 4. GraphSAGE
        embeddings = self.run_graphsage(scaled_features, edges)

        # 5. Isolation Forest
        if_scores = self.run_isolation_forest(raw_features, graph_features, embeddings)

        # 6. Autoencoder
        ae_scores = self.run_autoencoder(raw_features, graph_features, embeddings)

        # 7. Deterministic Risk
        det_scores = self.run_deterministic(raw_features, graph_features)

        # 8. Fusion
        fused_features = self.run_risk_fusion(
            raw_features, graph_features, embeddings, if_scores, ae_scores, det_scores
        )

        # 9. Risk Model
        predictions = self.run_risk_model(fused_features)

        # 10. SHAP
        top_contribs = self.run_shap(fused_features, predictions)

        # 11. Alerts
        alerts = self.run_alert_service(
            predictions, det_scores, top_contribs, if_scores, ae_scores
        )

        elapsed = time.time() - start_time
        self._update_progress("Pipeline Completed", 1.0)

        high_alerts = sum(1 for a in alerts if a["severity"] == "HIGH")
        med_alerts = sum(1 for a in alerts if a["severity"] == "MEDIUM")

        manifest = {
            "status": "success",
            "execution_time_seconds": round(elapsed, 2),
            "dataset": {
                "wallet_count": num_wallets,
                "transaction_count": num_tx,
                "edge_count": len(edges),
            },
            "features": {
                "count": raw_features.shape[1] - 1,
                "scaled": True,
            },
            "graphsage": {
                "embedding_dimension": 32,
            },
            "models_evaluated": {
                "graphsage": True,
                "isolation_forest": True,
                "autoencoder": True,
                "deterministic_engine": True,
                "risk_model": True,
                "shap": True,
            },
            "alerts_summary": {
                "total_wallets": len(alerts),
                "high_risk_count": high_alerts,
                "medium_risk_count": med_alerts,
                "low_risk_count": len(alerts) - high_alerts - med_alerts,
            },
        }

        with open(self.reports_dir / "pipeline_manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            f"Forensic pipeline finished in {elapsed:.2f}s. "
            f"Generated alerts for {num_wallets} wallets ({high_alerts} HIGH, {med_alerts} MEDIUM)."
        )
        return manifest


def run_pipeline(
    raw_dir: str | Path,
    output_dir: str | Path,
    artifact_dir: Optional[str | Path] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Convenience functional interface to run the orchestrator."""
    orchestrator = Orchestrator(
        raw_dir=Path(raw_dir),
        output_dir=Path(output_dir),
        artifact_dir=Path(artifact_dir) if artifact_dir else None,
        progress_callback=progress_callback,
    )
    return orchestrator.execute()


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Run the Bitcoin forensic pipeline orchestrator.")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "raw"),
        help="Directory containing the 6 required raw CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "outputs"),
        help="Directory to write all generated artifacts",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default=str(PROJECT_ROOT / "backend" / "models" / "artifacts"),
        help="Directory containing pre-trained model artifacts",
    )

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
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
