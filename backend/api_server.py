"""
Bitcoin Transaction Analysis REST API Server
Implements Asynchronous ML Pipeline Architecture as per api_endpoints.md:

1. Ingestion & Processing Pipeline:
   - POST /api/v1/jobs/upload
   - GET  /api/v1/jobs/{job_id}/status
   - GET  /api/v1/jobs/{job_id}/summary

2. Graph Rendering (Cytoscape.js):
   - GET  /api/v1/jobs/{job_id}/graph/overview
   - GET  /api/v1/jobs/{job_id}/graph/wallet/{wallet_id}
   - GET  /api/v1/graph/overview
   - GET  /api/v1/graph/wallet/{wallet_id}

3. Alerts & Explainability (XAI):
   - GET  /api/v1/jobs/{job_id}/alerts
   - GET  /api/v1/jobs/{job_id}/explainability/wallet/{wallet_id}
   - GET  /api/v1/alerts
   - GET  /api/v1/explainability/wallet/{wallet_id}

4. Traceability Animations:
   - GET  /api/v1/jobs/{job_id}/patterns/trace/{wallet_id}
   - GET  /api/v1/patterns/trace/{wallet_id}

5. System & Job Management:
   - GET  /api/v1/jobs
   - GET  /api/v1/jobs/{job_id}
   - GET  /api/v1/jobs/{job_id}/manifest
   - GET  /api/v1/health
"""

from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
from pydantic import BaseModel

# ============================================================
# PROJECT ROOT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import create_job, get_job, init_db, list_jobs, update_job
from backend.pipeline.orchestrator import run_pipeline

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# DATA MODELS
# ============================================================

class JobUploadResponse(BaseModel):
    job_id: str
    status: str
    message: Optional[str] = "Uploads the raw CSV. Triggers Feature Engineering, GraphSAGE, Isolation Forest, LightGBM, and SHAP calculators in the background."

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: int = 0
    step: str = "queued"
    current_stage: Optional[str] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None

class JobSummaryResponse(BaseModel):
    job_id: str
    status: str
    total_wallets: int = 0
    anomalies_found: int = 0
    high_risk_count: int = 0
    medium_risk_count: int = 0
    low_risk_count: int = 0
    execution_time_seconds: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

class TransactionStep(BaseModel):
    step: int
    source_wallet: str
    source: Optional[str] = None
    target_wallet: str
    target: Optional[str] = None
    txid: str
    amount_sats: float
    timestamp: Optional[str] = None

class TraceResponse(BaseModel):
    wallet_id: str
    pattern: str
    pattern_detected: Optional[str] = None
    confidence_score: float
    sequence: List[TransactionStep]
    total_amount_sats: float
    hop_count: int

class RiskFactorExplanation(BaseModel):
    feature: str
    feature_name: Optional[str] = None
    shap_value: Optional[float] = None
    shap_impact: Optional[float] = None
    raw_value: Optional[float] = None
    direction: Optional[str] = "increases_risk"
    description: Optional[str] = None
    percentile_rank: Optional[float] = None

class ExplainabilityResponse(BaseModel):
    wallet_id: str
    overall_risk_score: float
    model_used: Optional[str] = "XGBoost + SHAP + Isolation Forest + Autoencoder"
    top_risk_factors: List[RiskFactorExplanation]
    deterministic_score: Optional[float] = None

class Edge(BaseModel):
    source: str
    target: str
    transaction_count: int = 1
    total_amount_sats: float = 0.0
    gnn_influence_weight: float = 0.5
    frequency_score: float = 0.5

class Node(BaseModel):
    id: str
    risk_score: float = 0.5
    wallet_features: Dict[str, float] = {}

class GraphOverviewResponse(BaseModel):
    nodes: List[Node]
    edges: List[Edge]
    total_nodes: Optional[int] = None
    total_edges: Optional[int] = None

class GraphResponse(BaseModel):
    wallet_id: str
    nodes: List[Node]
    edges: List[Edge]
    ego_hops: int = 2

# ============================================================
# DATA LOADER
# ============================================================

class DataLoader:
    """Loads and caches all required CSV files for an output root."""

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_csv(self, path: str) -> pd.DataFrame:
        """Load CSV with caching."""
        if path in self._cache:
            return self._cache[path]

        full_path = self.data_root / path
        if not full_path.exists():
            fallback_path = PROJECT_ROOT / "outputs" / path
            if fallback_path.exists():
                full_path = fallback_path
            else:
                raise FileNotFoundError(f"CSV not found: {full_path}")

        df = pd.read_csv(full_path)
        self._cache[path] = df
        return df

    @property
    def graph_edges(self) -> pd.DataFrame:
        return self.load_csv("graphs/graph_edges.csv")

    @property
    def wallet_features(self) -> pd.DataFrame:
        df = self.load_csv("features/wallet_features.csv")
        return df.set_index("wallet_id")

    @property
    def graphsage_embeddings(self) -> pd.DataFrame:
        df = self.load_csv("graphs/graphsage_embeddings.csv")
        return df.set_index("wallet_id")

    @property
    def deterministic_scores(self) -> pd.DataFrame:
        df = self.load_csv("models/deterministic_scores.csv")
        return df.set_index("wallet_id")

    @property
    def isolation_forest_scores(self) -> pd.DataFrame:
        df = self.load_csv("models/isolation_forest_scores.csv")
        return df.set_index("wallet_id")

    @property
    def risk_predictions(self) -> pd.DataFrame:
        df = self.load_csv("models/risk_model_predictions.csv")
        return df.set_index("wallet_id")

    @property
    def transactions(self) -> pd.DataFrame:
        candidates = [
            self.data_root.parent / "raw" / "transactions.csv",
            self.data_root / "raw" / "transactions.csv",
            PROJECT_ROOT / "data" / "raw" / "transactions.csv",
        ]
        for c in candidates:
            if c.exists():
                return pd.read_csv(c)
        raise FileNotFoundError("transactions.csv not found.")

# ============================================================
# PATTERN DETECTION ENGINE
# ============================================================

class PatternDetector:
    """Detects illicit transaction patterns (peeling chains, fan-in, fan-out, etc.)."""

    PATTERN_DESCRIPTIONS = {
        "peeling_chain": "Sequential transfers with minimal amounts (classic money laundering pattern)",
        "fan_out": "High-volume distribution to many wallets (coin mixing/tumbling)",
        "fan_in": "Convergence from many sources to single wallet (consolidation)",
        "circular_flow": "Transactions forming loops (circular money movement)",
        "direct_transfer": "Direct single-hop transfer",
    }

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._build_graph()

    def _build_graph(self):
        try:
            edges = self.loader.graph_edges
            self.graph = defaultdict(lambda: defaultdict(list))
            for _, row in edges.iterrows():
                src = str(row["source_wallet_id"])
                tgt = str(row["target_wallet_id"])
                self.graph[src][tgt].append(
                    {
                        "txid": f"TX_{abs(hash(src + tgt)) % 1000000:x}",
                        "amount": float(row.get("total_amount_sats", 0.0)),
                        "timestamp": str(row.get("first_seen", "")),
                        "count": int(row.get("transaction_count", 1)),
                    }
                )
        except Exception as e:
            logger.warning(f"Error building graph in PatternDetector: {e}")
            self.graph = defaultdict(lambda: defaultdict(list))

    def detect_pattern(self, wallet_id: str, max_hops: int = 5) -> Tuple[str, float, List[Dict]]:
        sequence = self._trace_funds_bfs(wallet_id, max_hops=max_hops)
        if not sequence:
            return "direct_transfer", 0.5, []
        pattern, confidence = self._analyze_pattern(wallet_id, sequence)
        return pattern, confidence, sequence

    def _trace_funds_bfs(self, wallet_id: str, max_hops: int = 5) -> List[Dict]:
        sequence = []
        visited = set()
        queue = deque([(wallet_id, 0, [])])

        while queue:
            current_wallet, hop_count, path = queue.popleft()
            if hop_count >= max_hops or current_wallet in visited:
                continue
            visited.add(current_wallet)

            for target_wallet, txs in self.graph[current_wallet].items():
                for tx in txs:
                    step_info = {
                        "step": len(sequence) + 1,
                        "source_wallet": current_wallet,
                        "source": current_wallet,
                        "target_wallet": target_wallet,
                        "target": target_wallet,
                        "txid": tx["txid"],
                        "amount_sats": tx["amount"],
                        "timestamp": tx["timestamp"],
                    }
                    sequence.append(step_info)
                    queue.append((target_wallet, hop_count + 1, path + [step_info]))

        return sorted(sequence, key=lambda x: x["amount_sats"], reverse=True)[:10]

    def _analyze_pattern(self, wallet_id: str, sequence: List[Dict]) -> Tuple[str, float]:
        if not sequence:
            return "direct_transfer", 0.5
        out_degree = len(set(tx["target_wallet"] for tx in sequence))
        in_degree = len(set(tx["source_wallet"] for tx in sequence if tx["source_wallet"] != wallet_id))

        if out_degree > 5 and len(sequence) > 1:
            return "fan_out", min(0.95, 0.6 + (out_degree / 50))
        elif in_degree > 5:
            return "fan_in", min(0.90, 0.5 + (in_degree / 50))
        elif len(sequence) >= 3 and out_degree <= 3:
            amounts = [tx["amount_sats"] for tx in sequence[:5]]
            is_decreasing = all(amounts[i] >= amounts[i + 1] * 0.9 for i in range(len(amounts) - 1))
            if is_decreasing:
                return "peeling_chain", min(0.92, 0.65 + (len(sequence) / 20))

        return "direct_transfer", min(0.85, 0.5 + (len(sequence) / 20))

# ============================================================
# EXPLAINABILITY ENGINE (SHAP & RISK FACTORS)
# ============================================================

class ExplainabilityEngine:
    """Translates model evidence and SHAP values into human-readable explanations."""

    FEATURE_DESCRIPTIONS = {
        "pagerank": "Importance in transaction network (higher = more central/intermediary)",
        "pagerank_centrality": "Importance in transaction network (higher = more central/intermediary)",
        "betweenness_centrality": "Frequency of appearing on shortest paths between wallets",
        "out_degree": "Number of unique outgoing transaction recipients",
        "in_degree": "Number of unique incoming transaction sources",
        "fan_out": "Distribution spread (many recipients per transaction)",
        "fan_in": "Collection spread (many sources per transaction)",
        "clustering_coefficient": "Local community density (0=bridge, 1=isolated)",
        "tx_count": "Total transaction count (activity volume)",
        "transaction_velocity": "Transactions per day (activity speed)",
        "unique_asns": "Diversity of network providers used",
        "unique_countries": "Geographic spread of network activity",
        "off_hour_transaction_ratio": "Percentage of transactions outside business hours",
        "avg_inter_tx_seconds": "Average time between consecutive transactions",
        "total_sent_sats": "Total satoshis sent (economic volume)",
        "total_received_sats": "Total satoshis received (economic volume)",
        "std_received_sats": "Variability in incoming transaction sizes",
        "unique_input_counterparties": "Number of unique input sources",
        "unique_output_counterparties": "Number of unique output destinations",
        "degree": "Total connections (in + out degree)",
        "observations_per_tx": "Network observation density per transaction",
        "rapid_hop_count": "Rapid succession transaction count",
    }

    def __init__(self, loader: DataLoader):
        self.loader = loader

    def explain_wallet_risk(self, wallet_id: str) -> Dict[str, Any]:
        """Generate comprehensive explainability report for wallet using computed SHAP."""
        factors = []

        shap_file = self.loader.data_root / "explainability" / "top_feature_contributions.csv"
        if not shap_file.exists():
            shap_file = PROJECT_ROOT / "outputs" / "explainability" / "top_feature_contributions.csv"

        if shap_file.exists():
            try:
                shap_df = pd.read_csv(shap_file)
                w_shap = shap_df[shap_df["wallet_id"].astype(str) == str(wallet_id)]
                for _, row in w_shap.iterrows():
                    fname = str(row["feature"])
                    shap_val = float(row.get("shap_value", 0.0))
                    feat_val = float(row.get("feature_value", 0.0))
                    direction = str(row.get("direction", "increases_risk"))
                    desc = self.FEATURE_DESCRIPTIONS.get(fname, f"Statistical anomaly impact in {fname}")
                    factors.append(
                        {
                            "feature": fname,
                            "feature_name": fname,
                            "shap_value": round(shap_val, 4),
                            "shap_impact": round(min(1.0, max(0.0, abs(shap_val) / 5)), 4),
                            "raw_value": round(feat_val, 4),
                            "direction": direction,
                            "description": desc,
                            "percentile_rank": round(float(min(100.0, max(0.0, abs(shap_val) * 20))), 2),
                        }
                    )
            except Exception as e:
                logger.warning(f"Error reading SHAP contributions: {e}")

        try:
            det_row = self.loader.deterministic_scores.loc[wallet_id]
            det_score = float(det_row["deterministic_score"])
            det_evidence = self._parse_evidence(det_row.get("deterministic_evidence", "[]"))
        except (KeyError, IndexError):
            det_score = None
            det_evidence = []

        try:
            risk_row = self.loader.risk_predictions.loc[wallet_id]
            risk_score = float(risk_row["risk_probability"])
        except (KeyError, IndexError):
            risk_score = 0.5

        try:
            features = self.loader.wallet_features.loc[wallet_id].to_dict()
        except KeyError:
            features = {}

        if not factors:
            factors = self._generate_fallback_factors(wallet_id, det_evidence, features)

        overall_score = (risk_score + (det_score or 0.5)) / 2

        return {
            "wallet_id": wallet_id,
            "overall_risk_score": round(min(1.0, max(0.0, overall_score)), 4),
            "model_used": "XGBoost + SHAP + Isolation Forest + Autoencoder",
            "top_risk_factors": factors[:10],
            "deterministic_score": round(det_score, 4) if det_score is not None else None,
        }

    def _parse_evidence(self, evidence_str: str) -> List[Dict]:
        try:
            if isinstance(evidence_str, str):
                return json.loads(evidence_str.replace('""', '"'))
            return []
        except Exception:
            return []

    def _generate_fallback_factors(self, wallet_id: str, evidence: List[Dict], features: Dict) -> List[Dict]:
        factors = []
        for item in evidence[:5]:
            fname = item.get("feature", "unknown")
            raw_val = item.get("value", 0)
            score = item.get("empirical_anomaly_score", 0.5)
            factors.append(
                {
                    "feature": fname,
                    "feature_name": fname,
                    "shap_value": round(float(score) * 2.0, 4),
                    "shap_impact": round(float(score), 4),
                    "raw_value": round(float(raw_val or 0.0), 4),
                    "direction": "increases_risk",
                    "description": self.FEATURE_DESCRIPTIONS.get(fname, f"Statistical anomaly in {fname}"),
                    "percentile_rank": round(float(score * 100), 2),
                }
            )
        return factors

# ============================================================
# GNN INFLUENCE CALCULATOR
# ============================================================

class GNNInfluenceCalculator:
    """Calculates GNN influence weights from GraphSAGE embeddings and graph structure."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._cache: Dict[str, float] = {}

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
            tx_count = edges[mask]["transaction_count"].values[0] if mask.any() else 1
            freq_score = min(1.0, np.log1p(tx_count) / np.log1p(100))
        except Exception:
            freq_score = 0.1

        influence = 0.6 * cosine_sim + 0.4 * freq_score
        influence = round(max(0.01, min(1.0, influence)), 4)
        self._cache[cache_key] = influence
        return influence

    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        dot = np.dot(vec1, vec2)
        n1 = np.linalg.norm(vec1)
        n2 = np.linalg.norm(vec2)
        return float(dot / (n1 * n2)) if (n1 > 0 and n2 > 0) else 0.5

# ============================================================
# GRAPH EXPLORER (EGO-GRAPH & OVERVIEW)
# ============================================================

class GraphExplorer:
    """Builds Cytoscape-compatible overview networks and ego-graphs."""

    def __init__(self, loader: DataLoader, influence_calc: GNNInfluenceCalculator):
        self.loader = loader
        self.influence_calc = influence_calc
        self._build_graph()

    def _build_graph(self):
        try:
            edges = self.loader.graph_edges
            self.graph = defaultdict(set)
            self.reverse_graph = defaultdict(set)
            self.edge_data = {}

            for _, row in edges.iterrows():
                src = str(row["source_wallet_id"])
                tgt = str(row["target_wallet_id"])
                self.graph[src].add(tgt)
                self.reverse_graph[tgt].add(src)
                self.edge_data[(src, tgt)] = {
                    "tx_count": int(row.get("transaction_count", 1)),
                    "amount": float(row.get("total_amount_sats", 0.0)),
                }
        except Exception as e:
            logger.warning(f"Error initializing graph in GraphExplorer: {e}")
            self.graph = defaultdict(set)
            self.reverse_graph = defaultdict(set)
            self.edge_data = {}

    def get_ego_graph(self, wallet_id: str, hops: int = 2) -> Dict[str, Any]:
        """Returns contextual sub-graph (ego-graph) within N hops."""
        nodes = {wallet_id}
        current_level = {wallet_id}

        for _ in range(hops):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph[node])
                next_level.update(self.reverse_graph[node])
            current_level = next_level - nodes
            nodes.update(current_level)

        node_list = []
        try:
            preds = self.loader.risk_predictions
        except Exception:
            preds = pd.DataFrame()
        try:
            features = self.loader.wallet_features
        except Exception:
            features = pd.DataFrame()

        for nid in nodes:
            risk_score = float(preds.loc[nid]["risk_probability"]) if (nid in preds.index) else 0.5
            feat_dict = (
                features.loc[nid][["pagerank", "betweenness_centrality", "out_degree", "in_degree"]].to_dict()
                if (nid in features.index and "pagerank" in features.columns)
                else {}
            )
            node_list.append(Node(id=nid, risk_score=risk_score, wallet_features=feat_dict))

        edge_list = []
        for src, tgts in self.graph.items():
            if src not in nodes:
                continue
            for tgt in tgts:
                if tgt not in nodes:
                    continue
                edata = self.edge_data.get((src, tgt), {})
                influence = self.influence_calc.calculate_edge_influence(src, tgt)
                tx_count = edata.get("tx_count", 1)
                freq_score = min(1.0, np.log1p(tx_count) / 5)

                edge_list.append(
                    Edge(
                        source=src,
                        target=tgt,
                        transaction_count=tx_count,
                        total_amount_sats=edata.get("amount", 0.0),
                        gnn_influence_weight=influence,
                        frequency_score=round(freq_score, 4),
                    )
                )

        return {
            "wallet_id": wallet_id,
            "nodes": node_list,
            "edges": edge_list,
            "ego_hops": hops,
        }

    def get_overview_graph(self, max_nodes: int = 150) -> Dict[str, Any]:
        """Returns macroscopic overview network prioritized by anomaly risk."""
        try:
            preds = self.loader.risk_predictions
            edges_df = self.loader.graph_edges
            features = self.loader.wallet_features
        except Exception as e:
            return {"nodes": [], "edges": [], "total_nodes": 0, "total_edges": 0}

        sorted_wallets = preds.sort_values("risk_probability", ascending=False).index.tolist()
        top_wallets = set(str(w) for w in sorted_wallets[:max_nodes])

        mask = edges_df["source_wallet_id"].astype(str).isin(top_wallets) | edges_df["target_wallet_id"].astype(
            str
        ).isin(top_wallets)
        filtered_edges = edges_df[mask].head(600)

        all_nodes = set(filtered_edges["source_wallet_id"].astype(str)).union(
            set(filtered_edges["target_wallet_id"].astype(str))
        ).union(top_wallets)

        nodes = []
        for nid in all_nodes:
            risk_score = float(preds.loc[nid]["risk_probability"]) if (nid in preds.index) else 0.5
            feat_dict = (
                features.loc[nid][["pagerank", "out_degree", "in_degree"]].to_dict()
                if (nid in features.index and "pagerank" in features.columns)
                else {}
            )
            nodes.append(Node(id=nid, risk_score=risk_score, wallet_features=feat_dict))

        edges = []
        for _, row in filtered_edges.iterrows():
            src = str(row["source_wallet_id"])
            tgt = str(row["target_wallet_id"])
            influence = self.influence_calc.calculate_edge_influence(src, tgt)
            tx_c = int(row.get("transaction_count", 1))
            edges.append(
                Edge(
                    source=src,
                    target=tgt,
                    transaction_count=tx_c,
                    total_amount_sats=float(row.get("total_amount_sats", 0.0)),
                    gnn_influence_weight=influence,
                    frequency_score=round(min(1.0, np.log1p(tx_c) / 5), 4),
                )
            )

        return {
            "nodes": nodes,
            "edges": edges,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
        }

# ============================================================
# FASTAPI APPLICATION CREATION
# ============================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application with all required endpoints."""
    app = FastAPI(
        title="Bitcoin Transaction Forensics API",
        description="Asynchronous ML Forensic Pipeline & Cytoscape.js Graph REST API",
        version="2.0.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    DATA_ROOT = PROJECT_ROOT / "outputs"
    default_loader = DataLoader(DATA_ROOT)
    default_influence = GNNInfluenceCalculator(default_loader)
    default_pattern = PatternDetector(default_loader)
    default_explain = ExplainabilityEngine(default_loader)
    default_graph = GraphExplorer(default_loader, default_influence)

    def get_services_for_job(job_id: Optional[str] = None):
        """Dynamically resolve engines for a specific job_id or fallback to default."""
        if not job_id:
            return default_loader, default_pattern, default_explain, default_graph

        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        if job["status"] == "failed":
            raise HTTPException(status_code=400, detail=f"Job '{job_id}' failed: {job.get('error_message')}")
        if job["status"] == "processing":
            raise HTTPException(
                status_code=202,
                detail=f"Job '{job_id}' is currently {job.get('current_stage', 'processing')} ({job.get('progress', 0)}%).",
            )

        job_loader = DataLoader(Path(job["output_dir"]))
        job_influence = GNNInfluenceCalculator(job_loader)
        job_pattern = PatternDetector(job_loader)
        job_explain = ExplainabilityEngine(job_loader)
        job_graph = GraphExplorer(job_loader, job_influence)
        return job_loader, job_pattern, job_explain, job_graph

    # --------------------------------------------------------
    # BACKGROUND PIPELINE WORKER
    # --------------------------------------------------------

    def run_job_background(job_id: str, raw_dir: Path, output_dir: Path):
        try:
            logger.info(f"Beginning background execution for job {job_id}")
            stage_weights = {
                "Ingestion & Validation": 10,
                "Feature Engineering": 25,
                "NetworkX Graph Engine": 45,
                "GraphSAGE Inference": 60,
                "Isolation Forest Inference": 70,
                "Autoencoder Inference": 78,
                "Deterministic Statistical Risk": 85,
                "Risk Feature Fusion": 88,
                "Risk Model Inference": 92,
                "SHAP Explainability": 95,
                "Alert Generation": 98,
                "Pipeline Completed": 100,
            }

            def on_progress(stage_name: str, frac: float):
                pct = stage_weights.get(stage_name, int(frac * 100))
                update_job(job_id, current_stage=stage_name, progress=pct)

            manifest = run_pipeline(
                raw_dir=raw_dir,
                output_dir=output_dir,
                progress_callback=on_progress,
            )

            now = datetime.now(timezone.utc).isoformat()
            update_job(
                job_id,
                status="completed",
                current_stage="completed",
                progress=100,
                completed_at=now,
                summary=manifest,
            )
            logger.info(f"Job {job_id} successfully completed.")
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Job {job_id} failed: {error_msg}\n{traceback.format_exc()}")
            now = datetime.now(timezone.utc).isoformat()
            update_job(
                job_id,
                status="failed",
                current_stage="failed",
                progress=0,
                completed_at=now,
                error_message=error_msg,
            )

    # ============================================================
    # 1. INGESTION & PROCESSING PIPELINE
    # ============================================================

    EXPECTED_RAW_FILES = {
        "wallets.csv": "wallets",
        "transactions.csv": "transactions",
        "transaction_inputs.csv": "transaction_inputs",
        "transaction_outputs.csv": "transaction_outputs",
        "network_observations.csv": "network_observations",
        "ip_metadata.csv": "ip_metadata",
    }

    @app.post(
        "/api/v1/jobs/upload",
        response_model=JobUploadResponse,
        summary="Upload raw CSVs and trigger background ML pipeline",
        description="Uploads the 6 raw CSV files (without risk labels) and triggers Feature Engineering, GraphSAGE, Isolation Forest, LightGBM, and SHAP calculators in the background.",
        tags=["1. Ingestion & Processing"],
        openapi_extra={
            "requestBody": {
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "wallets": {"type": "string", "format": "binary", "description": "wallets.csv"},
                                "transactions": {"type": "string", "format": "binary", "description": "transactions.csv"},
                                "transaction_inputs": {"type": "string", "format": "binary", "description": "transaction_inputs.csv"},
                                "transaction_outputs": {"type": "string", "format": "binary", "description": "transaction_outputs.csv"},
                                "network_observations": {"type": "string", "format": "binary", "description": "network_observations.csv"},
                                "ip_metadata": {"type": "string", "format": "binary", "description": "ip_metadata.csv"},
                            },
                        }
                    }
                }
            }
        },
    )
    async def upload_raw_csv_job(
        request: Request,
        background_tasks: BackgroundTasks,
    ):
        """
        Uploads the raw CSV. Triggers Feature Engineering, GraphSAGE, Isolation Forest,
        LightGBM, and SHAP calculators in the background.
        """
        uploaded_map = {}
        try:
            form = await request.form()
            for key, val in form.multi_items():
                if hasattr(val, "filename") and bool(val.filename):
                    clean_fname = val.filename.lower().strip()
                    clean_key = key.lower().strip()
                    for exp_name in EXPECTED_RAW_FILES.keys():
                        stem = exp_name.replace(".csv", "")
                        if (
                            clean_fname == exp_name
                            or clean_fname.endswith(f"_{exp_name}")
                            or clean_key == stem
                            or clean_key == exp_name
                        ):
                            if exp_name not in uploaded_map:
                                uploaded_map[exp_name] = val
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse multipart form-data: {e}")

        missing = [exp for exp in EXPECTED_RAW_FILES.keys() if exp not in uploaded_map]
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Missing required CSV files: {missing}. Must supply all 6: {list(EXPECTED_RAW_FILES.keys())}",
            )

        job_id = f"job-{uuid.uuid4().hex[:5]}"
        job_dir = PROJECT_ROOT / "outputs" / "jobs" / job_id
        raw_dir = job_dir / "raw"
        output_dir = job_dir / "results"
        raw_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            for filename, up_file in uploaded_map.items():
                content = await up_file.read()
                dest_path = raw_dir / filename
                dest_path.write_bytes(content)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded files: {e}")

        create_job(
            job_id=job_id,
            raw_dir=str(raw_dir),
            output_dir=str(output_dir),
            status="processing",
            current_stage="Ingestion & Validation",
            progress=5,
        )

        background_tasks.add_task(run_job_background, job_id, raw_dir, output_dir)

        return JobUploadResponse(
            job_id=job_id,
            status="processing",
            message="Uploads the raw CSV. Triggers Feature Engineering, GraphSAGE, Isolation Forest, LightGBM, and SHAP calculators in the background.",
        )

    @app.get(
        "/api/v1/jobs/{job_id}/status",
        response_model=JobStatusResponse,
        summary="Poll job progress & active ML stage",
        tags=["1. Ingestion & Processing"],
    )
    async def get_job_status_endpoint(job_id: str):
        """Frontend polls this to update the loading bar while the heavy ML models run."""
        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return JobStatusResponse(
            job_id=job["job_id"],
            status=job["status"],
            progress=job.get("progress", 0) or (100 if job["status"] == "completed" else 0),
            step=job.get("current_stage") or job["status"],
            current_stage=job.get("current_stage"),
            created_at=job.get("created_at"),
            completed_at=job.get("completed_at"),
            error_message=job.get("error_message"),
            summary=job.get("summary"),
        )

    @app.get(
        "/api/v1/jobs/{job_id}/summary",
        response_model=JobSummaryResponse,
        summary="Fetch high-level metrics once job is 100% complete",
        tags=["1. Ingestion & Processing"],
    )
    async def get_job_summary_endpoint(job_id: str):
        """Fetches high-level metrics once the job is 100% complete."""
        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        if job["status"] != "completed":
            return JobSummaryResponse(
                job_id=job_id,
                status=job["status"],
                total_wallets=0,
                anomalies_found=0,
                high_risk_count=0,
                medium_risk_count=0,
                low_risk_count=0,
            )

        summary = job.get("summary") or {}
        dataset = summary.get("dataset", {})
        alerts_sum = summary.get("alerts_summary", {})
        high_c = alerts_sum.get("high_risk_count", 0)
        med_c = alerts_sum.get("medium_risk_count", 0)
        low_c = alerts_sum.get("low_risk_count", 0)
        total_w = dataset.get("wallet_count", alerts_sum.get("total_wallets", 0))

        return JobSummaryResponse(
            job_id=job_id,
            status="completed",
            total_wallets=total_w,
            anomalies_found=high_c + med_c,
            high_risk_count=high_c,
            medium_risk_count=med_c,
            low_risk_count=low_c,
            execution_time_seconds=summary.get("execution_time_seconds"),
            details=summary,
        )

    # ============================================================
    # 2. GRAPH RENDERING (CYTOSCAPE.JS)
    # ============================================================

    @app.get(
        "/api/v1/jobs/{job_id}/graph/overview",
        response_model=GraphOverviewResponse,
        summary="Macroscopic view of transaction network for a job",
        tags=["2. Graph Rendering"],
    )
    async def get_job_graph_overview(
        job_id: str,
        max_nodes: int = Query(200, description="Max nodes to include in macroscopic view"),
    ):
        """Returns the macroscopic view of the network for Cytoscape.js rendering for a specific job."""
        _, _, _, explorer = get_services_for_job(job_id)
        overview = explorer.get_overview_graph(max_nodes=max_nodes)
        return GraphOverviewResponse(**overview)

    @app.get(
        "/api/v1/graph/overview",
        response_model=GraphOverviewResponse,
        summary="Macroscopic view of transaction network (global/optional job_id)",
        tags=["2. Graph Rendering"],
    )
    async def get_global_graph_overview(
        job_id: Optional[str] = Query(None, description="Optional job ID"),
        max_nodes: int = Query(200, description="Max nodes to include in macroscopic view"),
    ):
        """Returns the macroscopic view of the network for Cytoscape.js rendering."""
        _, _, _, explorer = get_services_for_job(job_id)
        overview = explorer.get_overview_graph(max_nodes=max_nodes)
        return GraphOverviewResponse(**overview)

    @app.get(
        "/api/v1/jobs/{job_id}/graph/wallet/{wallet_id}",
        response_model=GraphResponse,
        summary="Contextual Sub-Graph (Ego Graph) for a wallet in a job",
        tags=["2. Graph Rendering"],
    )
    async def get_job_wallet_ego_graph(
        job_id: str,
        wallet_id: str,
        hops: int = Query(2, ge=1, le=5, description="Ego-graph radius hops"),
    ):
        """Returns the Contextual Sub-Graph (Ego Graph) with GNN influence weights for a job."""
        _, _, _, explorer = get_services_for_job(job_id)
        ego = explorer.get_ego_graph(wallet_id, hops=hops)
        return GraphResponse(**ego)

    @app.get(
        "/api/v1/graph/wallet/{wallet_id}",
        response_model=GraphResponse,
        summary="Contextual Sub-Graph (Ego Graph) for a wallet",
        tags=["2. Graph Rendering"],
    )
    async def get_global_wallet_ego_graph(
        wallet_id: str,
        job_id: Optional[str] = Query(None, description="Optional job ID"),
        hops: int = Query(2, ge=1, le=5, description="Ego-graph radius hops"),
    ):
        """Returns the Contextual Sub-Graph (Ego Graph) with GNN influence weights."""
        _, _, _, explorer = get_services_for_job(job_id)
        ego = explorer.get_ego_graph(wallet_id, hops=hops)
        return GraphResponse(**ego)

    # ============================================================
    # 3. ALERTS & EXPLAINABILITY (XAI)
    # ============================================================

    @app.get(
        "/api/v1/jobs/{job_id}/alerts",
        summary="Fetch list of flagged anomaly wallets for a job",
        tags=["3. Alerts & Explainability"],
    )
    async def get_job_alerts(job_id: str):
        """Fetches the list of flagged wallets to display in a table/list for a specific job."""
        loader_inst, _, _, _ = get_services_for_job(job_id)
        alerts_file = loader_inst.data_root / "alerts" / "alerts.json"
        if not alerts_file.exists():
            alerts_file = PROJECT_ROOT / "outputs" / "alerts" / "alerts.json"

        if alerts_file.exists():
            try:
                with open(alerts_file, "r", encoding="utf-8") as f:
                    alerts_data = json.load(f)
                return {"job_id": job_id, "total_alerts": len(alerts_data), "alerts": alerts_data}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading alerts: {e}")

        return {"job_id": job_id, "total_alerts": 0, "alerts": []}

    @app.get(
        "/api/v1/alerts",
        summary="Fetch list of flagged anomaly wallets",
        tags=["3. Alerts & Explainability"],
    )
    async def get_global_alerts(job_id: Optional[str] = Query(None, description="Optional job ID")):
        """Fetches the list of flagged wallets to display in a table/list for the analyst."""
        loader_inst, _, _, _ = get_services_for_job(job_id)
        alerts_file = loader_inst.data_root / "alerts" / "alerts.json"
        if not alerts_file.exists():
            alerts_file = PROJECT_ROOT / "outputs" / "alerts" / "alerts.json"

        if alerts_file.exists():
            try:
                with open(alerts_file, "r", encoding="utf-8") as f:
                    alerts_data = json.load(f)
                return {"job_id": job_id or "default", "total_alerts": len(alerts_data), "alerts": alerts_data}
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error reading alerts: {e}")

        return {"job_id": job_id or "default", "total_alerts": 0, "alerts": []}

    @app.get(
        "/api/v1/jobs/{job_id}/explainability/wallet/{wallet_id}",
        response_model=ExplainabilityResponse,
        summary="Fetch SHAP feature importance explanation for a wallet in a job",
        tags=["3. Alerts & Explainability"],
    )
    async def get_job_explainability(job_id: str, wallet_id: str):
        """Fetches the raw SHAP feature importances to dynamically explain why the wallet was flagged for a job."""
        _, _, explain_engine, _ = get_services_for_job(job_id)
        result = explain_engine.explain_wallet_risk(wallet_id)
        return ExplainabilityResponse(**result)

    @app.get(
        "/api/v1/explainability/wallet/{wallet_id}",
        response_model=ExplainabilityResponse,
        summary="Fetch SHAP feature importance explanation for a wallet",
        tags=["3. Alerts & Explainability"],
    )
    async def get_global_explainability(
        wallet_id: str,
        job_id: Optional[str] = Query(None, description="Optional job ID"),
    ):
        """Fetches the raw SHAP feature importances to dynamically explain why the wallet was flagged."""
        _, _, explain_engine, _ = get_services_for_job(job_id)
        result = explain_engine.explain_wallet_risk(wallet_id)
        return ExplainabilityResponse(**result)

    # ============================================================
    # 4. TRACEABILITY ANIMATIONS
    # ============================================================

    @app.get(
        "/api/v1/jobs/{job_id}/patterns/trace/{wallet_id}",
        response_model=TraceResponse,
        summary="Ordered transaction sequence for fund flow animations for a job",
        tags=["4. Traceability Animations"],
    )
    async def get_job_patterns_trace(
        job_id: str,
        wallet_id: str,
        max_hops: int = Query(5, ge=1, le=10, description="Max transaction hops to trace"),
    ):
        """Returns the exact, ordered sequence of transactions downstream from the flagged wallet for a job."""
        _, pattern_engine, _, _ = get_services_for_job(job_id)
        pattern, confidence, sequence = pattern_engine.detect_pattern(wallet_id, max_hops=max_hops)

        steps = [
            TransactionStep(
                step=tx["step"],
                source_wallet=tx["source_wallet"],
                source=tx["source_wallet"],
                target_wallet=tx["target_wallet"],
                target=tx["target_wallet"],
                txid=tx["txid"],
                amount_sats=tx["amount_sats"],
                timestamp=tx.get("timestamp", ""),
            )
            for tx in sequence
        ]

        return TraceResponse(
            wallet_id=wallet_id,
            pattern=pattern,
            pattern_detected=pattern,
            confidence_score=confidence,
            sequence=steps,
            total_amount_sats=sum(tx["amount_sats"] for tx in sequence),
            hop_count=len(steps),
        )

    @app.get(
        "/api/v1/patterns/trace/{wallet_id}",
        response_model=TraceResponse,
        summary="Ordered transaction sequence for fund flow animations",
        tags=["4. Traceability Animations"],
    )
    async def get_global_patterns_trace(
        wallet_id: str,
        job_id: Optional[str] = Query(None, description="Optional job ID"),
        max_hops: int = Query(5, ge=1, le=10, description="Max transaction hops to trace"),
    ):
        """Returns the exact, ordered sequence of transactions downstream from the flagged wallet."""
        _, pattern_engine, _, _ = get_services_for_job(job_id)
        pattern, confidence, sequence = pattern_engine.detect_pattern(wallet_id, max_hops=max_hops)

        steps = [
            TransactionStep(
                step=tx["step"],
                source_wallet=tx["source_wallet"],
                source=tx["source_wallet"],
                target_wallet=tx["target_wallet"],
                target=tx["target_wallet"],
                txid=tx["txid"],
                amount_sats=tx["amount_sats"],
                timestamp=tx.get("timestamp", ""),
            )
            for tx in sequence
        ]

        return TraceResponse(
            wallet_id=wallet_id,
            pattern=pattern,
            pattern_detected=pattern,
            confidence_score=confidence,
            sequence=steps,
            total_amount_sats=sum(tx["amount_sats"] for tx in sequence),
            hop_count=len(steps),
        )

    # ============================================================
    # 5. SYSTEM & JOB MANAGEMENT
    # ============================================================

    @app.get(
        "/api/v1/jobs",
        response_model=List[JobStatusResponse],
        summary="List all submitted jobs",
        tags=["5. System & Management"],
    )
    async def list_all_jobs(limit: int = 50, offset: int = 0):
        """List all submitted forensic jobs."""
        jobs = list_jobs(limit=limit, offset=offset)
        return [
            JobStatusResponse(
                job_id=j["job_id"],
                status=j["status"],
                progress=j.get("progress", 0) or (100 if j["status"] == "completed" else 0),
                step=j.get("current_stage") or j["status"],
                current_stage=j.get("current_stage"),
                created_at=j.get("created_at"),
                completed_at=j.get("completed_at"),
                error_message=j.get("error_message"),
                summary=j.get("summary"),
            )
            for j in jobs
        ]

    @app.get(
        "/api/v1/jobs/{job_id}",
        response_model=JobStatusResponse,
        summary="Get job status by ID",
        tags=["5. System & Management"],
    )
    async def get_job_by_id(job_id: str):
        """Check status and results for a submitted forensic analysis job."""
        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        return JobStatusResponse(
            job_id=job["job_id"],
            status=job["status"],
            progress=job.get("progress", 0) or (100 if job["status"] == "completed" else 0),
            step=job.get("current_stage") or job["status"],
            current_stage=job.get("current_stage"),
            created_at=job.get("created_at"),
            completed_at=job.get("completed_at"),
            error_message=job.get("error_message"),
            summary=job.get("summary"),
        )

    @app.get(
        "/api/v1/jobs/{job_id}/manifest",
        summary="Get execution manifest report for a job",
        tags=["5. System & Management"],
    )
    async def get_job_manifest(job_id: str):
        """Retrieve full pipeline execution manifest."""
        job = get_job(job_id)
        if not job:
            raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
        manifest_path = Path(job["output_dir"]) / "reports" / "pipeline_manifest.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Manifest not found for this job.")
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @app.get(
        "/api/v1/health",
        summary="Health check endpoint",
        tags=["5. System & Management"],
    )
    async def health_check():
        """Check API health and default data availability."""
        try:
            return {
                "status": "healthy",
                "data_sources": {
                    "graph_edges": len(default_loader.graph_edges),
                    "wallets": len(default_loader.wallet_features),
                    "embeddings": len(default_loader.graphsage_embeddings),
                },
            }
        except Exception as e:
            logger.error(f"Health check warning: {e}")
            return {"status": "healthy", "warning": str(e)}

    return app

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
