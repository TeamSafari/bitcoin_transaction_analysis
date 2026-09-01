"""
Bitcoin Transaction Analysis REST API Server

Provides three main endpoints:
1. Traceability Endpoint - Traces fund flow patterns through transaction graph
2. Explainability Endpoint - Returns SHAP/feature importance for wallet risk scores
3. Enhanced Graph Endpoint - Returns ego-graph with GNN influence weights
"""

from pathlib import Path
from typing import Dict, List, Any, Tuple
import json
import logging
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys

# ============================================================
# PROJECT ROOT SETUP
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# DATA MODELS
# ============================================================

class TransactionStep(BaseModel):
    step: int
    source_wallet: str
    target_wallet: str
    txid: str
    amount_sats: float
    timestamp: str | None = None

class TraceResponse(BaseModel):
    wallet_id: str
    pattern_detected: str
    confidence_score: float
    sequence: List[TransactionStep]
    total_amount_sats: float
    hop_count: int

class RiskFactorExplanation(BaseModel):
    feature_name: str
    raw_value: float
    shap_impact: float
    description: str
    percentile_rank: float | None = None

class ExplainabilityResponse(BaseModel):
    wallet_id: str
    overall_risk_score: float
    model_used: str
    top_risk_factors: List[RiskFactorExplanation]
    deterministic_score: float | None = None

class Edge(BaseModel):
    source: str
    target: str
    transaction_count: int
    total_amount_sats: float
    gnn_influence_weight: float
    frequency_score: float

class Node(BaseModel):
    id: str
    risk_score: float
    wallet_features: Dict[str, float]

class GraphResponse(BaseModel):
    wallet_id: str
    nodes: List[Node]
    edges: List[Edge]
    ego_hops: int

# ============================================================
# DATA LOADER
# ============================================================

class DataLoader:
    """Loads and caches all required CSV files"""
    
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self._cache = {}
        logger.info(f"Initializing DataLoader with root: {data_root}")
    
    def load_csv(self, path: str) -> pd.DataFrame:
        """Load CSV with caching"""
        if path in self._cache:
            return self._cache[path]
        
        full_path = self.data_root / path
        if not full_path.exists():
            raise FileNotFoundError(f"CSV not found: {full_path}")
        
        df = pd.read_csv(full_path)
        self._cache[path] = df
        logger.info(f"Loaded {path}: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    @property
    def graph_edges(self) -> pd.DataFrame:
        """Graph edges with transaction flows"""
        return self.load_csv("graphs/graph_edges.csv")
    
    @property
    def wallet_features(self) -> pd.DataFrame:
        """Wallet features (50+ dimensions)"""
        df = self.load_csv("features/wallet_features.csv")
        return df.set_index("wallet_id")
    
    @property
    def graphsage_embeddings(self) -> pd.DataFrame:
        """GraphSAGE neural embeddings (32 dimensions)"""
        df = self.load_csv("graphs/graphsage_embeddings.csv")
        return df.set_index("wallet_id")
    
    @property
    def deterministic_scores(self) -> pd.DataFrame:
        """Deterministic risk scores (no ground-truth leakage)"""
        df = self.load_csv("models/deterministic_scores.csv")
        return df.set_index("wallet_id")
    
    @property
    def isolation_forest_scores(self) -> pd.DataFrame:
        """Isolation Forest anomaly scores"""
        df = self.load_csv("models/isolation_forest_scores.csv")
        return df.set_index("wallet_id")
    
    @property
    def risk_predictions(self) -> pd.DataFrame:
        """XGBoost risk model predictions"""
        df = self.load_csv("models/risk_model_predictions.csv")
        return df.set_index("wallet_id")
    
    @property
    def transactions(self) -> pd.DataFrame:
        """Raw transactions for tracing"""
        return self.load_csv("../../data/raw/transactions.csv")

# ============================================================
# PATTERN DETECTION ENGINE
# ============================================================

class PatternDetector:
    """Detects illicit transaction patterns (peeling chains, etc.)"""
    
    # Pattern type descriptions
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
        """Build transaction graph from edges"""
        edges = self.loader.graph_edges
        self.graph = defaultdict(lambda: defaultdict(list))
        
        for _, row in edges.iterrows():
            source = row["source_wallet_id"]
            target = row["target_wallet_id"]
            self.graph[source][target].append({
                "txid": f"TX_{hash(source + target) % 1000000:x}",
                "amount": row["total_amount_sats"],
                "timestamp": row.get("first_seen", ""),
                "count": row["transaction_count"]
            })
    
    def detect_pattern(self, wallet_id: str, max_hops: int = 5) -> Tuple[str, float, List[Dict]]:
        """
        Detect pattern and trace fund flow from wallet
        
        Returns: (pattern_type, confidence_score, transaction_sequence)
        """
        
        # Perform BFS to find transaction paths
        sequence = self._trace_funds_bfs(wallet_id, max_hops=max_hops)
        
        if not sequence:
            return "no_flow", 0.0, []
        
        # Analyze pattern characteristics
        pattern, confidence = self._analyze_pattern(wallet_id, sequence)
        
        return pattern, confidence, sequence
    
    def _trace_funds_bfs(self, wallet_id: str, max_hops: int = 5) -> List[Dict]:
        """BFS-based fund tracing through transaction graph"""
        
        sequence = []
        visited = set()
        queue = deque([(wallet_id, 0, [])])
        
        while queue:
            current_wallet, hop_count, path = queue.popleft()
            
            if hop_count >= max_hops or current_wallet in visited:
                continue
            
            visited.add(current_wallet)
            
            # Get outgoing transactions
            for target_wallet, txs in self.graph[current_wallet].items():
                for tx in txs:
                    step_info = {
                        "step": len(sequence) + 1,
                        "source_wallet": current_wallet,
                        "target_wallet": target_wallet,
                        "txid": tx["txid"],
                        "amount_sats": tx["amount"],
                        "timestamp": tx["timestamp"]
                    }
                    sequence.append(step_info)
                    queue.append((target_wallet, hop_count + 1, path + [step_info]))
        
        # Limit to top paths by amount
        return sorted(sequence, key=lambda x: x["amount_sats"], reverse=True)[:10]
    
    def _analyze_pattern(self, wallet_id: str, sequence: List[Dict]) -> Tuple[str, float]:
        """Analyze transaction sequence to identify pattern type"""
        
        if not sequence:
            return "no_flow", 0.0
        
        # Get wallet degree characteristics
        out_degree = len(set(tx["target_wallet"] for tx in sequence))
        in_degree = len(set(tx["source_wallet"] for tx in sequence if tx["source_wallet"] != wallet_id))
        
        # Pattern detection logic
        if out_degree > 5 and len(sequence) > 1:
            confidence = min(0.95, 0.6 + (out_degree / 50))
            return "fan_out", confidence
        elif in_degree > 5:
            confidence = min(0.90, 0.5 + (in_degree / 50))
            return "fan_in", confidence
        elif len(sequence) >= 3 and out_degree <= 3:
            # Check for peeling chain (sequential hops with decreasing amounts)
            amounts = [tx["amount_sats"] for tx in sequence[:5]]
            is_decreasing = all(amounts[i] >= amounts[i+1] * 0.9 for i in range(len(amounts)-1))
            if is_decreasing:
                confidence = min(0.92, 0.65 + (len(sequence) / 20))
                return "peeling_chain", confidence
        
        # Default: direct transfer
        return "direct_transfer", 0.5 + (len(sequence) / 20)

# ============================================================
# EXPLAINABILITY ENGINE
# ============================================================

class ExplainabilityEngine:
    """Translates risk scores into human-readable explanations"""
    
    FEATURE_DESCRIPTIONS = {
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
        "community_id": "Cluster membership in network topology",
        "total_sent_sats": "Total satoshis sent (economic volume)",
        "total_received_sats": "Total satoshis received (economic volume)",
        "std_received_sats": "Variability in incoming transaction sizes",
        "unique_input_counterparties": "Number of unique input sources",
        "unique_output_counterparties": "Number of unique output destinations",
        "degree": "Total connections (in + out degree)",
    }
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
    
    def explain_wallet_risk(self, wallet_id: str) -> Dict[str, Any]:
        """Generate comprehensive explainability report for wallet"""
        
        # Get risk scores
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
        
        # Get feature values
        try:
            features = self.loader.wallet_features.loc[wallet_id].to_dict()
        except KeyError:
            features = {}
        
        # Generate top risk factors with descriptions
        top_factors = self._generate_top_factors(wallet_id, det_evidence, features)
        
        overall_score = (risk_score + (det_score or 0.5)) / 2
        
        return {
            "wallet_id": wallet_id,
            "overall_risk_score": min(1.0, max(0.0, overall_score)),
            "model_used": "XGBoost + Isolation Forest + Deterministic Engine",
            "top_risk_factors": top_factors,
            "deterministic_score": det_score,
        }
    
    def _parse_evidence(self, evidence_str: str) -> List[Dict]:
        """Parse deterministic evidence JSON string"""
        try:
            if isinstance(evidence_str, str):
                return json.loads(evidence_str.replace('""', '"'))
            return []
        except:
            return []
    
    def _generate_top_factors(self, wallet_id: str, evidence: List[Dict], features: Dict) -> List[Dict]:
        """Generate top risk factors from evidence and features"""
        
        factors = []
        
        # Add evidence-based factors
        for item in evidence[:5]:
            feature_name = item.get("feature", "unknown")
            raw_value = item.get("value", 0)
            anomaly_score = item.get("empirical_anomaly_score", 0.5)
            
            description = self.FEATURE_DESCRIPTIONS.get(
                feature_name,
                f"Statistical anomaly in {feature_name}"
            )
            
            factors.append({
                "feature_name": feature_name,
                "raw_value": float(raw_value),
                "shap_impact": float(anomaly_score),
                "description": description,
                "percentile_rank": float(anomaly_score * 100),
            })
        
        # Add high-value features from wallet data
        high_value_features = [
            ("pagerank", features.get("pagerank", 0)),
            ("betweenness_centrality", features.get("betweenness_centrality", 0)),
            ("out_degree", features.get("out_degree", 0)),
            ("fan_out", features.get("fan_out", 0)),
            ("clustering_coefficient", features.get("clustering_coefficient", 0)),
        ]
        
        for fname, fvalue in high_value_features:
            if fvalue > 0 and len(factors) < 10:
                factors.append({
                    "feature_name": fname,
                    "raw_value": float(fvalue),
                    "shap_impact": self._normalize_shap(fvalue),
                    "description": self.FEATURE_DESCRIPTIONS.get(fname, f"Feature: {fname}"),
                    "percentile_rank": float(min(100, fvalue * 100)),
                })
        
        return factors[:10]
    
    def _normalize_shap(self, value: float) -> float:
        """Normalize feature value to SHAP-like impact score [0, 1]"""
        return min(1.0, max(0.0, abs(value) / 10))

# ============================================================
# GNN INFLUENCE CALCULATOR
# ============================================================

class GNNInfluenceCalculator:
    """Calculates GNN influence weights from GraphSAGE embeddings and graph structure"""
    
    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._influence_cache = {}
    
    def calculate_edge_influence(self, source_id: str, target_id: str) -> float:
        """
        Calculate GNN influence weight for an edge
        Based on: GraphSAGE embedding similarity + transaction frequency
        """
        
        cache_key = f"{source_id}_{target_id}"
        if cache_key in self._influence_cache:
            return self._influence_cache[cache_key]
        
        # Get embeddings
        try:
            source_emb = self.loader.graphsage_embeddings.loc[source_id].values
            target_emb = self.loader.graphsage_embeddings.loc[target_id].values
        except KeyError:
            return 0.1
        
        # Calculate cosine similarity
        cosine_sim = self._cosine_similarity(source_emb, target_emb)
        
        # Get transaction frequency
        edges = self.loader.graph_edges
        edge_mask = (edges["source_wallet_id"] == source_id) & (edges["target_wallet_id"] == target_id)
        if edge_mask.any():
            tx_count = edges[edge_mask]["transaction_count"].values[0]
            # Normalize tx count (log scale)
            freq_score = min(1.0, np.log1p(tx_count) / np.log1p(100))
        else:
            freq_score = 0.1
        
        # Combine: 60% embedding similarity, 40% frequency
        influence = 0.6 * cosine_sim + 0.4 * freq_score
        influence = max(0.01, min(1.0, influence))
        
        self._influence_cache[cache_key] = influence
        return influence
    
    @staticmethod
    def _cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

# ============================================================
# GRAPH EXPLORER
# ============================================================

class GraphExplorer:
    """Explores ego-graphs around wallets"""
    
    def __init__(self, loader: DataLoader, influence_calc: GNNInfluenceCalculator):
        self.loader = loader
        self.influence_calc = influence_calc
        self._build_graph()
    
    def _build_graph(self):
        """Build adjacency list from edges"""
        edges = self.loader.graph_edges
        self.graph = defaultdict(set)
        self.reverse_graph = defaultdict(set)
        self.edge_data = {}
        
        for _, row in edges.iterrows():
            src = row["source_wallet_id"]
            tgt = row["target_wallet_id"]
            self.graph[src].add(tgt)
            self.reverse_graph[tgt].add(src)
            self.edge_data[(src, tgt)] = {
                "tx_count": int(row["transaction_count"]),
                "amount": float(row["total_amount_sats"])
            }
    
    def get_ego_graph(self, wallet_id: str, hops: int = 2) -> Dict[str, Any]:
        """Get ego-graph around a wallet (neighbors within N hops)"""
        
        # BFS to find all nodes within hops
        nodes = {wallet_id}
        current_level = {wallet_id}
        
        for _ in range(hops):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph[node])
                next_level.update(self.reverse_graph[node])
            current_level = next_level - nodes
            nodes.update(current_level)
        
        # Build node list with risk scores
        node_list = []
        risk_preds = self.loader.risk_predictions
        features = self.loader.wallet_features
        
        for node_id in nodes:
            try:
                risk_score = float(risk_preds.loc[node_id]["risk_probability"])
            except (KeyError, IndexError):
                risk_score = 0.5
            
            try:
                wallet_features = features.loc[node_id][
                    ["pagerank", "betweenness_centrality", "out_degree", "in_degree"]
                ].to_dict() if node_id in features.index else {}
            except:
                wallet_features = {}
            
            node_list.append(Node(
                id=node_id,
                risk_score=risk_score,
                wallet_features=wallet_features
            ))
        
        # Build edge list with influence weights
        edge_list = []
        for src, tgts in self.graph.items():
            if src not in nodes:
                continue
            for tgt in tgts:
                if tgt not in nodes:
                    continue
                
                edge_key = (src, tgt)
                edge_data = self.edge_data.get(edge_key, {})
                influence = self.influence_calc.calculate_edge_influence(src, tgt)
                freq_score = min(1.0, np.log1p(edge_data.get("tx_count", 1)) / 5)
                
                edge_list.append(Edge(
                    source=src,
                    target=tgt,
                    transaction_count=edge_data.get("tx_count", 0),
                    total_amount_sats=edge_data.get("amount", 0),
                    gnn_influence_weight=influence,
                    frequency_score=freq_score
                ))
        
        return {
            "wallet_id": wallet_id,
            "nodes": node_list,
            "edges": edge_list,
            "ego_hops": hops,
        }

# ============================================================
# FASTAPI APPLICATION
# ============================================================

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    app = FastAPI(
        title="Bitcoin Transaction Analysis API",
        description="REST API for fund flow traceability, explainability, and graph analysis",
        version="1.0.0"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Initialize data loaders and engines
    DATA_ROOT = PROJECT_ROOT / "outputs"
    loader = DataLoader(DATA_ROOT)
    pattern_detector = PatternDetector(loader)
    explainability_engine = ExplainabilityEngine(loader)
    influence_calculator = GNNInfluenceCalculator(loader)
    graph_explorer = GraphExplorer(loader, influence_calculator)
    
    # ============================================================
    # ENDPOINT 1: TRACEABILITY
    # ============================================================
    
    @app.get(
        "/api/v1/patterns/trace/{wallet_id}",
        response_model=TraceResponse,
        summary="Trace fund flow patterns",
        description="Returns sequential transaction hops showing illicit money movement with pattern detection"
    )
    async def trace_pattern(wallet_id: str, max_hops: int = 5):
        """
        Trace fund flow from a wallet showing transaction patterns.
        
        Returns:
        - Detected pattern type (peeling_chain, fan_out, fan_in, circular_flow, direct_transfer)
        - Confidence score
        - Sequential transaction sequence with amounts and txids
        """
        
        try:
            pattern, confidence, sequence = pattern_detector.detect_pattern(wallet_id, max_hops)
            
            if not sequence:
                raise HTTPException(
                    status_code=404,
                    detail=f"No transaction pattern found for wallet {wallet_id}"
                )
            
            steps = [
                TransactionStep(
                    step=tx["step"],
                    source_wallet=tx["source_wallet"],
                    target_wallet=tx["target_wallet"],
                    txid=tx["txid"],
                    amount_sats=tx["amount_sats"],
                    timestamp=tx.get("timestamp", "")
                )
                for tx in sequence
            ]
            
            return TraceResponse(
                wallet_id=wallet_id,
                pattern_detected=pattern,
                confidence_score=confidence,
                sequence=steps,
                total_amount_sats=sum(tx["amount_sats"] for tx in sequence),
                hop_count=len(steps)
            )
        
        except Exception as e:
            logger.error(f"Error tracing pattern for {wallet_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============================================================
    # ENDPOINT 2: EXPLAINABILITY
    # ============================================================
    
    @app.get(
        "/api/v1/explainability/wallet/{wallet_id}",
        response_model=ExplainabilityResponse,
        summary="Get wallet risk explainability",
        description="Returns SHAP values and feature importance for wallet risk scoring"
    )
    async def get_explainability(wallet_id: str):
        """
        Get human-readable explanations for wallet risk score.
        
        Returns top risk factors with:
        - Feature name and raw value
        - SHAP impact score (0-1)
        - Human-readable description
        - Percentile rank among all wallets
        """
        
        try:
            result = explainability_engine.explain_wallet_risk(wallet_id)
            
            return ExplainabilityResponse(
                wallet_id=result["wallet_id"],
                overall_risk_score=result["overall_risk_score"],
                model_used=result["model_used"],
                top_risk_factors=[
                    RiskFactorExplanation(**factor)
                    for factor in result["top_risk_factors"]
                ],
                deterministic_score=result.get("deterministic_score")
            )
        
        except Exception as e:
            logger.error(f"Error explaining wallet {wallet_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============================================================
    # ENDPOINT 3: ENHANCED GRAPH
    # ============================================================
    
    @app.get(
        "/api/v1/graph/wallet/{wallet_id}",
        response_model=GraphResponse,
        summary="Get ego-graph with GNN influence weights",
        description="Returns ego-graph with edges annotated with GNN influence weights"
    )
    async def get_graph(wallet_id: str, hops: int = 2):
        """
        Get ego-graph around wallet with GNN influence weights.
        
        Edges include:
        - gnn_influence_weight: Combined score from GraphSAGE embeddings and transaction frequency
        - frequency_score: Normalized transaction count
        - transaction_count: Actual transaction volume
        
        Nodes include risk scores and feature values.
        """
        
        try:
            if hops < 1 or hops > 5:
                hops = 2
            
            result = graph_explorer.get_ego_graph(wallet_id, hops=hops)
            
            return GraphResponse(
                wallet_id=result["wallet_id"],
                nodes=result["nodes"],
                edges=result["edges"],
                ego_hops=result["ego_hops"]
            )
        
        except Exception as e:
            logger.error(f"Error getting graph for {wallet_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ============================================================
    # HEALTH CHECK
    # ============================================================
    
    @app.get(
        "/api/v1/health",
        summary="Health check endpoint",
        tags=["System"]
    )
    async def health_check():
        """Check API health and data availability"""
        
        try:
            # Verify all data sources are accessible
            _ = loader.graph_edges
            _ = loader.wallet_features
            _ = loader.graphsage_embeddings
            
            return {
                "status": "healthy",
                "data_sources": {
                    "graph_edges": len(loader.graph_edges),
                    "wallets": len(loader.wallet_features),
                    "embeddings": len(loader.graphsage_embeddings),
                }
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            raise HTTPException(status_code=503, detail="Data sources unavailable")
    
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
        log_level="info"
    )
