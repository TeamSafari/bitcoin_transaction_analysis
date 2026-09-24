"""Pydantic response models for the REST API."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class JobUploadResponse(BaseModel):
    job_id: str
    status: str
    message: str = "Pipeline job queued."


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
    model_used: Optional[str] = "XGBoost + SHAP + Statistical Fusion"
    top_risk_factors: List[RiskFactorExplanation]
    deterministic_score: Optional[float] = None
    fusion_score: Optional[float] = None


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
    is_anomaly: bool = False
    degree: int = 0
    pagerank: float = 0.0
    community_id: str = "0"
    wallet_features: Dict[str, Any] = {}


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


class AlertsResponse(BaseModel):
    job_id: str
    status: str
    total_alerts: int
    alerts: List[Dict[str, Any]]


class LLMExplanationResponse(BaseModel):
    wallet_id: str
    explanation: str
    model_used: str = "Qwen2.5-1.5B-Instruct"
    cached: bool = False
