"""
Configurable settings for the LLM explanation generator.

Edit the feature lists below to change what context the model receives.
Restart the server after changes.
"""

from __future__ import annotations

from backend.config import ARTIFACT_DIR

# ---------------------------------------------------------------------------
# Model settings
# ---------------------------------------------------------------------------
LLM_MODEL_DIR = ARTIFACT_DIR / "llm"
LLM_MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
LLM_CONTEXT_SIZE = 2048
LLM_MAX_TOKENS = 300
LLM_TEMPERATURE = 0.3
LLM_N_THREADS = 4  # CPU threads for inference

# ---------------------------------------------------------------------------
# SHAP feature selection
# ---------------------------------------------------------------------------
# How many top SHAP factors to include in the prompt
TOP_SHAP_COUNT = 5

# Prefixes to EXCLUDE from SHAP factors (opaque embeddings)
SHAP_EXCLUDE_PREFIXES = [
    "graphsage_dim_",
]

# ---------------------------------------------------------------------------
# Behavioral features from wallet_features.csv
# ---------------------------------------------------------------------------
BEHAVIORAL_FEATURES = [
    "tx_count",
    "total_sent_sats",
    "total_received_sats",
    "median_sent_sats",
    "active_duration_seconds",
    "burstiness",
    "rapid_hop_count",
    "unique_counterparties",
    "fan_in",
    "fan_out",
    "in_out_ratio",
    "observations_per_tx",
]

# ---------------------------------------------------------------------------
# Deterministic evidence
# ---------------------------------------------------------------------------
TOP_DET_EVIDENCE_COUNT = 3

# ---------------------------------------------------------------------------
# Human-readable display names for features
# ---------------------------------------------------------------------------
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "median_sent_sats": "Median Sent Amount",
    "isolation_forest_anomaly_score": "Isolation Forest Score",
    "observations_per_tx": "Network Observations per Tx",
    "country_entropy": "Geographic Diversity (Country Entropy)",
    "avg_sent_sats": "Average Sent Amount",
    "autoencoder_reconstruction_error": "Autoencoder Reconstruction Error",
    "deterministic_score": "Rule-Based Risk Score",
    "max_sent_sats": "Max Sent Amount",
    "avg_outputs_per_tx": "Avg Outputs per Tx",
    "asn_entropy": "Network Provider Diversity (ASN Entropy)",
    "std_sent_sats": "Sent Amount Variability",
    "total_bytes_sent": "Total Network Bytes Sent",
    "unique_src_ips": "Unique Source IPs",
    "weighted_out_degree": "Weighted Out-Degree",
    "median_received_sats": "Median Received Amount",
    "active_duration_seconds": "Active Duration",
    "total_sent_sats": "Total Sent",
    "total_received_sats": "Total Received",
    "median_latency_ms": "Median Network Latency",
    "burstiness": "Burstiness",
    "pagerank": "PageRank Centrality",
    "betweenness_centrality": "Betweenness Centrality",
    "tx_count": "Transaction Count",
    "rapid_hop_count": "Rapid Hop Count",
    "unique_counterparties": "Unique Counterparties",
    "fan_in": "Fan-In",
    "fan_out": "Fan-Out",
    "in_out_ratio": "In/Out Ratio",
    "std_inter_tx_seconds": "Tx Timing Variability",
    "min_inter_tx_seconds": "Min Inter-Tx Time",
    "avg_inter_tx_seconds": "Avg Inter-Tx Time",
    "unique_asns": "Unique ASNs",
    "unique_countries": "Unique Countries",
    "clustering_coefficient": "Clustering Coefficient",
    "degree": "Graph Degree",
    "in_degree": "In-Degree",
    "out_degree": "Out-Degree",
}
