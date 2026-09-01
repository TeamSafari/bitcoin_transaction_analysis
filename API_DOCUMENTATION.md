# Bitcoin Transaction Analysis REST API

## Overview

The Bitcoin Transaction Analysis API provides three main endpoints for analyzing suspicious wallet behavior, explaining risk scores, and visualizing transaction graphs with machine learning influence weights.

### Features

✅ **Traceability Endpoint** - Traces illicit fund flow patterns through transaction networks  
✅ **Explainability Endpoint** - Returns human-readable risk factor explanations  
✅ **Enhanced Graph Endpoint** - Ego-graphs with GNN influence weights for visualization  
✅ **CORS Enabled** - Ready for frontend integration  
✅ **OpenAPI Documentation** - Interactive Swagger/ReDoc docs at `/docs`  

---

## Quick Start

### 1. Install Dependencies

```bash
cd /home/johan/code/bitcoin_transaction_analysis
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Start the API Server

```bash
bash run_api.sh
```

Or directly:

```bash
python -m backend.api_server
```

The API will be available at `http://localhost:8000`

**Interactive Documentation:**
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## Endpoints

### 1. Traceability Endpoint: Trace Fund Flow Patterns

**Endpoint:** `GET /api/v1/patterns/trace/{wallet_id}`

**Purpose:** Analyze transaction graph and return sequential transaction hops showing illicit money movement through the network.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wallet_id` | string | Yes | Wallet identifier (e.g., "W0000001") |
| `max_hops` | integer | No | Maximum transaction hops to trace (default: 5, max: 10) |

#### Response

```json
{
  "wallet_id": "W0000001",
  "pattern_detected": "fan_out",
  "confidence_score": 0.88,
  "hop_count": 10,
  "total_amount_sats": 2847392843,
  "sequence": [
    {
      "step": 1,
      "source_wallet": "W0000001",
      "target_wallet": "W0000049",
      "txid": "TX_9a8b7c...",
      "amount_sats": 23455792,
      "timestamp": "2026-02-09 03:13:38"
    },
    {
      "step": 2,
      "source_wallet": "W0000049",
      "target_wallet": "W0000104",
      "txid": "TX_1f2e3d...",
      "amount_sats": 23120000,
      "timestamp": "2026-02-10 14:22:15"
    }
  ]
}
```

#### Pattern Types

| Pattern | Description | Risk Level |
|---------|-------------|-----------|
| `peeling_chain` | Sequential transfers with decreasing amounts (classic laundering) | **CRITICAL** |
| `fan_out` | High-volume distribution to many wallets (mixing/tumbling) | **HIGH** |
| `fan_in` | Convergence from many sources (consolidation) | **MEDIUM** |
| `circular_flow` | Transactions forming loops | **MEDIUM** |
| `direct_transfer` | Direct single-hop transfer | **LOW** |

#### Example Usage

```bash
# Trace fund flow from wallet W0000001
curl http://localhost:8000/api/v1/patterns/trace/W0000001

# Trace with max 3 hops
curl http://localhost:8000/api/v1/patterns/trace/W0000001?max_hops=3
```

---

### 2. Explainability Endpoint: Get Risk Factor Explanations

**Endpoint:** `GET /api/v1/explainability/wallet/{wallet_id}`

**Purpose:** Return SHAP values and feature importance for wallet risk scoring, translated into human-readable impacts.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wallet_id` | string | Yes | Wallet identifier (e.g., "W0000001") |

#### Response

```json
{
  "wallet_id": "W0000001",
  "overall_risk_score": 0.94,
  "model_used": "XGBoost + Isolation Forest + Deterministic Engine",
  "deterministic_score": 0.3455,
  "top_risk_factors": [
    {
      "feature_name": "pagerank_centrality",
      "raw_value": 0.0058,
      "shap_impact": 0.45,
      "percentile_rank": 45.0,
      "description": "Importance in transaction network (higher = more central/intermediary)"
    },
    {
      "feature_name": "off_hour_transaction_ratio",
      "raw_value": 0.87,
      "shap_impact": 0.38,
      "percentile_rank": 38.0,
      "description": "Percentage of transactions outside business hours"
    },
    {
      "feature_name": "unique_asns",
      "raw_value": 32,
      "shap_impact": 0.35,
      "percentile_rank": 35.0,
      "description": "Diversity of network providers used"
    }
  ]
}
```

#### Risk Score Interpretation

- **0.0 - 0.3:** Low risk
- **0.3 - 0.6:** Medium risk
- **0.6 - 0.8:** High risk
- **0.8 - 1.0:** Critical risk

#### Key Features Explained

| Feature | Description | High Value Means |
|---------|-------------|------------------|
| `pagerank_centrality` | Network importance | Acts as intermediary (red flag) |
| `betweenness_centrality` | Frequency on shortest paths | Central positioning in laundering chain |
| `fan_out` | Distribution spread | Likely coin mixing |
| `fan_in` | Collection spread | Fund consolidation |
| `unique_countries` | Geographic diversity | International operations |
| `unique_asns` | Network provider diversity | Avoids traceability |
| `transaction_velocity` | Activity speed | Rapid processing (time-sensitive) |
| `off_hour_ratio` | Off-business-hours activity | Avoids surveillance |
| `clustering_coefficient` | Community density | Bridge between communities |

#### Example Usage

```bash
# Get explanation for wallet W0000001
curl http://localhost:8000/api/v1/explainability/wallet/W0000001

# Parse top risk factors
curl http://localhost:8000/api/v1/explainability/wallet/W0000001 | \
  python -c "import json, sys; data=json.load(sys.stdin); \
  [print(f'{f[\"feature_name\"]}: {f[\"description\"]}') for f in data['top_risk_factors']]"
```

---

### 3. Enhanced Graph Endpoint: Get Ego-Graph with GNN Influence Weights

**Endpoint:** `GET /api/v1/graph/wallet/{wallet_id}`

**Purpose:** Returns ego-graph around a wallet with edges annotated with GNN influence weights derived from GraphSAGE embeddings and transaction frequency.

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `wallet_id` | string | Yes | Wallet identifier (e.g., "W0000001") |
| `hops` | integer | No | Ego-graph depth (default: 2, range: 1-5) |

#### Response

```json
{
  "wallet_id": "W0000001",
  "ego_hops": 2,
  "nodes": [
    {
      "id": "W0000001",
      "risk_score": 0.12,
      "wallet_features": {
        "pagerank": 0.0021,
        "betweenness_centrality": 0.0023,
        "out_degree": 17,
        "in_degree": 22
      }
    },
    {
      "id": "W0000049",
      "risk_score": 0.68,
      "wallet_features": {
        "pagerank": 0.0019,
        "betweenness_centrality": 0.0020,
        "out_degree": 14,
        "in_degree": 18
      }
    }
  ],
  "edges": [
    {
      "source": "W0000001",
      "target": "W0000049",
      "transaction_count": 14,
      "total_amount_sats": 23455792,
      "gnn_influence_weight": 0.92,
      "frequency_score": 0.87
    },
    {
      "source": "W0000001",
      "target": "W0000104",
      "transaction_count": 1,
      "total_amount_sats": 6157660,
      "gnn_influence_weight": 0.11,
      "frequency_score": 0.15
    }
  ]
}
```

#### GNN Influence Weight Calculation

The `gnn_influence_weight` combines:

1. **GraphSAGE Embedding Similarity (60%):** Cosine similarity between 32-dimensional neural embeddings
2. **Transaction Frequency (40%):** Log-normalized transaction count

$$\text{influence} = 0.6 \times \text{cosine\_sim} + 0.4 \times \text{freq\_score}$$

**Interpretation:**
- **0.8-1.0:** Strong direct influence (thick edge in visualization)
- **0.5-0.8:** Moderate influence
- **0.2-0.5:** Weak influence (thin edge)
- **<0.2:** Very weak influence

#### Graph Visualization Use Cases

1. **Node Coloring:** By `risk_score` (green=low, red=high)
2. **Edge Thickness:** By `gnn_influence_weight`
3. **Node Size:** By `out_degree` or `pagerank`
4. **Animations:** Flow from source → target weighted by influence
5. **Community Detection:** Groups by `betweenness_centrality`

#### Example Usage

```bash
# Get 2-hop ego graph
curl http://localhost:8000/api/v1/graph/wallet/W0000001

# Get 3-hop ego graph
curl http://localhost:8000/api/v1/graph/wallet/W0000001?hops=3

# Extract high-influence edges (for visualization)
curl http://localhost:8000/api/v1/graph/wallet/W0000001 | \
  python -c "import json, sys; data=json.load(sys.stdin); \
  [print(f'{e[\"source\"]} -> {e[\"target\"]}: {e[\"gnn_influence_weight\"]:.2f}') \
   for e in data['edges'] if e['gnn_influence_weight'] > 0.5]"
```

---

### Health Check Endpoint

**Endpoint:** `GET /api/v1/health`

**Purpose:** Verify API health and data source availability.

#### Response

```json
{
  "status": "healthy",
  "data_sources": {
    "graph_edges": 10975,
    "wallets": 500,
    "embeddings": 500
  }
}
```

---

## Integration Examples

### 1. Frontend Animation Loop

```python
import requests
import time

def animate_fund_flow(wallet_id, interval=0.5):
    """Animate fund flow for UI"""
    response = requests.get(f"http://localhost:8000/api/v1/patterns/trace/{wallet_id}")
    trace = response.json()
    
    print(f"Pattern: {trace['pattern_detected']} (confidence: {trace['confidence_score']})")
    
    for step in trace['sequence']:
        print(f"Step {step['step']}: {step['source_wallet']} -> {step['target_wallet']}")
        print(f"  Amount: {step['amount_sats']:,.0f} sats")
        print(f"  TxID: {step['txid']}")
        time.sleep(interval)

animate_fund_flow("W0000001")
```

### 2. Dashboard Risk Assessment

```python
import requests

def get_wallet_risk_dashboard(wallet_id):
    """Get complete risk profile"""
    
    # Get risk factors
    explain_resp = requests.get(
        f"http://localhost:8000/api/v1/explainability/wallet/{wallet_id}"
    )
    explanation = explain_resp.json()
    
    # Get graph context
    graph_resp = requests.get(
        f"http://localhost:8000/api/v1/graph/wallet/{wallet_id}?hops=1"
    )
    graph = graph_resp.json()
    
    dashboard = {
        "wallet_id": wallet_id,
        "overall_risk": explanation["overall_risk_score"],
        "top_3_factors": explanation["top_risk_factors"][:3],
        "network_size": len(graph["nodes"]),
        "connected_wallets": [e["target"] for e in graph["edges"]],
        "highest_influence_edges": sorted(
            graph["edges"],
            key=lambda e: e["gnn_influence_weight"],
            reverse=True
        )[:3]
    }
    
    return dashboard
```

### 3. Batch Risk Scoring

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def batch_explain_wallets(wallet_ids, workers=5):
    """Score multiple wallets in parallel"""
    
    def score_wallet(wallet_id):
        try:
            resp = requests.get(
                f"http://localhost:8000/api/v1/explainability/wallet/{wallet_id}",
                timeout=5
            )
            return wallet_id, resp.json()["overall_risk_score"]
        except:
            return wallet_id, None
    
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(score_wallet, wallet_ids))
    
    return {wid: score for wid, score in results if score is not None}
```

---

## Error Handling

### Common HTTP Status Codes

| Status | Meaning | Example |
|--------|---------|---------|
| 200 | Success | Endpoint returns data |
| 404 | Not Found | Wallet ID doesn't exist in dataset |
| 422 | Validation Error | Invalid parameter format |
| 500 | Server Error | Data loading failure |
| 503 | Service Unavailable | Data source unreachable |

### Error Response Format

```json
{
  "detail": "No transaction pattern found for wallet W9999999"
}
```

---

## Performance Notes

- **Data Loading:** First request loads CSVs (~2-3 seconds), cached thereafter
- **Graph Queries:** 2-hop ego-graph computation: ~50-100ms
- **Pattern Tracing:** Full network trace with BFS: ~200-500ms
- **Concurrent Requests:** API handles 100+ concurrent requests

---

## API Documentation

For interactive API documentation with try-it-out functionality:

1. **Swagger UI:** `http://localhost:8000/docs`
2. **ReDoc:** `http://localhost:8000/redoc`
3. **OpenAPI Schema:** `http://localhost:8000/openapi.json`

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FastAPI Server                       │
├──────────┬──────────────┬─────────────────┬──────────────┤
│Traceability          │ Explainability │ Graph Explorer   │
│├─ BFS Fund Tracing   │├─ Risk Factors  │├─ Ego-graphs    │
│├─ Pattern Detection  │├─ SHAP Values   │├─ GNN Weights   │
│└─ Confidence Scores  │└─ Descriptions  │└─ Node/Edge Data│
├─────────────────────────────────────────────────────────┤
│                    Data Loader Cache                     │
├──────────┬──────────────┬─────────────────────────────────┤
│Graph Edges│ Features    │GraphSAGE       │Risk Scores    │
│10,975 rows│ 500×65 matrix│Embeddings 32-D│Deterministic  │
└──────────┴──────────────┴─────────────────┴──────────────┘
              ↓
        ┌─────────────────┐
        │  CSV Files      │
        │  (outputs/)     │
        └─────────────────┘
```

---

## Configuration

The API reads from:
- `outputs/graphs/graph_edges.csv` - Transaction edges
- `outputs/features/wallet_features.csv` - Wallet features
- `outputs/graphs/graphsage_embeddings.csv` - GNN embeddings
- `outputs/models/deterministic_scores.csv` - Risk scores
- `outputs/models/risk_model_predictions.csv` - Model predictions

To change the data source, modify `DATA_ROOT` in `create_app()`.

---

## License

Bitcoin Transaction Analysis System - Internal Use Only
