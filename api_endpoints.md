# API Architecture: Asynchronous ML Pipeline

This document describes the live REST API implemented in `backend/api/`.

**Run the server:** `cd backend && python main.py` (with venv activated).

Because the full forensic pipeline is computationally expensive, uploads use a job-based async workflow (`backend/api/worker.py` → `backend/pipeline/orchestrator.py`).

## 1. Ingestion & Processing Pipeline
These endpoints handle the file upload and asynchronous ML processing.

| Endpoint | Method | Function | Output / Returns |
| :--- | :---: | :--- | :--- |
| `/api/v1/jobs/upload` | **POST** | Uploads the raw CSV. Triggers Feature Engineering, GraphSAGE, Isolation Forest, LightGBM, and SHAP calculators in the background. | `{"job_id": "job-8f92a", "status": "processing"}` |
| `/api/v1/jobs/{job_id}/status` | **GET** | Frontend polls this to update the loading bar while the heavy ML models run. | `{"progress": 75, "step": "Calculating SHAP values..."}` |
| `/api/v1/jobs/{job_id}/summary` | **GET** | Fetches high-level metrics once the job is 100% complete. | `{"total_wallets": 1500, "anomalies_found": 12}` |

## 2. Graph Rendering
These endpoints serve the nodes and edges required for Cytoscape.js to draw the network.

| Endpoint | Method | Function | Output / Returns |
| :--- | :---: | :--- | :--- |
| `/api/v1/jobs/{job_id}/graph/overview` | **GET** | Returns the macroscopic view of the network. *(Note: If the CSV is massive, this should return a down-sampled graph or only the clusters containing anomalies to prevent the browser from crashing).* | `{ "nodes": [...], "edges": [...] }` |
| `/api/v1/jobs/{job_id}/graph/wallet/{wallet_id}` | **GET** | Returns the **Contextual Sub-Graph (Ego Graph)**. Used when an analyst searches for a specific wallet to view its 2-hop radius. | `{ "nodes": [...], "edges": [{"source": "W1", "target": "W2", "gnn_influence_weight": 0.85}] }` |

## 3. Alerts & Explainability (XAI)
These endpoints populate the Sidebar panel without needing any hardcoded text.

| Endpoint | Method | Function | Output / Returns |
| :--- | :---: | :--- | :--- |
| `/api/v1/jobs/{job_id}/alerts` | **GET** | Fetches the list of flagged wallets to display in a table/list for the analyst to investigate. | `{"alerts": [{"wallet_id": "W001", "risk_probability": 0.98}, ...]}` |
| `/api/v1/jobs/{job_id}/explainability/wallet/{wallet_id}` | **GET** | Fetches the raw SHAP feature importances to dynamically explain *why* the wallet was flagged. | `{"top_risk_factors": [{"feature": "pagerank", "shap_value": 2.43, "direction": "increases_risk"}]}` |

## 4. Traceability Animations
This endpoint provides the logic for the visual tracing features.

| Endpoint | Method | Function | Output / Returns |
| :--- | :---: | :--- | :--- |
| `/api/v1/jobs/{job_id}/patterns/trace/{wallet_id}` | **GET** | Returns the exact, ordered sequence of transactions (e.g., a peeling chain or mixing path) downstream from the flagged wallet. | `{"pattern": "peeling_chain", "sequence": [{"source": "W1", "target": "W2"}, {"source": "W2", "target": "W3"}]}` |

---

## How the Frontend Will Use This
1. The user uploads `dataset_A.csv`. The UI calls `POST /upload` and gets back `job_id = "123"`.
2. The UI polls `GET /status` until it hits 100%.
3. The UI calls `GET /graph/overview?job_id=123` and renders the global Cytoscape network.
4. The user clicks a red node (Wallet X).
5. The UI calls `GET /explainability/wallet/X?job_id=123` and populates the SHAP Phase 3 sidebar.
6. The user clicks "Trace Flow". The UI calls `GET /patterns/trace/X?job_id=123` and animates the edges returned in that array.

---

## Detailed Workflow: `POST /api/v1/jobs/upload`
When the frontend sends the CSV file to this endpoint, the backend does **not** just save the file. Instead, this endpoint acts as the master trigger for your entire data science pipeline. 

Because the pipeline takes time, this endpoint kicks off a **background task** and immediately returns a `job_id` to the frontend.

After upload, `backend/api/worker.py` runs the orchestrator (`backend/pipeline/orchestrator.py`) in a background task:

1. **Data Ingestion** — validate and normalize raw CSVs
2. **Identity Resolution** — link wallets to entities and IP fingerprints
3. **Graph Construction** — build wallet graph edges and structural features
4. **Feature Engineering** — transaction, temporal, network, and correlation features
5. **Rule Engine** — deterministic statistical risk scoring
6. **GNN Engine** — GraphSAGE embeddings via `graphsage.pt`
7. **Anomaly Detectors** — Isolation Forest and Autoencoder
8. **Feature Fusion + Risk Model** — XGBoost inference via `risk_model.pkl`
9. **Fusion & Scoring** — correlation-adjusted multi-detector fusion
10. **Explainability** — SHAP feature contributions
11. **Alert Generation** — ranked alert queue per wallet

All artifacts are written to `outputs/jobs/{job_id}/results/`; job metadata is stored in SQLite (`outputs/jobs.db`).

While the pipeline runs, the frontend polls `GET /api/v1/jobs/{job_id}/status` for progress and stage names.

## Global routes (optional `job_id` query param)

| Endpoint | Method |
| :--- | :---: |
| `/api/v1/health` | GET |
| `/api/v1/jobs` | GET |
| `/api/v1/jobs/{job_id}` | GET |
| `/api/v1/jobs/{job_id}/manifest` | GET |
| `/api/v1/graph/overview?job_id=` | GET |
| `/api/v1/graph/wallet/{wallet_id}?job_id=` | GET |
| `/api/v1/alerts?job_id=` | GET |
| `/api/v1/explainability/wallet/{wallet_id}?job_id=` | GET |
| `/api/v1/patterns/trace/{wallet_id}?job_id=` | GET |
