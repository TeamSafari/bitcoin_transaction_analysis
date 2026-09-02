# API Architecture: Asynchronous ML Pipeline

This document outlines the API endpoints required to transition from a static demo to a dynamic application that processes any uploaded Bitcoin CSV file. Because processing a bulk CSV (Feature Engineering, GraphSAGE, Isolation Forest, LightGBM, SHAP) is computationally expensive, the architecture uses a Job/Task-based asynchronous workflow.

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

Here is exactly what happens in the backend memory after that endpoint is hit:
1. **Step 1: Ingestion.** The backend reads the uploaded CSV into a Pandas DataFrame.
2. **Step 2: Feature Engineering.** The backend automatically passes that DataFrame into your existing scripts (`temporal_features.py`, `network_features.py`, etc.) to calculate PageRank, degrees, and transaction frequencies.
3. **Step 3: GraphSAGE.** The backend constructs the NetworkX graph and runs the pre-trained `graphsage.pt` model to generate embeddings.
4. **Step 4: ML Pipeline.** The backend feeds the features and embeddings into your pre-trained `isolation_forest.pkl` and `risk_model.pkl` to generate the final predictions.
5. **Step 5: SHAP Engine.** Finally, it runs `shap_engine.py` on the flagged anomalies to calculate the feature importances.
6. **Step 6: Storage.** The backend saves all these final outputs into the database (or output folders) tagged with that specific `job_id`.

While the backend is busy running Steps 1 through 6, the frontend is continuously pinging the second endpoint: `GET /api/v1/jobs/{job_id}/status` to display the progress to the user.
