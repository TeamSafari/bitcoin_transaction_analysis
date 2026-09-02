"""
Comprehensive Verification Suite for all endpoints in api_endpoints.md
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import requests
import uvicorn

from backend.api_server import create_app
from backend.database import init_db

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TEST_PORT = 8799
BASE_URL = f"http://127.0.0.1:{TEST_PORT}/api/v1"


def run_test_server():
    app = create_app()
    config = uvicorn.Config(app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    time.sleep(2)
    return server


def test_api():
    server = run_test_server()
    print("\n" + "=" * 60)
    print("TESTING COMPLETE API ARCHITECTURE (api_endpoints.md)")
    print("=" * 60)

    # 1. Health Check
    health_resp = requests.get(f"{BASE_URL}/health")
    assert health_resp.status_code == 200
    print("[PASS] 1. GET /api/v1/health passed:", health_resp.json())

    # 2. Upload CSV files
    print("\n2. POST /api/v1/jobs/upload")
    files_payload = {
        "wallets": ("wallets.csv", open(RAW_DIR / "wallets.csv", "rb"), "text/csv"),
        "transactions": ("transactions.csv", open(RAW_DIR / "transactions.csv", "rb"), "text/csv"),
        "transaction_inputs": ("transaction_inputs.csv", open(RAW_DIR / "transaction_inputs.csv", "rb"), "text/csv"),
        "transaction_outputs": ("transaction_outputs.csv", open(RAW_DIR / "transaction_outputs.csv", "rb"), "text/csv"),
        "network_observations": ("network_observations.csv", open(RAW_DIR / "network_observations.csv", "rb"), "text/csv"),
        "ip_metadata": ("ip_metadata.csv", open(RAW_DIR / "ip_metadata.csv", "rb"), "text/csv"),
    }
    upload_resp = requests.post(f"{BASE_URL}/jobs/upload", files=files_payload)
    for _, f_tuple in files_payload.items():
        f_tuple[1].close()

    assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
    upload_data = upload_resp.json()
    job_id = upload_data["job_id"]
    assert job_id.startswith("job-")
    assert upload_data["status"] == "processing"
    print(f"[PASS] Upload succeeded: job_id={job_id}, status={upload_data['status']}")

    # 3. Poll Status
    print(f"\n3. GET /api/v1/jobs/{job_id}/status (Polling...)")
    max_wait = 90
    start = time.time()
    completed = False
    while time.time() - start < max_wait:
        st_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/status")
        assert st_resp.status_code == 200
        st_data = st_resp.json()
        status = st_data["status"]
        progress = st_data.get("progress", 0)
        step = st_data.get("step")
        print(f"  [{time.time() - start:.1f}s] progress={progress}%, step={step}, status={status}")
        if status == "completed":
            completed = True
            break
        elif status == "failed":
            raise RuntimeError(f"Job failed: {st_data.get('error_message')}")
        time.sleep(3)

    assert completed, f"Job {job_id} did not finish in {max_wait}s"
    print(f"[PASS] Job completed in {time.time() - start:.1f}s!")

    # 4. Summary Endpoint
    print(f"\n4. GET /api/v1/jobs/{job_id}/summary")
    sum_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/summary")
    assert sum_resp.status_code == 200
    sum_data = sum_resp.json()
    print(f"  Summary: {json.dumps(sum_data, indent=2)}")
    assert sum_data["status"] == "completed"
    assert sum_data["total_wallets"] > 0
    assert sum_data["anomalies_found"] > 0
    print("[PASS] Summary endpoint validated!")

    # 5. Graph Overview Endpoint
    print(f"\n5. GET /api/v1/jobs/{job_id}/graph/overview")
    graph_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/graph/overview?max_nodes=100")
    assert graph_resp.status_code == 200
    graph_data = graph_resp.json()
    assert len(graph_data["nodes"]) > 0
    assert len(graph_data["edges"]) > 0
    assert "gnn_influence_weight" in graph_data["edges"][0]
    print(f"[PASS] Graph Overview: {len(graph_data['nodes'])} nodes, {len(graph_data['edges'])} edges")

    # 6. Ego-Graph Endpoint
    test_wallet = graph_data["nodes"][0]["id"]
    print(f"\n6. GET /api/v1/jobs/{job_id}/graph/wallet/{test_wallet}")
    ego_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/graph/wallet/{test_wallet}?hops=2")
    assert ego_resp.status_code == 200
    ego_data = ego_resp.json()
    assert ego_data["wallet_id"] == test_wallet
    assert len(ego_data["nodes"]) > 0
    print(f"[PASS] Ego-Graph: {len(ego_data['nodes'])} nodes, {len(ego_data['edges'])} edges for {test_wallet}")

    # 7. Alerts Endpoint
    print(f"\n7. GET /api/v1/jobs/{job_id}/alerts")
    alerts_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/alerts")
    assert alerts_resp.status_code == 200
    alerts_data = alerts_resp.json()
    assert len(alerts_data["alerts"]) > 0
    flagged_wallet = alerts_data["alerts"][0]["wallet_id"]
    print(f"[PASS] Alerts: {len(alerts_data['alerts'])} alerts retrieved (Top: {flagged_wallet})")

    # 8. Explainability Endpoint
    print(f"\n8. GET /api/v1/jobs/{job_id}/explainability/wallet/{flagged_wallet}")
    exp_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/explainability/wallet/{flagged_wallet}")
    assert exp_resp.status_code == 200
    exp_data = exp_resp.json()
    assert exp_data["wallet_id"] == flagged_wallet
    assert len(exp_data["top_risk_factors"]) > 0
    top_factor = exp_data["top_risk_factors"][0]
    assert "feature" in top_factor
    assert "shap_value" in top_factor
    assert "direction" in top_factor
    print(f"  Top Factor: {top_factor['feature']} (shap={top_factor['shap_value']}, dir={top_factor['direction']})")
    print(f"[PASS] Explainability endpoint validated!")

    # 9. Traceability Animations Endpoint
    print(f"\n9. GET /api/v1/jobs/{job_id}/patterns/trace/{flagged_wallet}")
    trace_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/patterns/trace/{flagged_wallet}")
    assert trace_resp.status_code == 200
    trace_data = trace_resp.json()
    assert "pattern" in trace_data
    assert "sequence" in trace_data
    print(f"  Pattern: {trace_data['pattern']} (confidence={trace_data['confidence_score']:.2f}, hops={trace_data['hop_count']})")
    print(f"[PASS] Traceability endpoint validated!")

    # 10. List Jobs
    print(f"\n10. GET /api/v1/jobs")
    jobs_resp = requests.get(f"{BASE_URL}/jobs")
    assert jobs_resp.status_code == 200
    assert len(jobs_resp.json()) > 0
    print(f"[PASS] Jobs listing returned {len(jobs_resp.json())} jobs")

    # 11. Test query param versions (?job_id=...)
    print(f"\n11. Testing Query Param versions (?job_id={job_id})")
    assert requests.get(f"{BASE_URL}/graph/overview?job_id={job_id}").status_code == 200
    assert requests.get(f"{BASE_URL}/graph/wallet/{flagged_wallet}?job_id={job_id}").status_code == 200
    assert requests.get(f"{BASE_URL}/alerts?job_id={job_id}").status_code == 200
    assert requests.get(f"{BASE_URL}/explainability/wallet/{flagged_wallet}?job_id={job_id}").status_code == 200
    assert requests.get(f"{BASE_URL}/patterns/trace/{flagged_wallet}?job_id={job_id}").status_code == 200
    print("[PASS] All query param versions (?job_id=...) passed!")

    print("\n" + "=" * 60)
    print("ALL API ENDPOINTS VALIDATED WITH 100% SUCCESS!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    test_api()
