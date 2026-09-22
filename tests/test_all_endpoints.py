"""Integration tests for all documented API endpoints."""

from __future__ import annotations

import sys
from pathlib import Path
import threading
import time

import requests
import uvicorn

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.main import create_app

PROJECT_ROOT = _ROOT
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TEST_PORT = 8799
BASE_URL = f"http://127.0.0.1:{TEST_PORT}/api/v1"


def run_test_server():
    app = create_app()
    server = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    )
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(2)
    return server


def test_api():
    run_test_server()
    print("\n" + "=" * 60)
    print("TESTING API ENDPOINTS")
    print("=" * 60)

    assert requests.get(f"{BASE_URL}/health").status_code == 200
    print("[PASS] GET /api/v1/health")

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

    assert upload_resp.status_code == 200, upload_resp.text
    job_id = upload_resp.json()["job_id"]
    print(f"[PASS] POST /api/v1/jobs/upload -> {job_id}")

    max_wait = 180
    start = time.time()
    while time.time() - start < max_wait:
        st_data = requests.get(f"{BASE_URL}/jobs/{job_id}/status").json()
        if st_data["status"] == "completed":
            break
        if st_data["status"] == "failed":
            raise RuntimeError(st_data.get("error_message"))
        time.sleep(3)
    else:
        raise AssertionError(f"Job {job_id} did not finish in {max_wait}s")
    print(f"[PASS] Job completed in {time.time() - start:.1f}s")

    sum_data = requests.get(f"{BASE_URL}/jobs/{job_id}/summary").json()
    assert sum_data["status"] == "completed" and sum_data["total_wallets"] > 0
    print("[PASS] GET /api/v1/jobs/{id}/summary")

    graph_data = requests.get(f"{BASE_URL}/jobs/{job_id}/graph/overview?max_nodes=100").json()
    assert graph_data["nodes"] and graph_data["edges"]
    test_wallet = graph_data["nodes"][0]["id"]
    print("[PASS] GET /api/v1/jobs/{id}/graph/overview")

    ego_data = requests.get(f"{BASE_URL}/jobs/{job_id}/graph/wallet/{test_wallet}").json()
    assert ego_data["wallet_id"] == test_wallet
    print("[PASS] GET /api/v1/jobs/{id}/graph/wallet/{wallet_id}")

    alerts_data = requests.get(f"{BASE_URL}/jobs/{job_id}/alerts").json()
    flagged_wallet = alerts_data["alerts"][0]["wallet_id"]
    print("[PASS] GET /api/v1/jobs/{id}/alerts")

    exp_data = requests.get(f"{BASE_URL}/jobs/{job_id}/explainability/wallet/{flagged_wallet}").json()
    assert exp_data["top_risk_factors"]
    print("[PASS] GET /api/v1/jobs/{id}/explainability/wallet/{wallet_id}")

    trace_data = requests.get(f"{BASE_URL}/jobs/{job_id}/patterns/trace/{flagged_wallet}").json()
    assert "pattern" in trace_data and "sequence" in trace_data
    print("[PASS] GET /api/v1/jobs/{id}/patterns/trace/{wallet_id}")

    assert requests.get(f"{BASE_URL}/jobs").status_code == 200
    assert requests.get(f"{BASE_URL}/graph/overview?job_id={job_id}").status_code == 200
    print("[PASS] Global query-param routes")

    print("\nALL API ENDPOINTS VALIDATED\n")


if __name__ == "__main__":
    test_api()
