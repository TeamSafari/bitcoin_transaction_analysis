"""
End-to-End Test Suite for Orchestrator, SQLite Database, and Jobs API
Using live test server and requests.
"""

from __future__ import annotations

import json
from pathlib import Path
import threading
import time
import requests
import uvicorn

from backend.api_server import create_app
from backend.database import init_db, create_job, update_job, get_job, list_jobs
from backend.pipeline.orchestrator import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
TEST_PORT = 8765
BASE_URL = f"http://127.0.0.1:{TEST_PORT}/api/v1"


def test_database():
    print("\n" + "=" * 50)
    print("1. TESTING DATABASE LAYER")
    print("=" * 50)
    test_db_path = PROJECT_ROOT / "outputs" / "test_jobs.db"
    if test_db_path.exists():
        test_db_path.unlink()

    init_db(test_db_path)
    job = create_job(
        job_id="job-test1",
        raw_dir="/tmp/raw",
        output_dir="/tmp/out",
        db_path=test_db_path,
    )
    assert job["job_id"] == "job-test1"
    assert job["status"] == "processing"

    update_job(
        job_id="job-test1",
        status="completed",
        current_stage="completed",
        summary={"test": True},
        db_path=test_db_path,
    )

    retrieved = get_job("job-test1", db_path=test_db_path)
    assert retrieved is not None
    assert retrieved["status"] == "completed"
    assert retrieved["summary"] == {"test": True}

    all_jobs = list_jobs(db_path=test_db_path)
    assert len(all_jobs) == 1
    assert all_jobs[0]["job_id"] == "job-test1"

    print("[PASS] Database Layer tests passed!")


def test_jobs_upload_api():
    print("\n" + "=" * 50)
    print("2. TESTING /api/v1/jobs/upload API")
    print("=" * 50)

    # 1. Test missing files error (should return 400)
    resp = requests.post(f"{BASE_URL}/jobs/upload", files={})
    assert resp.status_code == 400
    assert "Missing required CSV files" in resp.json()["detail"]
    print("[PASS] Missing file validation test passed!")

    # 2. Test successful 6 CSV files upload
    files_payload = {
        "wallets": ("wallets.csv", open(RAW_DIR / "wallets.csv", "rb"), "text/csv"),
        "transactions": (
            "transactions.csv",
            open(RAW_DIR / "transactions.csv", "rb"),
            "text/csv",
        ),
        "transaction_inputs": (
            "transaction_inputs.csv",
            open(RAW_DIR / "transaction_inputs.csv", "rb"),
            "text/csv",
        ),
        "transaction_outputs": (
            "transaction_outputs.csv",
            open(RAW_DIR / "transaction_outputs.csv", "rb"),
            "text/csv",
        ),
        "network_observations": (
            "network_observations.csv",
            open(RAW_DIR / "network_observations.csv", "rb"),
            "text/csv",
        ),
        "ip_metadata": (
            "ip_metadata.csv",
            open(RAW_DIR / "ip_metadata.csv", "rb"),
            "text/csv",
        ),
    }

    upload_resp = requests.post(f"{BASE_URL}/jobs/upload", files=files_payload)
    for _, f_tuple in files_payload.items():
        f_tuple[1].close()

    assert upload_resp.status_code == 200, f"Upload failed: {upload_resp.text}"
    data = upload_resp.json()
    job_id = data["job_id"]
    assert job_id.startswith("job-")
    assert data["status"] == "processing"
    print(f"[PASS] Job uploaded successfully! Assigned job_id: {job_id}")
    print(f"  Response: {json.dumps(data, indent=2)}")

    # 3. Poll job status
    print(f"\nPolling status for {job_id}...")
    max_wait = 90
    start = time.time()
    completed = False
    while time.time() - start < max_wait:
        st_resp = requests.get(f"{BASE_URL}/jobs/{job_id}")
        assert st_resp.status_code == 200
        st_data = st_resp.json()
        status = st_data["status"]
        current_stage = st_data.get("current_stage")
        print(f"  [{time.time() - start:.1f}s] Status: {status}, Stage: {current_stage}")
        if status == "completed":
            completed = True
            break
        elif status == "failed":
            raise RuntimeError(f"Job failed: {st_data.get('error_message')}")
        time.sleep(3)

    assert completed, f"Job did not complete within {max_wait}s"
    print(f"[PASS] Job completed successfully in {time.time() - start:.1f}s!")

    # 4. Check alerts endpoint
    alerts_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/alerts")
    assert alerts_resp.status_code == 200
    alerts_data = alerts_resp.json()
    assert alerts_data["status"] == "completed"
    assert len(alerts_data["alerts"]) > 0
    high_count = sum(1 for a in alerts_data["alerts"] if a["severity"] == "HIGH")
    print(f"[PASS] Retrieved {len(alerts_data['alerts'])} alerts ({high_count} HIGH) for {job_id}")

    # 5. Check manifest endpoint
    manifest_resp = requests.get(f"{BASE_URL}/jobs/{job_id}/manifest")
    assert manifest_resp.status_code == 200
    manifest_data = manifest_resp.json()
    assert manifest_data["status"] == "success"
    print(f"[PASS] Retrieved execution manifest for {job_id}")

    # 6. Check list jobs
    jobs_resp = requests.get(f"{BASE_URL}/jobs")
    assert jobs_resp.status_code == 200
    jobs_list = jobs_resp.json()
    assert any(j["job_id"] == job_id for j in jobs_list)
    print(f"[PASS] Jobs list returned {len(jobs_list)} total jobs")


def main():
    app = create_app()
    server_config = uvicorn.Config(app, host="127.0.0.1", port=TEST_PORT, log_level="warning")
    server = uvicorn.Server(server_config)

    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()

    # Wait for server to be ready
    for _ in range(20):
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                print("Server is ready on port", TEST_PORT)
                break
        except Exception:
            time.sleep(0.5)

    try:
        test_database()
        test_jobs_upload_api()
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED WITH 100% SUCCESS!")
        print("=" * 50)
    finally:
        server.should_exit = True


if __name__ == "__main__":
    main()
