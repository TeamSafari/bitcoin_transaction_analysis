import requests

base_url = "http://127.0.0.1:8000/api/v1"
job_id = "job-d4745"
wallet_id = "W000001"

print(f"Fetching {wallet_id} for {job_id}...")
try:
    req_shap = requests.get(f"{base_url}/jobs/{job_id}/explainability/wallet/{wallet_id}")
    if req_shap.status_code == 200:
        data = req_shap.json()
        print(f"SHAP: {len(data.get('top_risk_factors', []))} factors.")
    else:
        print(f"SHAP error: {req_shap.status_code}")
        
    req_trace = requests.get(f"{base_url}/jobs/{job_id}/patterns/trace/{wallet_id}")
    if req_trace.status_code == 200:
        data = req_trace.json()
        print(f"Trace: pattern={data.get('pattern')}, hops={data.get('hop_count')}")
    else:
        print(f"Trace error: {req_trace.status_code}")
except Exception as e:
    print("Error:", e)
