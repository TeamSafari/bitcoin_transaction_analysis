import requests

base_url = "http://127.0.0.1:8000/api/v1"
try:
    job_id = "job-ad989"
    
    # Get all nodes from graph overview
    r = requests.get(f"{base_url}/jobs/{job_id}/graph/overview")
    nodes = r.json().get("nodes", [])
    print(f"Found {len(nodes)} nodes.")
    
    zero_shap = 0
    zero_trace = 0
    for n in nodes:
        wid = n["id"]
        req_shap = requests.get(f"{base_url}/jobs/{job_id}/explainability/wallet/{wid}")
        if req_shap.status_code == 200:
            data = req_shap.json()
            if len(data.get('top_risk_factors', [])) == 0:
                zero_shap += 1
        
        req_trace = requests.get(f"{base_url}/jobs/{job_id}/patterns/trace/{wid}")
        if req_trace.status_code == 200:
            data = req_trace.json()
            if data.get('hop_count', 0) == 0:
                zero_trace += 1
                
    print(f"Nodes with 0 SHAP: {zero_shap}/{len(nodes)}")
    print(f"Nodes with 0 trace hops: {zero_trace}/{len(nodes)}")

except Exception as e:
    print("Error:", e)
