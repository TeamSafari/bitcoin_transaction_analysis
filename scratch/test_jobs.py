import requests

base_url = "http://127.0.0.1:8000/api/v1"
try:
    r = requests.get(f"{base_url}/jobs")
    if r.status_code == 200:
        jobs = r.json()
        print(f"Found {len(jobs)} jobs.")
        for j in jobs:
            print(f"Job: {j['job_id']}, Status: {j['status']}")
            
            # Fetch for first job
            if j['status'] == 'completed':
                job_id = j['job_id']
                print(f"Fetching W001 for {job_id}...")
                
                req_shap = requests.get(f"{base_url}/jobs/{job_id}/explainability/wallet/W001")
                if req_shap.status_code == 200:
                    data = req_shap.json()
                    print(f"SHAP: {len(data.get('top_risk_factors', []))} factors.")
                else:
                    print(f"SHAP error: {req_shap.status_code} {req_shap.text}")
                    
                req_trace = requests.get(f"{base_url}/jobs/{job_id}/patterns/trace/W001")
                if req_trace.status_code == 200:
                    data = req_trace.json()
                    print(f"Trace: pattern={data.get('pattern')}, hops={data.get('hop_count')}")
                else:
                    print(f"Trace error: {req_trace.status_code} {req_trace.text}")
    else:
        print(f"Error fetching jobs: {r.status_code} {r.text}")

except Exception as e:
    print("Error:", e)
