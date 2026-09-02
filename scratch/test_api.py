import requests

base_url = "http://127.0.0.1:8000/api/v1"
try:
    print("Fetching explainability for W0000182 (job-11194)...")
    r = requests.get(f"{base_url}/jobs/job-11194/explainability/wallet/W0000182")
    if r.status_code == 200:
        data = r.json()
        print(f"Explainability returned {len(data.get('top_risk_factors', []))} factors.")
    else:
        print(f"Explainability error: {r.status_code} {r.text}")
        
    print("\nFetching trace for W0000182 (job-11194)...")
    r = requests.get(f"{base_url}/jobs/job-11194/patterns/trace/W0000182")
    if r.status_code == 200:
        data = r.json()
        print(f"Trace returned pattern: {data.get('pattern')}, hops: {data.get('hop_count')}")
    else:
        print(f"Trace error: {r.status_code} {r.text}")

except Exception as e:
    print("Error:", e)
