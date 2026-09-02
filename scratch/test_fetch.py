import urllib.request
import json
import traceback

def fetch(url):
    try:
        print(f"Fetching {url}")
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
            return data
    except urllib.error.HTTPError as e:
        print(f"HTTPError: {e.code} - {e.read().decode()}")
    except Exception as e:
        print(f"Error: {e}")

job_id = "job-ad989"
wallet_id = "W001"
url1 = f"http://127.0.0.1:8000/api/v1/jobs/{job_id}/explainability/wallet/{wallet_id}"
url2 = f"http://127.0.0.1:8000/api/v1/jobs/{job_id}/patterns/trace/{wallet_id}"

print("Explainability:")
res1 = fetch(url1)
if res1:
    print(f"Got {len(res1.get('top_risk_factors', []))} factors")

print("\nTrace:")
res2 = fetch(url2)
if res2:
    print(f"Got pattern {res2.get('pattern')} with {res2.get('hop_count')} hops")
