"""Diagnostic: send a single request to the search API and print response details."""
import time
import json
import requests

API = "http://127.0.0.1:8068/search"
query = (
    "For 222 Rajpur, Dehradun, how many total residences are planned and over how many acres is the project spread?"
)

def main():
    payload = {"user_id": 1, "query": query, "k": 5}
    print(f"Sending POST to {API} with payload keys: {list(payload.keys())}")
    t0 = time.time()
    try:
        r = requests.post(API, json=payload, timeout=10)
    except Exception as e:
        print("Request error:", e)
        return
    t1 = time.time()
    elapsed_ms = (t1 - t0) * 1000
    print(f"Status: {r.status_code}; Elapsed: {elapsed_ms:.1f} ms")
    print("Response headers:", dict(r.headers))
    text = r.text
    print(f"Response length: {len(text)} chars")
    try:
        j = r.json()
        print("JSON keys:", list(j.keys()))
        print(json.dumps(j, indent=2)[:4000])
    except Exception:
        print("Non-JSON response (first 200 chars):")
        print(text[:200])

if __name__ == '__main__':
    main()
