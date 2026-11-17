import requests
import json
import urllib3
urllib3.disable_warnings()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Origin": "https://esankhyiki.mospi.gov.in",
    "Content-Type": "application/json"
}

# Test indicator 4 (% share) and 5 (quantity consumed)
for code in [1, 3, 4, 5]:
    try:
        r = requests.post(f"https://api.mospi.gov.in/api/hces/getHcesFilterByIndicatorId", json={"indicator_code": code}, headers=headers, verify=False, timeout=8)
        print(f"Filter for code {code}: {r.status_code}")
        if r.status_code == 200:
            print(" ", json.dumps(r.json(), indent=2)[:300])
    except Exception as e:
        print(f"Error {code}: {e}")
