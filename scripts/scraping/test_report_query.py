import requests
import json
import urllib3
urllib3.disable_warnings()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://agmarknet.gov.in",
    "Referer": "https://agmarknet.gov.in/",
    "Content-Type": "application/json"
}

payload = {
    "state": [20],       # Maharashtra
    "district": [364],   # Pune
    "commodity": [23],   # Onion
    "market": [],
    "grade": [],
    "from_date": "2026-11-01",
    "to_date": "2026-11-17",
    "type": "100004",    # Price
    "page": 1,
    "limit": 5
}

url = "https://api.agmarknet.gov.in/v1/daily-price-arrival/report"
print("Sending test request to Agmarknet 2.0 report API...")
r = requests.post(url, json=payload, headers=headers, verify=False, timeout=15)
print("Status code:", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
try:
    data = r.json()
    print("Response JSON keys:", list(data.keys()))
    print("Status in response:", data.get("status"))
    print("Message in response:", data.get("message"))
    d = data.get("data", {})
    if isinstance(d, dict):
        print("Data keys:", list(d.keys()))
        records = d.get("records", [])
        print(f"Total records returned: {len(records)}")
        if records:
            print("First record sample:")
            print(json.dumps(records[0], indent=2))
    elif isinstance(d, list):
        print(f"Data is list of {len(d)} items")
        if d:
            print("First item sample:", json.dumps(d[0], indent=2))
except Exception as e:
    print("Failed to parse JSON:", e)
    print("Response text:", r.text[:500])
