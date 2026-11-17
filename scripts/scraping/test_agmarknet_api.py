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

endpoints = [
    ("GET", "https://api.agmarknet.gov.in/v1/commodities"),
    ("GET", "https://api.agmarknet.gov.in/v1/location/state"),
    ("GET", "https://api.agmarknet.gov.in/v1/daily-price-arrival/filters"),
    ("GET", "https://api.agmarknet.gov.in/v1/dashboard-default-data/"),
]

for method, url in endpoints:
    try:
        r = requests.request(method, url, headers=headers, verify=False, timeout=10)
        print(f"{method} {url} -> Status: {r.status_code}")
        print("  Snippet:", r.text[:200])
    except Exception as e:
        print(f"{method} {url} -> Error: {e}")
