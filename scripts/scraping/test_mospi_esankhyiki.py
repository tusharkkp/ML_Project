import requests
import json
import urllib3
urllib3.disable_warnings()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://esankhyiki.mospi.gov.in",
    "Referer": "https://esankhyiki.mospi.gov.in/"
}

candidates = [
    ("GET", "https://api.mospi.gov.in/api/cpi/getCpiBaseYear"),
    ("POST", "https://api.mospi.gov.in/api/cpi/getCpiBaseYear"),
    ("GET", "https://api.mospi.gov.in/api/esankhyiki/cpi/getCpiBaseYear"),
    ("POST", "https://api.mospi.gov.in/api/esankhyiki/cpi/getCpiBaseYear"),
    ("GET", "https://esankhyiki.mospi.gov.in/api/cpi/getCpiBaseYear"),
]

for method, url in candidates:
    try:
        r = requests.request(method, url, headers=headers, verify=False, timeout=8)
        print(f"{method} {url} -> Status: {r.status_code}")
        print("  Snippet:", r.text[:200])
    except Exception as e:
        print(f"{method} {url} -> Error: {e}")
