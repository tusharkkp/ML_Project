import requests
import json
import urllib3
urllib3.disable_warnings()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json"
}

url = "https://api.mospi.gov.in/api/getCPIIndex"
payload = {
    "Series": "Current_series_2012",
    "Year": "2024",
    "Month": "1",
    "Group_code": "1",    # Food and beverages
    "Sector": "3",        # Combined
    "Format": "JSON"
}

print("Testing MoSPI getCPIIndex...")
try:
    r = requests.post(url, json=payload, headers=headers, verify=False, timeout=10)
    print("Status code:", r.status_code)
    print("Content-Type:", r.headers.get("Content-Type"))
    print("Snippet:", r.text[:300])
except Exception as e:
    print("Error:", e)
