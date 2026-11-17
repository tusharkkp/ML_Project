import requests
import json
import urllib3
urllib3.disable_warnings()

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Origin": "https://agmarknet.gov.in",
    "Referer": "https://agmarknet.gov.in/"
}
r = requests.get('https://api.agmarknet.gov.in/v1/daily-price-arrival/filters', headers=headers, verify=False, timeout=10)
print("Status:", r.status_code)
data = r.json()
print('Filter keys:', list(data.get('data', {}).keys()))
for k, v in data.get('data', {}).items():
    if isinstance(v, list):
        print(f'  {k}: {len(v)} items (sample: {v[0] if v else None})')

# Let's save the filters to json for reference
with open('data/metadata/agmarknet_filters.json', 'w', encoding='utf-8') as f:
    json.dump(data.get('data', {}), f, indent=2)
print("Saved agmarknet_filters.json")
