import requests
import json
import urllib3
urllib3.disable_warnings()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json"
}

payload = {
    "output": "plot-output.figure",
    "inputs": [
        {"id": "url", "property": "search", "value": "?viz_req=true&viz_type=hbar&indicator_code=4"},
        {"id": "color-scale-dropdown", "property": "value", "value": "Blues"},
        {"id": "axis-scale-dropdown", "property": "value", "value": "x-y"}
    ],
    "changedPropIds": ["url.search"]
}

url = "https://esankhyiki.mospi.gov.in/viz/hces/_dash-update-component"
r = requests.post(url, json=payload, headers=headers, verify=False, timeout=15)
print("Status:", r.status_code)
try:
    data = r.json()
    fig = data.get("response", {}).get("plot-output", {}).get("figure", {})
    print("Title:", fig.get("layout", {}).get("title"))
    traces = fig.get("data", [])
    print(f"Traces count: {len(traces)}")
    for t in traces[:3]:
        print(f"Trace: {t.get('name')} | x count: {len(t.get('x', []))} | y count: {len(t.get('y', []))}")
        print("  Sample y (categories):", t.get("y", [])[:6])
        print("  Sample x (values):", t.get("x", [])[:6])
except Exception as e:
    print("Error:", e)
    print("Raw text:", r.text[:300])
