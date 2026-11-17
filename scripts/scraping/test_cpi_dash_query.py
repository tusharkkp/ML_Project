import requests
import json
import urllib3
urllib3.disable_warnings()

headers = {
    "User-Agent": "Mozilla/5.0",
    "Content-Type": "application/json"
}

# In viz/cpi, let's see what parameters are valid:
# Earlier from JS: /viz/cpi?viz_req=true&viz_type=line&baseyear=2012&level=Group
payload = {
    "output": "..plot-output.figure...table-store.data...request-store.data..",
    "inputs": [
        {"id": "url", "property": "search", "value": "?viz_req=true&viz_type=line&baseyear=2012&level=Group"},
        {"id": "color-scale-dropdown", "property": "value", "value": "Blues"},
        {"id": "axis-scale-dropdown", "property": "value", "value": "x-y"},
        {"id": "plot-output", "property": "relayoutData", "value": None}
    ],
    "changedPropIds": ["url.search"]
}

url = "https://esankhyiki.mospi.gov.in/viz/cpi/_dash-update-component"
r = requests.post(url, json=payload, headers=headers, verify=False, timeout=15)
print("Status:", r.status_code)
try:
    data = r.json()
    print("Keys in response:", list(data.keys()))
    resp = data.get("response", {})
    print("Keys in response inner:", list(resp.keys()))
    table_data = resp.get("table-store", {}).get("data")
    if table_data:
        print("Table data type:", type(table_data))
        if isinstance(table_data, list):
            print(f"Table data items: {len(table_data)}")
            print("First item:", table_data[0])
    fig = resp.get("plot-output", {}).get("figure", {})
    if fig:
        print("Figure title:", fig.get("layout", {}).get("title"))
        traces = fig.get("data", [])
        print(f"Figure traces: {len(traces)}")
        for t in traces[:3]:
            print("  Trace:", t.get("name"), "pts:", len(t.get("x", [])))
except Exception as e:
    print("Error:", e)
    print("Snippet:", r.text[:300])
