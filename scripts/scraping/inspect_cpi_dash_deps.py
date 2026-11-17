import requests
import json
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
deps = requests.get('https://esankhyiki.mospi.gov.in/viz/cpi/_dash-dependencies', headers=headers, verify=False).json()
print(f"Total callbacks in /viz/cpi: {len(deps)}")
for i, d in enumerate(deps):
    print(f"Callback {i}:")
    print("  Output:", d.get('output'))
    print("  Inputs:", [(inp.get('id'), inp.get('property')) for inp in d.get('inputs', [])])
    print("  State:", [(s.get('id'), s.get('property')) for s in d.get('state', [])])
