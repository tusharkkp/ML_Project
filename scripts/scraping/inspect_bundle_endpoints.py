import requests
import re
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
js = requests.get('https://agmarknet.gov.in/static/js/main.07aebdd1.js', headers=headers, verify=False).text

matches = [m.start() for m in re.finditer(r'api\.agmarknet\.gov\.in', js)]
print(f"Matches for api.agmarknet.gov.in: {len(matches)}")
for idx in matches:
    start = max(0, idx - 100)
    end = min(len(js), idx + 200)
    print(f"Context at {idx}:")
    print(js[start:end])
    print("-" * 50)

# Also search for endpoint strings like "commodity", "market", "prices", "state"
terms = re.findall(r'["\'](?:/v1/|[a-zA-Z0-9_\-]+/)[a-zA-Z0-9_\-]+["\']', js)
print("Some API path segments:")
print(set([t for t in terms if any(k in t.lower() for k in ['commodity', 'market', 'price', 'arrival', 'state', 'district', 'report'])])[:20])
