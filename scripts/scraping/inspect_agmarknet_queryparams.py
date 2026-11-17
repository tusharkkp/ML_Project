import requests
import re

headers = {"User-Agent": "Mozilla/5.0"}
js = requests.get('https://agmarknet.gov.in/static/js/main.07aebdd1.js', headers=headers, verify=False).text

matches = [m.start() for m in re.finditer(r'fetchReport\(\{queryParams:', js)]
print(f"Matches for queryParams: {len(matches)}")
for idx in matches:
    print(js[max(0, idx - 100): min(len(js), idx + 400)])
    print("-" * 50)
