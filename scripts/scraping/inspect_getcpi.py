import requests
import re
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://esankhyiki.mospi.gov.in/static/js/main.ecf16667.js"
r = requests.get(url, headers=headers, verify=False, timeout=15)
matches = [m.start() for m in re.finditer(r'getCpi', r.text)]
print(f"Occurrences of getCpi: {len(matches)}")
for idx in matches[:5]:
    print(r.text[max(0, idx - 100): min(len(r.text), idx + 200)])
    print("-" * 50)
