import requests
import re
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://esankhyiki.mospi.gov.in/static/js/main.ecf16667.js"
r = requests.get(url, headers=headers, verify=False, timeout=15)
print("JS length:", len(r.text))

# Search for API endpoints
urls = set(re.findall(r'https?://[a-zA-Z0-9_\-\.:]+/[a-zA-Z0-9_\-/\.]+', r.text))
print("Full URLs found:")
for u in sorted(urls):
    if any(k in u.lower() for k in ['api', 'cpi', 'mospi', 'catalogue', 'download']):
        print(" ", u)

paths = set(re.findall(r'["\'](/api/[a-zA-Z0-9_\-/\.]+)["\']', r.text))
print("Relative /api/ paths found:")
for p in sorted(paths):
    print(" ", p)
