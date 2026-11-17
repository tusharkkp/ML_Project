import requests
import re
import json
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://api.mospi.gov.in/static/js/main.addddd18.js"
r = requests.get(url, headers=headers, verify=False, timeout=15)
print("JS length:", len(r.text))

# Search for Swagger / OpenAPI or URL paths
urls = set(re.findall(r'https?://[a-zA-Z0-9_\-\.:]+/[a-zA-Z0-9_\-/\.]+', r.text))
print("Full URLs found:")
for u in sorted(urls):
    print(" ", u)

paths = set(re.findall(r'["\'](/[a-zA-Z0-9_\-/\.]+\.[a-zA-Z0-9]+|/api/[a-zA-Z0-9_\-/\.]+)["\']', r.text))
print("\nPaths found:")
for p in sorted(paths):
    print(" ", p)
