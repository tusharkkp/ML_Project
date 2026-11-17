import requests
import re

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://esankhyiki.mospi.gov.in/static/js/main.ecf16667.js"
r = requests.get(url, headers=headers, verify=False, timeout=15)
js = r.text

cpi_endpoints = set(re.findall(r'["\'](/cpi/[a-zA-Z0-9_\-]+)["\']', js))
print("CPI endpoints found in eSankhyiki bundle:")
for e in sorted(cpi_endpoints):
    print(" ", e)

hces_endpoints = set(re.findall(r'["\'](/hces/[a-zA-Z0-9_\-]+)["\']', js))
print("\nHCES endpoints found in eSankhyiki bundle:")
for e in sorted(hces_endpoints):
    print(" ", e)
