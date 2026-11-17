import requests
import re
import json
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://esankhyiki.mospi.gov.in/viz/cpi"
r = requests.get(url, headers=headers, verify=False, timeout=15)
print("viz/cpi status:", r.status_code)
print("Length:", len(r.content))
# Let's search inside the HTML
scripts = re.findall(r'<script[^>]+src=[\'"]([^\'"]+)[\'"]', r.text)
print("Scripts on viz/cpi:", scripts)
