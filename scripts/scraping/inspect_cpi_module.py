import requests
import re
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://esankhyiki.mospi.gov.in/static/js/main.ecf16667.js"
r = requests.get(url, headers=headers, verify=False, timeout=15)
js = r.text

idx = js.find('getCpiBaseYear')
if idx != -1:
    print(js[max(0, idx - 200): min(len(js), idx + 800)])
