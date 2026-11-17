import requests
import re
import json

headers = {"User-Agent": "Mozilla/5.0"}
js = requests.get('https://agmarknet.gov.in/static/js/main.07aebdd1.js', headers=headers, verify=False).text

idx = js.find("priceArrivalReport/fetchReport")
if idx != -1:
    print("=== Context around fetchReport ===")
    print(js[max(0, idx - 300): min(len(js), idx + 600)])
