import requests
import re
import json

headers = {"User-Agent": "Mozilla/5.0"}
js = requests.get('https://agmarknet.gov.in/static/js/main.07aebdd1.js', headers=headers, verify=False).text

targets = ["daily-price-arrival/report", "date-wise/specific-commodity", "commodity-wise/daily-report-state", "commodity-market/daily-report-state-marketwise"]

for t in targets:
    idx = js.find(t)
    if idx != -1:
        print(f"=== Target: {t} at {idx} ===")
        print(js[max(0, idx - 150): min(len(js), idx + 250)])
        print()
