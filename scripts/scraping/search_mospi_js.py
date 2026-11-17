import requests
import re

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://api.mospi.gov.in/static/js/main.addddd18.js"
js = requests.get(url, headers=headers, verify=False, timeout=15).text

for word in ["swagger", "openapi", "getCPIIndex", "usersignup", "User Manual", ".json", "spec"]:
    matches = [m.start() for m in re.finditer(word, js, re.IGNORECASE)]
    print(f"Word '{word}': {len(matches)} matches")
    for idx in matches[:3]:
        print("  Snippet:", js[max(0, idx - 50): min(len(js), idx + 100)])
