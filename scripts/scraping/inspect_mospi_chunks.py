import requests
import json
import re
import urllib3
urllib3.disable_warnings()

manifest = requests.get('https://api.mospi.gov.in/asset-manifest.json', verify=False).json()
for name, path in manifest.get('files', {}).items():
    if path.endswith('.js'):
        url = f"https://api.mospi.gov.in{path}"
        r = requests.get(url, verify=False)
        for term in ['getCPIIndex', 'usersignup', 'login', 'CPI', 'swagger']:
            if term.lower() in r.text.lower():
                print(f"Found '{term}' in {name} (matches: {len(re.findall(term, r.text, re.IGNORECASE))})")
                matches = [m.start() for m in re.finditer(term, r.text, re.IGNORECASE)]
                for idx in matches[:2]:
                    print("  Snippet:", r.text[max(0, idx - 40): min(len(r.text), idx + 80)].replace('\n', ' '))
