import requests
import re
import urllib3
urllib3.disable_warnings()

manifest = requests.get('https://api.mospi.gov.in/asset-manifest.json', verify=False).json()
for name, path in manifest.get('files', {}).items():
    if path.endswith('.js'):
        url = f"https://api.mospi.gov.in{path}"
        r = requests.get(url, verify=False)
        yamls = re.findall(r'["\']([^"\']+\.yaml)["\']', r.text)
        if yamls:
            print(f"YAMLs in {name}:", yamls)
            for y in yamls:
                # Try downloading it
                y_clean = y.lstrip('./').lstrip('/')
                y_url = f"https://api.mospi.gov.in/{y_clean}"
                y_resp = requests.get(y_url, verify=False)
                print(f"  {y_url} -> {y_resp.status_code} ({len(y_resp.content)} bytes)")
                if y_resp.status_code == 200:
                    fname = y_clean.replace('/', '_')
                    with open(f"data/metadata/{fname}", "wb") as f:
                        f.write(y_resp.content)
                    print(f"  Saved to data/metadata/{fname}")
