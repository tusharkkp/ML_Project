import requests
import re
import json
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get('https://api.ceda.ashoka.edu.in/documentation/', headers=headers, verify=False)
print('Page status:', r.status_code)
# Look for swagger / openapi json url
matches = re.findall(r'url:\s*["\']([^"\']+)["\']', r.text)
print('Swagger config url matches:', matches)

# Also check common swagger json paths
for path in ['swagger.json', 'openapi.json', 'v1/swagger.json', 'api/v1/swagger.json', 'api-docs']:
    url = f'https://api.ceda.ashoka.edu.in/{path}'
    try:
        resp = requests.get(url, headers=headers, verify=False, timeout=5)
        print(f'{url} -> status {resp.status_code}')
        if resp.status_code == 200:
            try:
                data = resp.json()
                print('Found JSON spec! Title:', data.get('info', {}).get('title'))
                print('Paths:', list(data.get('paths', {}).keys())[:15])
                with open('data/metadata/ceda_openapi_spec.json', 'w') as f:
                    json.dump(data, f, indent=2)
                break
            except Exception as e:
                print('Could not parse as json:', e)
    except Exception as e:
        print(f'{url} error: {e}')
