import requests
import re
import json
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def inspect_agmarknet_bundles():
    print("=== INSPECTING AGMARKNET 2.0 FRONTEND BUNDLES ===")
    r = requests.get("https://agmarknet.gov.in/", headers=headers, verify=False, timeout=15)
    html = r.text
    print("Agmarknet HTML length:", len(html))
    # Find script src
    scripts = re.findall(r'<script[^>]+src=[\'"]([^\'"]+)[\'"]', html)
    print("Scripts found:", scripts)
    
    # Check if there are JS chunks
    for s in scripts:
        if s.startswith('/'):
            js_url = f"https://agmarknet.gov.in{s}"
        elif s.startswith('http'):
            js_url = s
        else:
            js_url = f"https://agmarknet.gov.in/{s}"
        try:
            js_resp = requests.get(js_url, headers=headers, verify=False, timeout=10)
            print(f"Fetched {js_url}: {len(js_resp.text)} chars")
            # Look for API endpoints in JS
            api_calls = re.findall(r'[\'"](/api/[^\'"]+|https?://[^\'"]+/api/[^\'"]+)[\'"]', js_resp.text)
            if api_calls:
                print(f"  API endpoints found in {s}:", list(set(api_calls))[:10])
        except Exception as e:
            print(f"Error fetching {js_url}: {e}")

if __name__ == "__main__":
    inspect_agmarknet_bundles()
