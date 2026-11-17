import requests
import re

headers = {"User-Agent": "Mozilla/5.0"}
js = requests.get('https://agmarknet.gov.in/static/js/main.07aebdd1.js', headers=headers, verify=False).text

# Look for axios calls: .get("..."), .post("...")
calls = re.findall(r'\.(?:get|post)\(["\']([a-zA-Z0-9_\-\/]+)["\']', js)
print("Axios endpoint calls in Agmarknet 2.0 bundle:")
for c in sorted(set(calls)):
    print(" ", c)
