import requests
import re

js = requests.get('https://esankhyiki.mospi.gov.in/static/js/main.ecf16667.js', verify=False).text
matches = set(re.findall(r'["\'](/hces/[^"\']+)["\']', js))
print("All /hces/ paths in JS:")
for m in sorted(matches):
    print(" ", m)

# Also search for indicator_code in axios calls
lines = [line for line in js.split(';') if 'getHces' in line]
print(f"Lines with getHces: {len(lines)}")
for l in lines[:5]:
    print(" ", l[:200])
