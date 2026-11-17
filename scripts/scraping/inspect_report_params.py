import requests
import re

headers = {"User-Agent": "Mozilla/5.0"}
js = requests.get('https://agmarknet.gov.in/static/js/main.07aebdd1.js', headers=headers, verify=False).text

# Search for the component that dispatches fetchReport
idx = js.find('daily-price-arrival/report')
# Search around for state_id, cmdt_id, from_date, to_date
matches = re.findall(r'[a-zA-Z0-9_]+:\s*[a-zA-Z0-9_]+\.(?:state_id|district_id|market_id|cmdt_id|from_date|to_date|variety_id|grade_id)', js)
print("Parameter mappings found in JS:", set(matches))

# Let's also check for object literals with from_date or to_date
blocks = re.findall(r'\{[^{}]*(?:from_date|to_date|cmdt_id|state_id)[^{}]*\}', js)
print(f"Blocks found: {len(blocks)}")
for b in blocks[:10]:
    print("  Block:", b)
