import requests
import json
import urllib3
urllib3.disable_warnings()

params = {
    'base_year': '2012',
    'series': 'Current',
    'year': '2024',
    'month_code': '1',
    'state_code': '27',
    'group_code': '1',
    'sector_code': '3',
    'page': '1',
    'Format': 'JSON'
}

r = requests.get('https://api.mospi.gov.in/api/cpi/getCPIIndex', params=params, verify=False)
print("Status:", r.status_code)
data = r.json()
print("Keys:", list(data.keys()))
print("Pagination / Extra keys:", {k: v for k, v in data.items() if k != 'data'})
records = data.get('data', [])
print(f"Data length: {len(records)}")
for item in records:
    print(" ", item.get('subgroup'), "| Index:", item.get('index'), "| Inflation:", item.get('inflation'))
