import requests
import json

url = "https://api.worldbank.org/v2/country/IND/indicator/FP.CPI.TOTL.ZG?format=json&per_page=30"
r = requests.get(url, timeout=10)
print("World Bank Status:", r.status_code)
data = r.json()
if len(data) > 1 and data[1]:
    print("Indicator:", data[1][0]['indicator']['value'])
    for obs in data[1][:10]:
        print(f"Year: {obs['date']} -> Inflation: {obs['value']}%")
