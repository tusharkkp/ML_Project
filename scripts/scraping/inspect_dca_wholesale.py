import requests
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
candidates = [
    "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx",
    "https://fcainfoweb.nic.in/Reports/DB/DBprices_wholesale.aspx",
    "https://fcainfoweb.nic.in/Reports/DB/DBpricesWholesale.aspx",
    "https://fcainfoweb.nic.in/Reports/DB/DB_wholesale_prices.aspx",
    "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx?type=wholesale",
    "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx?type=ws",
    "https://fcainfoweb.nic.in/Reports/DB/DB_prices_ws.aspx"
]

for url in candidates:
    try:
        r = requests.get(url, headers=headers, verify=False, timeout=8)
        print(f"{url} -> {r.status_code} (len: {len(r.content)})")
        if r.status_code == 200 and 'wholesale' in r.text.lower():
            print("  Found 'wholesale' in response!")
    except Exception as e:
        print(f"{url} -> Error: {e}")
