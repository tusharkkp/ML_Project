import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://fcainfoweb.nic.in/Reports/DB/DBprices_wholesale.aspx"
r = requests.get(url, headers=headers, verify=False, timeout=15)
print("Status:", r.status_code)
print("Length:", len(r.content))
soup = BeautifulSoup(r.text, 'html.parser')
tables = soup.find_all('table')
print(f"Tables: {len(tables)}")
for i, t in enumerate(tables):
    rows = t.find_all('tr')
    print(f"Table {i} has {len(rows)} rows")
    for row in rows[:5]:
        print(" ", [c.get_text(strip=True) for c in row.find_all(['th', 'td'])])
if len(tables) == 0:
    print("Body snippet:", r.text[:500])
