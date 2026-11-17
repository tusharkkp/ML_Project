import requests
from bs4 import BeautifulSoup
import re
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://rbi.org.in/Scripts/PublicationsView.aspx?id=23348"
r = requests.get(url, headers=headers, verify=False, timeout=15)
print("Status:", r.status_code, "Length:", len(r.content))
soup = BeautifulSoup(r.text, 'html.parser')
links = soup.find_all('a')
print(f"Total links: {len(links)}")
for l in links:
    href = l.get('href', '')
    text = l.get_text(strip=True)
    if any(k in (href + text).lower() for k in ['cpi', 'price', 'table', 'xlsx', 'xls', '163']):
        print(f"  Link: {text} -> {href}")
