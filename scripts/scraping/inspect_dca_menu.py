import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get('https://fcainfoweb.nic.in/reports/report_menu_web.aspx', headers=headers, verify=False, timeout=20)
soup = BeautifulSoup(r.text, 'html.parser')

print("Title:", soup.title.string.strip() if soup.title else "No title")
dropdowns = soup.find_all('select')
for d in dropdowns:
    print(f"Dropdown: name='{d.get('name')}', id='{d.get('id')}'")
    options = [(o.get('value'), o.get_text(strip=True)) for o in d.find_all('option')]
    print(f"  Options ({len(options)}): {options[:10]}")

inputs = soup.find_all('input')
for i in inputs:
    print(f"Input: name='{i.get('name')}', type='{i.get('type')}', value='{i.get('value')}'")
