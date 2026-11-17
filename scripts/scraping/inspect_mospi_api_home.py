import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
r = requests.get('https://api.mospi.gov.in/', headers=headers, verify=False, timeout=15)
print("Status:", r.status_code, "Length:", len(r.content))
soup = BeautifulSoup(r.text, 'html.parser')
print("Title:", soup.title.string.strip() if soup.title else "No title")
for a in soup.find_all('a'):
    print("Link:", a.get_text(strip=True), "->", a.get('href'))
for btn in soup.find_all('button'):
    print("Button:", btn.get_text(strip=True))
for f in soup.find_all('form'):
    print("Form:", f.get('action'), [i.get('name') for i in f.find_all(['input', 'button'])])
