import requests
from bs4 import BeautifulSoup
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://esankhyiki.mospi.gov.in/catalogue-main/catalogue?product=CPI"
r = requests.get(url, headers=headers, verify=False, timeout=15)
soup = BeautifulSoup(r.text, 'html.parser')
scripts = [s.get('src') for s in soup.find_all('script') if s.get('src')]
print("eSankhyiki scripts:")
for s in scripts:
    print(" ", s)
