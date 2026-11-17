import requests
from bs4 import BeautifulSoup
import json
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def inspect_ceda_agri():
    print("--- Inspecting CEDA Agmarknet Frontend ---")
    url = "https://agmarknet.ceda.ashoka.edu.in/"
    try:
        r = requests.get(url, headers=headers, verify=False, timeout=10)
        print("Status:", r.status_code)
        soup = BeautifulSoup(r.text, 'html.parser')
        scripts = [s.get('src') for s in soup.find_all('script') if s.get('src')]
        print("Scripts count:", len(scripts))
        for s in scripts[:5]:
            print("  Script:", s)
    except Exception as e:
        print("Error:", e)

def inspect_dca():
    print("\n--- Inspecting DCA Price Monitoring (fcainfoweb.nic.in) ---")
    urls = [
        "https://fcainfoweb.nic.in/reports/report_menu_web.aspx",
        "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx",
        "https://fcainfoweb.nic.in/Reports/Report_Menu_web.aspx"
    ]
    for u in urls:
        try:
            r = requests.get(u, headers=headers, verify=False, timeout=20)
            print(f"{u} -> Status: {r.status_code}, Length: {len(r.content)}")
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                forms = soup.find_all('form')
                print(f"  Forms found: {len(forms)}")
                for f in forms:
                    print("  Form action:", f.get('action'))
                    inputs = [i.get('name') for i in f.find_all(['input', 'select'])]
                    print(f"  Inputs: {inputs[:10]}")
                tables = soup.find_all('table')
                print(f"  Tables found: {len(tables)}")
        except Exception as e:
            print(f"{u} -> Error: {e}")

def inspect_agmarknet():
    print("\n--- Inspecting AGMARKNET (agmarknet.gov.in) ---")
    urls = [
        "https://agmarknet.gov.in/PriceTrends/SearchPriceTrends.aspx",
        "https://agmarknet.gov.in/SearchCmmMkt.aspx"
    ]
    for u in urls:
        try:
            r = requests.get(u, headers=headers, verify=False, timeout=20)
            print(f"{u} -> Status: {r.status_code}, Length: {len(r.content)}")
            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                print("  Title:", soup.title.string.strip() if soup.title else "No title")
                forms = soup.find_all('form')
                print(f"  Forms: {len(forms)}")
                selects = [s.get('name') or s.get('id') for s in soup.find_all('select')]
                print(f"  Select dropdowns: {selects}")
        except Exception as e:
            print(f"{u} -> Error: {e}")

def inspect_mospi_api():
    print("\n--- Inspecting MoSPI API ---")
    urls = [
        "https://api.mospi.gov.in/API/CPI%20API%20User%20Manual.pdf",
        "https://api.mospi.gov.in/cpi",
        "https://esankhyiki.mospi.gov.in/catalogue-main/catalogue?product=CPI"
    ]
    for u in urls:
        try:
            r = requests.get(u, headers=headers, verify=False, timeout=20)
            print(f"{u} -> Status: {r.status_code}, Length: {len(r.content)}, Content-Type: {r.headers.get('Content-Type')}")
        except Exception as e:
            print(f"{u} -> Error: {e}")

if __name__ == "__main__":
    inspect_ceda_agri()
    inspect_dca()
    inspect_agmarknet()
    inspect_mospi_api()
