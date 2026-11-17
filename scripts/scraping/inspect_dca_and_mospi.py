import requests
from bs4 import BeautifulSoup
import pypdf
import json
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

def inspect_dca_dbprices():
    print("=== INSPECTING DCA DBprices.aspx ===")
    url = "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx"
    r = requests.get(url, headers=headers, verify=False, timeout=25)
    soup = BeautifulSoup(r.text, 'html.parser')
    tables = soup.find_all('table')
    print(f"Total tables: {len(tables)}")
    for idx, t in enumerate(tables):
        rows = t.find_all('tr')
        print(f"Table {idx}: {len(rows)} rows")
        for r_idx, row in enumerate(rows[:6]):
            cols = [c.get_text(strip=True) for c in row.find_all(['td', 'th'])]
            print(f"  Row {r_idx}: {cols[:10]}")
    with open('data/metadata/dca_dbprices_sample.html', 'w', encoding='utf-8') as f:
        f.write(r.text)
    print("Saved HTML sample to data/metadata/dca_dbprices_sample.html")

def inspect_mospi_manual():
    print("\n=== INSPECTING MoSPI CPI API MANUAL ===")
    url = "https://api.mospi.gov.in/API/CPI%20API%20User%20Manual.pdf"
    pdf_path = "data/metadata/mospi_cpi_manual.pdf"
    r = requests.get(url, headers=headers, verify=False, timeout=30)
    with open(pdf_path, "wb") as f:
        f.write(r.content)
    print(f"Downloaded PDF ({len(r.content)} bytes)")
    
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    extracted_text = []
    for i, page in enumerate(reader.pages[:8]):
        text = page.extract_text() or ""
        print(f"--- Page {i+1} ---")
        print(text[:400])
        print("...\n")
        extracted_text.append(f"--- Page {i+1} ---\n" + text)
    with open("data/metadata/mospi_cpi_manual_text.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(extracted_text))

if __name__ == "__main__":
    inspect_dca_dbprices()
    inspect_mospi_manual()
