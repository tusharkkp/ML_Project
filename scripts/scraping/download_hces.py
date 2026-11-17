import requests
import os
import pypdf
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://www.mospi.gov.in/sites/default/files/publication_reports/Factsheet_HCES_2022-23.pdf"
out_dir = os.path.join("data", "raw", "household_expenditure")
pdf_path = os.path.join(out_dir, "Factsheet_HCES_2022-23.pdf")

print("Downloading HCES Factsheet from MoSPI...")
r = requests.get(url, headers=headers, verify=False, timeout=30)
print("Status:", r.status_code, "Length:", len(r.content))
if r.status_code == 200:
    with open(pdf_path, "wb") as f:
        f.write(r.content)
    print(f"Saved to {pdf_path}")
    reader = pypdf.PdfReader(pdf_path)
    print(f"Total pages: {len(reader.pages)}")
    for i in range(min(5, len(reader.pages))):
        print(f"--- Page {i+1} ---")
        print((reader.pages[i].extract_text() or "")[:300])
