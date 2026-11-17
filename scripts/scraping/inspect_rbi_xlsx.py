import requests
import pandas as pd
import os
import urllib3
urllib3.disable_warnings()

headers = {"User-Agent": "Mozilla/5.0"}
url = "https://rbidocs.rbi.org.in/rdocs/Publications/DOCs/174T_29082025B9033D235E734A2CBE6B9FDE3E429A20.XLSX"
out_path = os.path.join("data", "raw", "cpi", "rbi_cpi_monthly.xlsx")

print("Downloading RBI CPI monthly XLSX...")
r = requests.get(url, headers=headers, verify=False, timeout=20)
print("Status:", r.status_code, "Length:", len(r.content))
with open(out_path, "wb") as f:
    f.write(r.content)
print("Saved to", out_path)

xl = pd.ExcelFile(out_path)
print("Sheet names:", xl.sheet_names)
for s in xl.sheet_names:
    df = xl.parse(s)
    print(f"\nSheet {s} shape: {df.shape}")
    print("Head:")
    print(df.iloc[:8, :8])
