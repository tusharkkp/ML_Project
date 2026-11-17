import pdfplumber
import json
import os
import pandas as pd

pdf_path = "data/raw/household_expenditure/Factsheet_HCES_2022-23.pdf"
print("Scanning tables in HCES Factsheet...")

with pdfplumber.open(pdf_path) as pdf:
    for page_idx, page in enumerate(pdf.pages):
        tables = page.extract_tables()
        if tables:
            print(f"\n=== Page {page_idx + 1}: Found {len(tables)} tables ===")
            for t_idx, t in enumerate(tables):
                print(f"Table {t_idx + 1} has {len(t)} rows, {len(t[0]) if t else 0} cols")
                # Print header and sample row
                for r in t[:4]:
                    clean_row = [str(c).replace('\n', ' ').strip() if c else '' for c in r]
                    print("  ", clean_row[:8])
