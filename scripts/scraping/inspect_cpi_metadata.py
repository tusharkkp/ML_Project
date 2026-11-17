import pandas as pd

excel_path = "data/raw/cpi/CPI_Metadata.xlsx"
xl = pd.ExcelFile(excel_path)
print("=== CPI METADATA SHEETS ===")
print("Sheet names:", xl.sheet_names)

for sheet in xl.sheet_names:
    df = xl.parse(sheet)
    print(f"\n--- Sheet: {sheet} (Shape: {df.shape}) ---")
    print("Columns:", list(df.columns))
    print("Sample rows:")
    print(df.head(3))
