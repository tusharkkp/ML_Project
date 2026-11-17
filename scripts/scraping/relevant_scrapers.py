"""
Relevant Scrapers & Historical Data Acquisition Script
Implements:
1. DCA PMS Daily Retail Bulletin Scraper:
   - Extracts daily prices and multi-year benchmarks (today, 1wk, 1mo, 6mo, 1yr, 2yr back)
     from fcainfoweb.nic.in/Reports/DB/DBprices.aspx
2. Historical Agmarknet Mandi Price & Arrival Harvester:
   - Retrieves multi-year daily mandi prices (Min, Max, Modal) and arrivals (Tonnes)
     for Maharashtra markets (Pune, Nashik, Ahmednagar, Mumbai, Nagpur) covering
     Tomato, Onion, Potato, Pulses, Mustard Oil, Sugar.
3. Historical DCA Retail Price Harvester:
   - Retrieves daily centre-level retail prices across essential commodities for
     Maharashtra reporting centres (Mumbai, Pune, Nagpur, Nashik).
4. MoSPI HCES Factsheet Tabular Extractor:
   - Extracts official NSSO MPCE and food basket percentage shares from the 2022-23 Factsheet PDF.
"""

import os
import sys
import time
import io
import re
import json
import logging
from datetime import datetime
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pdfplumber
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

LOG_DIR = Path("data/metadata/collection_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "scraping_collection.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("ScraperHarvester")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (Academic Research: Food Inflation Project)"
}

class DCABulletinScraper:
    """Scrapes daily average retail prices and historical benchmarks from DCA fcainfoweb."""
    URL = "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx"

    def __init__(self):
        self.raw_dir = Path("data/raw/retail")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def scrape(self):
        logger.info(f"Scraping live DCA price bulletin from {self.URL}...")
        try:
            r = requests.get(self.URL, headers=HEADERS, verify=False, timeout=25)
            if r.status_code != 200:
                logger.error(f"Failed to fetch DCA bulletin: HTTP {r.status_code}")
                return None

            # Save raw HTML
            today_str = datetime.now().strftime("%Y%m%d")
            html_path = self.raw_dir / f"dca_bulletin_raw_{today_str}.html"
            with open(html_path, "w", encoding="utf-8") as f:
                f.write(r.text)
            logger.info(f"Saved raw HTML to {html_path}")

            soup = BeautifulSoup(r.text, "html.parser")
            tables = soup.find_all("table")
            if len(tables) < 2:
                logger.error("DCA bulletin did not contain expected price table.")
                return None

            table = tables[1]
            rows = table.find_all("tr")
            if len(rows) < 4:
                logger.error("Price table has insufficient rows.")
                return None

            # Parse headers
            date_row = [c.get_text(strip=True) for c in rows[2].find_all(["th", "td"])]
            dates = {
                "current": date_row[0] if len(date_row) > 0 else today_str,
                "1_week_back": date_row[1] if len(date_row) > 1 else "",
                "1_month_back": date_row[2] if len(date_row) > 2 else "",
                "6_months_back": date_row[3] if len(date_row) > 3 else "",
                "1_year_back": date_row[4] if len(date_row) > 4 else "",
                "2_years_back": date_row[5] if len(date_row) > 5 else ""
            }

            records = []
            for row in rows[3:]:
                cols = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                if len(cols) < 3:
                    continue
                commodity = cols[0]
                unit = cols[1] if len(cols) > 1 else "1 Kg"
                curr_price = cols[2] if len(cols) > 2 else ""

                # Extract pure numeric price and variation
                def clean_price(val):
                    val = val.replace("--", "").strip()
                    m = re.match(r"^([\d\.]+)", val)
                    return float(m.group(1)) if m else None

                records.append({
                    "commodity_name": commodity,
                    "unit": unit,
                    "observation_date": dates["current"],
                    "retail_price": clean_price(curr_price),
                    "raw_current_str": curr_price,
                    "price_1_week_back": clean_price(cols[3]) if len(cols) > 3 else None,
                    "price_1_month_back": clean_price(cols[4]) if len(cols) > 4 else None,
                    "price_6_months_back": clean_price(cols[5]) if len(cols) > 5 else None,
                    "price_1_year_back": clean_price(cols[6]) if len(cols) > 6 else None,
                    "price_2_years_back": clean_price(cols[7]) if len(cols) > 7 else None,
                    "benchmark_dates": dates,
                    "geography": "All-India Average",
                    "price_type": "Retail",
                    "source": "Department of Consumer Affairs (fcainfoweb.nic.in)"
                })

            json_path = self.raw_dir / f"dca_daily_bulletin_{today_str}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2)
            logger.info(f"Successfully scraped {len(records)} commodities from DCA bulletin.")
            return records
        except Exception as e:
            logger.error(f"Error scraping DCA bulletin: {e}")
            return None


class HistoricalMandiHarvester:
    """Harvests historical Agmarknet daily mandi wholesale prices and arrivals for Maharashtra."""
    BASE_URL = "https://raw.githubusercontent.com/ICTD-IITD/Commodity_Analysis/master/Data/Original"

    def __init__(self):
        self.raw_wholesale_dir = Path("data/raw/wholesale")
        self.raw_arrivals_dir = Path("data/raw/market_arrivals")
        self.raw_wholesale_dir.mkdir(parents=True, exist_ok=True)
        self.raw_arrivals_dir.mkdir(parents=True, exist_ok=True)

    def harvest_commodity_months(self, commodity, state="Maharashtra", years=range(2016, 2021)):
        """
        Harvests monthly CSV archives for a commodity in Maharashtra.
        Parses daily arrivals, minimum, maximum, and modal wholesale prices.
        """
        months = ["January", "February", "March", "April", "May", "June", 
                  "July", "August", "September", "October", "November", "December"]
        all_rows = []
        logger.info(f"Harvesting historical mandi records for {commodity} in {state}...")

        for year in years:
            for month in months:
                file_name = f"mynewdata_{year}_{month}.csv"
                url = f"{self.BASE_URL}/{commodity}/{state}/{file_name}"
                try:
                    r = requests.get(url, headers=HEADERS, verify=False, timeout=15)
                    if r.status_code == 200:
                        lines = r.text.splitlines()
                        # Clean git conflict markers if present
                        clean_lines = []
                        header_found = False
                        for line in lines:
                            if "Market,Arrival Date" in line:
                                header_found = True
                            if header_found and not line.startswith("<<<<<<<") and not line.startswith("=======") and not line.startswith(">>>>>>>") and not line.startswith("version ") and not line.startswith("oid ") and not line.startswith("size "):
                                clean_lines.append(line)

                        if clean_lines:
                            csv_text = "\n".join(clean_lines)
                            try:
                                df = pd.read_csv(io.StringIO(csv_text), on_bad_lines="skip")
                                # Forward-fill Market column as Agmarknet only prints market on first row of date block
                                if "Market" in df.columns:
                                    df["Market"] = df["Market"].ffill()
                                df["Commodity"] = commodity
                                df["State"] = state
                                all_rows.append(df)
                                logger.info(f"  Harvested {commodity} {state} {year}-{month}: {len(df)} rows")
                            except Exception as parse_err:
                                logger.warning(f"  Parse error for {commodity} {year}-{month}: {parse_err}")
                    elif r.status_code == 404:
                        continue
                    time.sleep(0.1)
                except Exception as e:
                    logger.warning(f"Error fetching {url}: {e}")

        if all_rows:
            combined_df = pd.concat(all_rows, ignore_index=True)
            # Save raw combined CSV
            raw_out = self.raw_wholesale_dir / f"agmarknet_hist_{commodity.lower()}_{state.lower()}_2016_2020.csv"
            combined_df.to_csv(raw_out, index=False)
            logger.info(f"Saved {len(combined_df)} raw mandi records for {commodity} to {raw_out}")
            return combined_df
        return pd.DataFrame()


class HistoricalRetailHarvester:
    """Harvests historical centre-level daily retail prices from DCA records."""
    BASE_URL = "https://raw.githubusercontent.com/ICTD-IITD/Commodity_Analysis/master/Data/Original/RetailFCA"

    def __init__(self):
        self.raw_retail_dir = Path("data/raw/retail")
        self.raw_retail_dir.mkdir(parents=True, exist_ok=True)

    def harvest_sample_dates(self, date_samples):
        """
        Retrieves daily retail price matrix across Indian reporting centres for specified dates.
        """
        all_dfs = []
        logger.info(f"Harvesting historical retail price snapshots for {len(date_samples)} dates...")
        for date_str in date_samples:  # Format: YYYY_MM_DD
            url = f"{self.BASE_URL}/{date_str}.csv"
            try:
                r = requests.get(url, headers=HEADERS, verify=False, timeout=10)
                if r.status_code == 200:
                    lines = r.text.splitlines()
                    clean_lines = []
                    header_found = False
                    for line in lines:
                        if "Centre," in line:
                            header_found = True
                        if header_found and not line.startswith("<<<") and not line.startswith("===") and not line.startswith(">>>") and not line.startswith("version ") and not line.startswith("oid ") and not line.startswith("size "):
                            clean_lines.append(line)
                    if clean_lines:
                        df = pd.read_csv(io.StringIO("\n".join(clean_lines)))
                        df["Date"] = date_str.replace("_", "-")
                        all_dfs.append(df)
                time.sleep(0.05)
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")

        if all_dfs:
            combined_df = pd.concat(all_dfs, ignore_index=True)
            out_path = self.raw_retail_dir / f"dca_hist_retail_multi_centre_sample.csv"
            combined_df.to_csv(out_path, index=False)
            logger.info(f"Saved {len(combined_df)} historical retail centre records to {out_path}")
            return combined_df
        return pd.DataFrame()


class HCESFactsheetExtractor:
    """Extracts official tables from the MoSPI HCES 2022-23 Factsheet PDF."""
    def __init__(self, pdf_path="data/raw/household_expenditure/Factsheet_HCES_2022-23.pdf"):
        self.pdf_path = Path(pdf_path)
        self.out_dir = Path("data/raw/household_expenditure")
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def extract_key_statements(self):
        if not self.pdf_path.exists():
            logger.error(f"HCES Factsheet PDF not found at {self.pdf_path}")
            return None

        logger.info(f"Extracting structured tables from {self.pdf_path}...")
        extracted_data = {}

        with pdfplumber.open(self.pdf_path) as pdf:
            # Statement 1: Page 8 (Average MPCE & share of food/non-food)
            p8_tables = pdf.pages[7].extract_tables()
            if p8_tables:
                extracted_data["statement_1_food_share"] = p8_tables[0]

            # Statement 4: Page 11 (Average MPCE across 12 fractile classes)
            p11_tables = pdf.pages[10].extract_tables()
            if p11_tables:
                extracted_data["statement_4_fractile_classes"] = p11_tables[0]

            # Statement 5: Page 12 (Absolute and % break-up of MPCE by item group)
            p12_tables = pdf.pages[11].extract_tables()
            if p12_tables:
                extracted_data["statement_5_item_groups"] = p12_tables[0]

            # Statement 8: Page 15 (Average MPCE for each State/UT, including Maharashtra)
            p15_tables = pdf.pages[14].extract_tables()
            if p15_tables:
                extracted_data["statement_8_state_mpce"] = p15_tables[0]

        # Save extracted raw tables as JSON
        out_json = self.out_dir / "hces_factsheet_extracted_statements.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(extracted_data, f, indent=2)
        logger.info(f"Successfully extracted HCES statements to {out_json}")
        return extracted_data


def main():
    logger.info("=== STARTING SCRAPING AND HARVESTING PIPELINE ===")

    # 1. Scrape live DCA Bulletin
    dca = DCABulletinScraper()
    dca_records = dca.scrape()

    # 2. Extract HCES Factsheet statements
    hces = HCESFactsheetExtractor()
    hces_statements = hces.extract_key_statements()

    # 3. Harvest historical Maharashtra Mandi records
    mandi = HistoricalMandiHarvester()
    tomato_df = mandi.harvest_commodity_months("Tomato", state="Maharashtra", years=[2016, 2017, 2018])
    onion_df = mandi.harvest_commodity_months("Onion", state="Maharashtra", years=[2016, 2017, 2018])
    sugar_df = mandi.harvest_commodity_months("Sugar", state="Maharashtra", years=[2016, 2017, 2018])
    moong_df = mandi.harvest_commodity_months("Green Gram Dal (Moong Dal)", state="Maharashtra", years=[2016, 2017, 2018])

    # 4. Harvest sample historical DCA Retail records (e.g. monthly snapshots across 2010-2012)
    sample_dates = []
    for year in [2010, 2011, 2012]:
        for month in range(1, 13):
            sample_dates.append(f"{year}_{month:02d}_01")
            sample_dates.append(f"{year}_{month:02d}_15")
    retail = HistoricalRetailHarvester()
    retail_df = retail.harvest_sample_dates(sample_dates)

    logger.info("=== SCRAPING AND HARVESTING PIPELINE FINISHED ===")

if __name__ == "__main__":
    main()
