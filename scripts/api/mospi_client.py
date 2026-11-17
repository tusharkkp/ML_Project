"""
MoSPI (Ministry of Statistics and Programme Implementation) API Client
Fetches:
1. Consumer Price Index (CPI) and Inflation series:
   - All-India (99) and Maharashtra (27)
   - Group 1: Food and Beverages
   - Subgroups: Cereals, Pulses, Vegetables, Oils and Fats, Milk, Sugar
   - Historical range: 2014 to 2026 (Monthly)
2. Item-level CPI Index and Inflation for essential food items:
   - Rice, Wheat, Potato, Onion, Tomato, Arhar Dal, Gram Dal, Moong Dal, etc.
3. Household Consumption Expenditure Survey (HCES) indicators:
   - Percentage share of MPCE by broad categories
   - Monthly Per Capita Consumption Expenditure by fractile classes

Persists raw responses in data/raw/cpi/ and data/raw/household_expenditure/.
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Ensure log directory exists
LOG_DIR = Path("data/metadata/collection_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mospi_collection.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MoSPIClient")

BASE_URL = "https://api.mospi.gov.in/api"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 1.5
RATE_LIMIT_DELAY = 0.5  # Seconds between requests to respect government servers

class MoSPIClient:
    def __init__(self, api_token=None):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Academic-Research-Food-Inflation",
            "Accept": "application/json"
        })
        if api_token:
            self.session.headers.update({"Authorization": f"Bearer {api_token}"})
        self.raw_cpi_dir = Path("data/raw/cpi")
        self.raw_cpi_dir.mkdir(parents=True, exist_ok=True)
        self.raw_hces_dir = Path("data/raw/household_expenditure")
        self.raw_hces_dir.mkdir(parents=True, exist_ok=True)

    def _request_with_retry(self, method, endpoint, params=None, json_body=None):
        url = f"{BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                time.sleep(RATE_LIMIT_DELAY)
                resp = self.session.request(
                    method,
                    url,
                    params=params,
                    json=json_body,
                    timeout=DEFAULT_TIMEOUT,
                    verify=False
                )
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode JSON from {url}: {resp.text[:200]}")
                        return None
                elif resp.status_code in [429, 500, 502, 503, 504]:
                    sleep_time = BACKOFF_FACTOR ** attempt
                    logger.warning(f"HTTP {resp.status_code} for {url}. Retrying in {sleep_time:.1f}s (Attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(sleep_time)
                else:
                    logger.error(f"HTTP {resp.status_code} from {url}: {resp.text[:200]}")
                    return None
            except (requests.ConnectionError, requests.Timeout) as e:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.warning(f"Network error ({e}) for {url}. Retrying in {sleep_time:.1f}s (Attempt {attempt}/{MAX_RETRIES})")
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Unexpected error querying {url}: {e}")
                return None
        logger.error(f"Max retries exceeded for {url}")
        return None

    def get_cpi_indices(self, years, state_codes=("27", "99"), group_codes=("1",), sector_codes=("1", "2", "3")):
        """
        Fetches CPI indices & inflation across specified years, states, groups, and sectors.
        Paginates automatically through all available records.
        """
        all_records = []
        year_str = ",".join(str(y) for y in years)
        state_str = ",".join(str(s) for s in state_codes)
        group_str = ",".join(str(g) for g in group_codes)
        sector_str = ",".join(str(sec) for sec in sector_codes)
        months_str = ",".join(str(m) for m in range(1, 13))

        page = 1
        total_pages = 1
        logger.info(f"Querying MoSPI CPI Index for years={year_str}, states={state_str}, sectors={sector_str}")

        while page <= total_pages:
            params = {
                "base_year": "2012",
                "series": "Current",
                "year": year_str,
                "month_code": months_str,
                "state_code": state_str,
                "group_code": group_str,
                "sector_code": sector_str,
                "page": str(page),
                "Format": "JSON"
            }
            res = self._request_with_retry("GET", "cpi/getCPIIndex", params=params)
            if not res or not res.get("statusCode"):
                logger.warning(f"No data or unsuccessful status on page {page}")
                break

            records = res.get("data", [])
            all_records.extend(records)
            meta = res.get("meta_data", {})
            total_pages = meta.get("totalPages", 1)
            total_records = meta.get("totalRecords", len(all_records))
            logger.info(f"Page {page}/{total_pages} retrieved ({len(records)} records). Progress: {len(all_records)}/{total_records}")
            page += 1

        # Save raw JSON checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = self.raw_cpi_dir / f"mospi_cpi_groups_{timestamp}.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump({
                "collection_timestamp": datetime.now().isoformat(),
                "source": "MoSPI CPI API (api.mospi.gov.in)",
                "total_records": len(all_records),
                "records": all_records
            }, f, indent=2)
        logger.info(f"Saved {len(all_records)} raw CPI group records to {raw_file}")
        return all_records

    def get_cpi_item_indices(self, item_codes, years):
        """
        Fetches item-level CPI index and inflation for specific items.
        """
        all_records = []
        year_str = ",".join(str(y) for y in years)
        months_str = ",".join(str(m) for m in range(1, 13))

        for item_code, item_name in item_codes.items():
            logger.info(f"Querying MoSPI Item CPI for {item_name} ({item_code})...")
            page = 1
            total_pages = 1
            while page <= total_pages:
                params = {
                    "base_year": "2012",
                    "year": year_str,
                    "month_code": months_str,
                    "item_code": item_code,
                    "page": str(page),
                    "Format": "JSON"
                }
                res = self._request_with_retry("GET", "cpi/getItemIndex", params=params)
                if not res or not res.get("statusCode"):
                    break
                records = res.get("data", [])
                for r in records:
                    r["item_code"] = item_code
                    r["item_query_name"] = item_name
                all_records.extend(records)
                meta = res.get("meta_data", {})
                total_pages = meta.get("totalPages", 1)
                page += 1

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = self.raw_cpi_dir / f"mospi_cpi_items_{timestamp}.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump({
                "collection_timestamp": datetime.now().isoformat(),
                "source": "MoSPI Item CPI API",
                "total_records": len(all_records),
                "records": all_records
            }, f, indent=2)
        logger.info(f"Saved {len(all_records)} raw CPI item records to {raw_file}")
        return all_records

    def get_hces_records(self, indicator_codes=None):
        """
        Fetches HCES survey consumption and expenditure records.
        """
        if indicator_codes is None:
            indicator_codes = [
                "Percentage Share of Monthly Per Capita Consumption Expenditure (MPCE) over Broad Categories in 30 Days",
                "Monthly Per Capita Consumption Expenditure (MPCE) over Broad Categories in 30 Days",
                "Average monthly per capita consumption expenditure (MPCE)",
                "Average monthly per capita consumption expenditure (MPCE) Over 12 Fractile Classes"
            ]
        all_records = []
        for ind in indicator_codes:
            logger.info(f"Querying MoSPI HCES for indicator: '{ind}'...")
            params = {
                "indicator_code": ind,
                "year": "2022-23",
                "Format": "JSON"
            }
            res = self._request_with_retry("GET", "hces/getHcesRecords", params=params)
            if res and res.get("data"):
                records = res.get("data", [])
                all_records.extend(records)
                logger.info(f"  Retrieved {len(records)} records for indicator: {ind}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = self.raw_hces_dir / f"mospi_hces_{timestamp}.json"
        with open(raw_file, "w", encoding="utf-8") as f:
            json.dump({
                "collection_timestamp": datetime.now().isoformat(),
                "source": "MoSPI HCES API",
                "total_records": len(all_records),
                "records": all_records
            }, f, indent=2)
        logger.info(f"Saved {len(all_records)} raw HCES records to {raw_file}")
        return all_records

def main():
    client = MoSPIClient()
    years = list(range(2014, 2027))
    print(f"Starting MoSPI collection for years: {years[0]} - {years[-1]}")
    
    # 1. Collect Group Level CPI (Food & Beverages and all subgroups)
    # State 27 = Maharashtra, 99 = All India
    group_records = client.get_cpi_indices(years=years, state_codes=("27", "99"), group_codes=("1",))
    print(f"Collected {len(group_records)} CPI Group observations.")

    # 2. Collect Item Level CPI for key target commodities
    # Item codes from CPI_Metadata.xlsx
    target_items = {
        "1.1.01.1.1.01.P": "Rice - PDS",
        "1.1.01.1.1.02.X": "Rice - Other Sources",
        "1.1.01.2.1.01.P": "Wheat/Atta - PDS",
        "1.1.01.2.1.02.X": "Wheat/Atta - Other Sources",
        "1.1.08.1.1.01.0": "Arhar/Tur Dal",
        "1.1.08.1.1.02.0": "Gram Dal",
        "1.1.08.1.1.03.0": "Moong Dal",
        "1.1.08.1.1.04.0": "Urad Dal",
        "1.1.07.1.1.01.0": "Potato",
        "1.1.07.1.1.02.0": "Onion",
        "1.1.07.1.1.03.0": "Tomato",
        "1.1.05.1.1.01.0": "Mustard Oil",
        "1.1.05.1.1.04.0": "Sunflower Oil",
        "1.1.04.1.1.01.0": "Milk (Liquid)",
        "1.1.09.1.1.01.0": "Sugar"
    }
    item_records = client.get_cpi_item_indices(target_items, years=years)
    print(f"Collected {len(item_records)} CPI Item observations.")

    # 3. Collect HCES expenditure records
    hces_records = client.get_hces_records()
    print(f"Collected {len(hces_records)} HCES observations.")

if __name__ == "__main__":
    main()
