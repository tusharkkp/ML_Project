"""
data.gov.in (Open Government Data Platform India) API Client
Acquires daily wholesale commodity prices and market arrivals from AGMARKNET resource:
Resource ID: 9ef84268-d588-465a-a308-a864a43d0070
("Current Daily Price of Various Commodities from Various Markets (Mandi)")

Features:
- Reads DATA_GOV_IN_API_KEY from environment or .env file (never hardcoded)
- Filters by State (Maharashtra), District (Pune, Mumbai, Nashik, Nagpur), Commodity
- Automated pagination via offset & limit
- Exponential backoff retry logic and connection resilience
- Rate limiting and incremental checkpointing in data/raw/wholesale/
- Logs activity in data/metadata/collection_logs/data_gov_collection.log
"""

import os
import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path
import requests
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

LOG_DIR = Path("data/metadata/collection_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "data_gov_collection.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DataGovClient")

BASE_API_URL = "https://api.data.gov.in/resource"
RESOURCE_ID_CURRENT_DAILY = "9ef84268-d588-465a-a308-a864a43d0070"

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 5
BACKOFF_FACTOR = 2.0
RATE_LIMIT_DELAY = 1.0  # Respect Open Government Data rate limits
MAX_PAGE_SIZE = 1000

class DataGovClient:
    def __init__(self, api_key=None, resource_id=RESOURCE_ID_CURRENT_DAILY):
        self.api_key = api_key or os.getenv("DATA_GOV_IN_API_KEY")
        self.resource_id = resource_id
        self.raw_dir = Path("data/raw/wholesale")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Academic-Research-Food-Inflation",
            "Accept": "application/json"
        })

    def is_configured(self):
        """Checks whether the API key is present."""
        return bool(self.api_key and self.api_key.strip() and self.api_key != "your_api_key_here")

    def _request_page(self, offset=0, limit=MAX_PAGE_SIZE, filters=None):
        """Fetches a single page with retries and exponential backoff."""
        if not self.is_configured():
            raise ValueError(
                "DATA_GOV_IN_API_KEY environment variable is missing or empty.\n"
                "Please configure DATA_GOV_IN_API_KEY in your .env file or environment."
            )

        url = f"{BASE_API_URL}/{self.resource_id}"
        params = {
            "api-key": self.api_key,
            "format": "json",
            "offset": str(offset),
            "limit": str(limit)
        }
        if filters:
            for k, v in filters.items():
                params[f"filters[{k}]"] = str(v)

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                time.sleep(RATE_LIMIT_DELAY)
                resp = self.session.get(url, params=params, timeout=DEFAULT_TIMEOUT)
                if resp.status_code == 200:
                    try:
                        return resp.json()
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON at offset {offset}")
                        return None
                elif resp.status_code in [429, 500, 502, 503, 504]:
                    sleep_time = BACKOFF_FACTOR ** attempt
                    logger.warning(f"HTTP {resp.status_code}. Backing off for {sleep_time:.1f}s (Attempt {attempt}/{MAX_RETRIES})")
                    time.sleep(sleep_time)
                elif resp.status_code in [400, 401, 403]:
                    logger.error(f"Authentication/Authorization error {resp.status_code}: {resp.text[:200]}")
                    return {"status": "auth_error", "message": resp.text}
                else:
                    logger.error(f"HTTP {resp.status_code} at offset {offset}: {resp.text[:200]}")
                    return None
            except (requests.ConnectionError, requests.Timeout) as e:
                sleep_time = BACKOFF_FACTOR ** attempt
                logger.warning(f"Network error ({e}). Retrying in {sleep_time:.1f}s")
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                return None
        return None

    def fetch_all(self, state="Maharashtra", districts=None, commodities=None, max_records=None):
        """
        Retrieves all records matching state and optional district/commodity filters.
        Paginates until all records are collected or max_records is reached.
        """
        if not self.is_configured():
            logger.warning("data.gov.in API key not configured. Skipping collection until configured.")
            return []

        all_records = []
        offset = 0
        total_records = None
        filters = {}
        if state:
            filters["state"] = state

        logger.info(f"Starting data.gov.in harvest for resource={self.resource_id}, filters={filters}")

        while True:
            res = self._request_page(offset=offset, limit=MAX_PAGE_SIZE, filters=filters)
            if not res or res.get("status") == "auth_error":
                logger.error("Failed to retrieve page or authorization failed.")
                break

            records = res.get("records", [])
            if not records:
                logger.info(f"No records returned at offset {offset}. Done.")
                break

            if total_records is None:
                total_records = int(res.get("total", len(records)))
                logger.info(f"Total matching records reported by data.gov.in: {total_records}")

            # Client-side filtering if specific districts or commodities are requested
            for r in records:
                if districts:
                    if r.get("district", "").strip().lower() not in [d.lower() for d in districts]:
                        continue
                if commodities:
                    if not any(c.lower() in r.get("commodity", "").lower() for c in commodities):
                        continue
                all_records.append(r)

            logger.info(f"Fetched offset {offset} (+{len(records)} raw, {len(all_records)} filtered).")
            offset += len(records)

            if max_records and len(all_records) >= max_records:
                logger.info(f"Reached max_records target ({max_records}). Stopping.")
                break

            if offset >= total_records:
                logger.info("Reached end of total records.")
                break

        # Save raw JSON checkpoint
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = self.raw_dir / f"data_gov_mandi_{state.lower()}_{timestamp}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump({
                "collection_timestamp": datetime.now().isoformat(),
                "resource_id": self.resource_id,
                "state": state,
                "total_collected": len(all_records),
                "records": all_records
            }, f, indent=2)
        logger.info(f"Successfully saved {len(all_records)} records to {out_file}")
        return all_records

    def fetch_maharashtra_daily_prices(self, limit=None):
        """Convenience method to retrieve Maharashtra mandi records."""
        return self.fetch_all(state="Maharashtra", max_records=limit)

def main():
    client = DataGovClient()
    if not client.is_configured():
        print("\n=======================================================")
        print("NOTICE: DATA_GOV_IN_API_KEY is not set.")
        print("Please copy .env.example to .env and configure your key:")
        print("  DATA_GOV_IN_API_KEY=your_actual_key_here")
        print("=======================================================\n")
        return

    print("Fetching complete daily mandi price records for Maharashtra...")
    records = client.fetch_maharashtra_daily_prices()
    print(f"Collection complete: {len(records)} Maharashtra mandi records retrieved.")

if __name__ == "__main__":
    main()
