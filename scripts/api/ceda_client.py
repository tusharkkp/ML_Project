"""
CEDA (Centre for Economic Data and Analysis - Ashoka University) API Client
Implements the OpenAPI 3.0 specification for CEDA Agmarknet endpoints:
- GET /v1/agmarknet/commodities
- GET /v1/agmarknet/geographies
- POST /v1/agmarknet/markets
- POST /v1/agmarknet/prices
- POST /v1/agmarknet/quantities

Uses CEDA_API_TOKEN from environment or .env.
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

load_dotenv()

LOG_DIR = Path("data/metadata/collection_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "ceda_collection.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("CEDAClient")

BASE_URL = "https://api.ceda.ashoka.edu.in/v1"
DEFAULT_TIMEOUT = 30
MAX_RETRIES = 3
BACKOFF_FACTOR = 2.0

class CEDAClient:
    def __init__(self, api_token=None):
        self.api_token = api_token or os.getenv("CEDA_API_TOKEN")
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Academic-Research-Food-Inflation",
            "Content-Type": "application/json"
        })
        if self.api_token:
            self.session.headers.update({"Authorization": f"Bearer {self.api_token}"})
        self.raw_dir = Path("data/raw/wholesale")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def is_configured(self):
        return bool(self.api_token and self.api_token.strip() and self.api_token != "optional_if_registered")

    def _request(self, method, endpoint, json_body=None):
        if not self.is_configured():
            logger.warning(
                "CEDA_API_TOKEN is not set. CEDA API requires authentication token.\n"
                "Register at https://ceda.ashoka.edu.in to obtain a token if desired."
            )
            return None

        url = f"{BASE_URL.rstrip('/')}/{endpoint.lstrip('/')}"
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.session.request(method, url, json=json_body, timeout=DEFAULT_TIMEOUT, verify=False)
                if resp.status_code == 200:
                    return resp.json()
                elif resp.status_code == 401:
                    logger.error("CEDA Authentication failed: invalid or expired token.")
                    return None
                else:
                    logger.error(f"HTTP {resp.status_code} from CEDA {url}: {resp.text[:200]}")
                    time.sleep(BACKOFF_FACTOR ** attempt)
            except Exception as e:
                logger.error(f"Error requesting {url}: {e}")
                time.sleep(BACKOFF_FACTOR ** attempt)
        return None

    def get_commodities(self):
        return self._request("GET", "agmarknet/commodities")

    def get_geographies(self):
        return self._request("GET", "agmarknet/geographies")

    def get_markets(self, commodity_id, state_id, district_id, indicator="price"):
        body = {
            "commodity_id": commodity_id,
            "state_id": state_id,
            "district_id": district_id,
            "indicator": indicator
        }
        return self._request("POST", "agmarknet/markets", json_body=body)

    def get_prices(self, commodity_id, state_id, from_date, to_date, district_ids=None, market_ids=None):
        body = {
            "commodity_id": commodity_id,
            "state_id": state_id,
            "from_date": from_date,
            "to_date": to_date
        }
        if district_ids:
            body["district_id"] = district_ids
        if market_ids:
            body["market_id"] = market_ids
        return self._request("POST", "agmarknet/prices", json_body=body)

    def get_quantities(self, commodity_id, state_id, from_date, to_date, district_ids=None, market_ids=None):
        body = {
            "commodity_id": commodity_id,
            "state_id": state_id,
            "from_date": from_date,
            "to_date": to_date
        }
        if district_ids:
            body["district_id"] = district_ids
        if market_ids:
            body["market_id"] = market_ids
        return self._request("POST", "agmarknet/quantities", json_body=body)

def main():
    client = CEDAClient()
    if not client.is_configured():
        print("CEDAClient: CEDA_API_TOKEN not configured. Integration paused until configured.")
        return
    res = client.get_commodities()
    print("CEDA Commodities sample:", res[:5] if res else "No response")

if __name__ == "__main__":
    main()
