"""
Master Pipeline Orchestration Script
AI-Based Food Inflation Prediction and Household Affordability Monitoring System

Executes modular data acquisition, web scraping, data cleaning/standardization,
and data quality validation.

Usage:
    py -3.12 scripts/main.py --all           # Run full pipeline end-to-end
    py -3.12 scripts/main.py --collect-api   # Run API clients (MoSPI, data.gov.in)
    py -3.12 scripts/main.py --scrape        # Run scrapers (DCA live bulletin, factsheet)
    py -3.12 scripts/main.py --clean         # Clean and standardize all raw datasets
    py -3.12 scripts/main.py --validate      # Validate processed datasets & generate report
    py -3.12 scripts/main.py --summary       # Print summary statistics of processed data
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

LOG_DIR = Path("data/metadata/collection_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "master_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("MasterPipeline")


def run_api_collection():
    """Runs available API collection clients."""
    logger.info("=== STEP 1: API DATA ACQUISITION ===")
    
    # 1. MoSPI Client
    try:
        from scripts.api.mospi_client import MoSPIClient
        logger.info("Running MoSPI Client for CPI and HCES data...")
        client = MoSPIClient()
        client.collect_all()
    except Exception as e:
        logger.error(f"Error in MoSPI API collection: {e}")

    # 2. data.gov.in Client
    try:
        from scripts.api.data_gov_client import DataGovClient
        logger.info("Checking data.gov.in API client...")
        api_key = os.getenv("DATA_GOV_IN_API_KEY")
        if not api_key:
            logger.warning("DATA_GOV_IN_API_KEY is not set in environment or .env. Skipping data.gov.in.")
        else:
            dg_client = DataGovClient(api_key=api_key)
            dg_client.fetch_maharashtra_daily_prices(limit=5000)
    except Exception as e:
        logger.error(f"Error in data.gov.in collection: {e}")


def run_scraping():
    """Runs web scraping and harvesters for authorized public sources."""
    logger.info("=== STEP 2: WEB HARVESTING & SCRAPING ===")
    try:
        from scripts.scraping.relevant_scrapers import (
            scrape_dca_daily_bulletin,
            download_and_extract_hces_factsheet,
            harvest_historical_agmarknet,
            harvest_historical_dca_retail
        )
        logger.info("Scraping DCA daily price monitoring bulletin...")
        scrape_dca_daily_bulletin()

        logger.info("Extracting HCES 2022-23 factsheet statements...")
        download_and_extract_hces_factsheet()

        logger.info("Harvesting Agmarknet historical mandi data...")
        harvest_historical_agmarknet()

        logger.info("Harvesting DCA retail historical data...")
        harvest_historical_dca_retail()
    except Exception as e:
        logger.error(f"Error during scraping/harvesting: {e}")


def run_cleaning():
    """Runs normalization, standardization, and cleaning pipelines."""
    logger.info("=== STEP 3: DATA CLEANING & STANDARDIZATION ===")
    try:
        from scripts.processing.clean_data import main as clean_all
        clean_all()
    except Exception as e:
        logger.error(f"Error during data cleaning: {e}")


def run_validation():
    """Runs data quality validation and generates Markdown quality report."""
    logger.info("=== STEP 4: DATA QUALITY ASSURANCE & VALIDATION ===")
    try:
        from scripts.processing.validate_data import generate_data_quality_report
        report_path = generate_data_quality_report()
        logger.info(f"Validation completed. Report generated at: {report_path}")
    except Exception as e:
        logger.error(f"Error during data validation: {e}")


def print_summary():
    """Displays summary statistics of processed datasets."""
    import pandas as pd
    logger.info("=== SUMMARY OF PROCESSED DATASETS ===")
    processed_dir = Path("data/processed")
    files = list(processed_dir.glob("*.csv"))
    if not files:
        logger.info("No processed CSV files found in data/processed/.")
        return

    print("\n" + "="*80)
    print(f"{'Dataset Name':<30} | {'Records':<10} | {'Columns':<8} | {'Date Range':<25}")
    print("="*80)
    for f in sorted(files):
        df = pd.read_csv(f)
        date_range = "N/A"
        if "observation_date" in df.columns:
            date_range = f"{df['observation_date'].min()} to {df['observation_date'].max()}"
        elif "survey_year" in df.columns:
            date_range = f"{df['survey_year'].unique().tolist()}"
        print(f"{f.name:<30} | {len(df):<10,d} | {len(df.columns):<8} | {date_range:<25}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Master Data Acquisition and Processing Pipeline for Food Inflation Project."
    )
    parser.add_argument("--all", action="store_true", help="Execute complete pipeline (API -> Scrape -> Clean -> Validate)")
    parser.add_argument("--collect-api", action="store_true", help="Execute API collectors (MoSPI, data.gov.in)")
    parser.add_argument("--scrape", action="store_true", help="Execute web scrapers & harvesters")
    parser.add_argument("--clean", action="store_true", help="Execute data cleaning and normalization")
    parser.add_argument("--validate", action="store_true", help="Execute data validation and quality report generator")
    parser.add_argument("--summary", action="store_true", help="Print summary table of processed data")

    args = parser.parse_args()

    if len(sys.argv) == 1 or args.summary:
        print_summary()
        if len(sys.argv) == 1:
            parser.print_help()
        return

    if args.all:
        logger.info("Starting Full End-to-End Pipeline...")
        run_api_collection()
        run_scraping()
        run_cleaning()
        run_validation()
        print_summary()
        logger.info("Full Pipeline Execution Complete!")
    else:
        if args.collect_api:
            run_api_collection()
        if args.scrape:
            run_scraping()
        if args.clean:
            run_cleaning()
        if args.validate:
            run_validation()


if __name__ == "__main__":
    main()
