"""
Data Cleaning and Standardization Pipeline
Transforms raw datasets from data/raw/ into standardized, validated processed files:
- data/processed/retail_prices.csv
- data/processed/wholesale_prices.csv
- data/processed/market_arrivals.csv
- data/processed/cpi_data.csv
- data/processed/household_expenditure.csv

Enforces:
- Date ISO 8601 YYYY-MM-DD standardization
- Canonical commodity naming with raw preservation
- Strict Retail vs Wholesale price separation
- Geographic standardization (State, District, Market / Reporting Centre)
- Numeric conversion and missing value / anomaly flagging
"""

import os
import sys
import glob
import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.processing.normalize_data import (
    standardize_commodity_name,
    standardize_state_name,
    standardize_district_name,
    parse_standard_date,
    clean_numeric_price
)

LOG_DIR = Path("data/metadata/collection_logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "data_cleaning.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DataCleaner")

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Known Maharashtra market-to-district mappings
MH_MARKET_DISTRICTS = {
    "pune": "Pune",
    "pune(pimpri)": "Pune",
    "pune(manjri)": "Pune",
    "pune(khadki)": "Pune",
    "mumbai": "Mumbai",
    "vashi": "Mumbai",
    "kalyan": "Mumbai",
    "nashik": "Nashik",
    "lasalgaon": "Nashik",
    "pimpalgaon": "Nashik",
    "yeola": "Nashik",
    "malegaon": "Nashik",
    "nagpur": "Nagpur",
    "nagpur(fruit & veg.)": "Nagpur",
    "kalmeshwar": "Nagpur",
    "ahmednagar": "Ahmednagar",
    "rahata": "Ahmednagar",
    "sangamner": "Ahmednagar",
    "kopargaon": "Ahmednagar",
    "shrirampur": "Ahmednagar",
    "solapur": "Solapur",
    "kolhapur": "Kolhapur",
    "jalgaon": "Jalgaon",
    "aurangabad": "Chhatrapati Sambhajinagar",
    "amravati": "Amravati"
}

def clean_retail_data():
    """
    Cleans raw retail datasets:
    1. DCA live bulletin (All-India averages + 2-year benchmarks)
    2. Historical DCA multi-centre daily matrices
    Outputs data/processed/retail_prices.csv
    """
    logger.info("--- Cleaning Retail Price Data ---")
    retail_rows = []

    # 1. Process DCA bulletin JSON
    bulletin_files = glob.glob("data/raw/retail/dca_daily_bulletin_*.json")
    for b_file in bulletin_files:
        try:
            with open(b_file, "r", encoding="utf-8") as f:
                records = json.load(f)
            for r in records:
                canonical_comm, raw_comm = standardize_commodity_name(r.get("commodity_name"))
                obs_date = parse_standard_date(r.get("observation_date"))
                price = clean_numeric_price(r.get("retail_price"))
                unit = r.get("unit", "1 Kg")
                if price is not None and obs_date:
                    retail_rows.append({
                        "observation_date": obs_date,
                        "commodity_name": canonical_comm,
                        "raw_commodity_name": raw_comm,
                        "state": "All India",
                        "district": "All India",
                        "reporting_centre": "All-India Average",
                        "market_type": "Retail Average",
                        "retail_price": price,
                        "price_unit": unit,
                        "price_type": "Retail",
                        "benchmark_tag": "Current Daily Average",
                        "data_source": "DCA Price Monitoring Bulletin (fcainfoweb.nic.in)"
                    })
                # Also include historical benchmark comparisons as explicit observations
                benchmarks = [
                    ("price_1_week_back", "1_week_back", "1 Week Back"),
                    ("price_1_month_back", "1_month_back", "1 Month Back"),
                    ("price_6_months_back", "6_months_back", "6 Months Back"),
                    ("price_1_year_back", "1_year_back", "1 Year Back"),
                    ("price_2_years_back", "2_years_back", "2 Years Back")
                ]
                b_dates = r.get("benchmark_dates", {})
                for p_col, d_key, tag in benchmarks:
                    b_price = clean_numeric_price(r.get(p_col))
                    b_date = parse_standard_date(b_dates.get(d_key))
                    if b_price is not None and b_date:
                        retail_rows.append({
                            "observation_date": b_date,
                            "commodity_name": canonical_comm,
                            "raw_commodity_name": raw_comm,
                            "state": "All India",
                            "district": "All India",
                            "reporting_centre": "All-India Average",
                            "market_type": "Retail Average",
                            "retail_price": b_price,
                            "price_unit": unit,
                            "price_type": "Retail",
                            "benchmark_tag": tag,
                            "data_source": "DCA Price Monitoring Bulletin (fcainfoweb.nic.in)"
                        })
        except Exception as e:
            logger.error(f"Error parsing {b_file}: {e}")

    # 2. Process Historical DCA multi-centre CSV
    hist_retail_files = glob.glob("data/raw/retail/*retail*.csv")
    for hr_file in hist_retail_files:
        try:
            df = pd.read_csv(hr_file)
            if "Centre" not in df.columns or "Date" not in df.columns:
                continue
            commodity_cols = [c for c in df.columns if c not in ["Centre", "Date"]]
            for _, row in df.iterrows():
                raw_centre = str(row["Centre"]).strip()
                raw_date = row["Date"]
                obs_date = parse_standard_date(raw_date)
                centre_lower = raw_centre.lower()
                
                # Determine state
                state = "Unknown"
                district = raw_centre.title()
                if any(c in centre_lower for c in ["mumbai", "pune", "nagpur", "nashik"]):
                    state = "Maharashtra"
                elif any(c in centre_lower for c in ["delhi"]):
                    state = "Delhi"
                elif any(c in centre_lower for c in ["lucknow", "kanpur", "varanasi"]):
                    state = "Uttar Pradesh"
                elif any(c in centre_lower for c in ["bengaluru", "bangalore"]):
                    state = "Karnataka"
                elif any(c in centre_lower for c in ["kolkata"]):
                    state = "West Bengal"
                elif any(c in centre_lower for c in ["chennai"]):
                    state = "Tamil Nadu"

                for c_col in commodity_cols:
                    raw_p = row[c_col]
                    price = clean_numeric_price(raw_p)
                    if price is not None and obs_date:
                        canonical_comm, raw_comm = standardize_commodity_name(c_col)
                        # Determine unit
                        unit = "1 Litre" if "oil" in canonical_comm.lower() or "milk" in canonical_comm.lower() else "1 Kg"
                        retail_rows.append({
                            "observation_date": obs_date,
                            "commodity_name": canonical_comm,
                            "raw_commodity_name": raw_comm,
                            "state": state,
                            "district": district,
                            "reporting_centre": raw_centre.title(),
                            "market_type": "Reporting Centre Retail",
                            "retail_price": price,
                            "price_unit": unit,
                            "price_type": "Retail",
                            "benchmark_tag": "Historical Daily",
                            "data_source": "DCA Retail Monitoring Historical Archive"
                        })
        except Exception as e:
            logger.error(f"Error parsing historical retail {hr_file}: {e}")

    retail_df = pd.DataFrame(retail_rows)
    if not retail_df.empty:
        # Deduplicate
        before_cnt = len(retail_df)
        retail_df = retail_df.drop_duplicates(
            subset=["observation_date", "commodity_name", "reporting_centre", "price_type", "retail_price"]
        )
        retail_df = retail_df.sort_values(by=["commodity_name", "observation_date", "reporting_centre"])
        out_csv = PROCESSED_DIR / "retail_prices.csv"
        retail_df.to_csv(out_csv, index=False)
        logger.info(f"Cleaned {len(retail_df)} retail price records (deduplicated from {before_cnt}). Saved to {out_csv}")
    else:
        logger.warning("No retail records found to process.")
    return retail_df


def clean_wholesale_and_arrivals_data():
    """
    Cleans raw wholesale datasets:
    1. Agmarknet historical monthly mandi records (Tomato, Onion, Sugar, Moong Dal)
    2. data.gov.in mandi collections
    Outputs:
    - data/processed/wholesale_prices.csv
    - data/processed/market_arrivals.csv
    """
    logger.info("--- Cleaning Wholesale Mandi & Arrival Data ---")
    wholesale_rows = []
    arrival_rows = []

    mandi_files = glob.glob("data/raw/wholesale/agmarknet_hist_*.csv")
    for mf in mandi_files:
        try:
            df = pd.read_csv(mf, on_bad_lines="skip")
            # Required columns: Market, Arrival Date, Arrivals (Tonnes), Variety, Minimum Price, Maximum Price, Modal Price
            for _, row in df.iterrows():
                raw_market = str(row.get("Market", "")).strip()
                if not raw_market or raw_market.lower() == "nan":
                    continue
                obs_date = parse_standard_date(row.get("Arrival Date"))
                if not obs_date:
                    continue

                raw_comm = row.get("Commodity", "")
                canonical_comm, _ = standardize_commodity_name(raw_comm)
                variety = str(row.get("Variety", "Other")).strip()
                state = standardize_state_name(row.get("State", "Maharashtra"))

                # Infer district from market name
                market_lower = raw_market.lower()
                district = MH_MARKET_DISTRICTS.get(market_lower, "")
                if not district:
                    # Partial match
                    for k, dist in MH_MARKET_DISTRICTS.items():
                        if k in market_lower:
                            district = dist
                            break
                if not district:
                    district = raw_market.title()

                # Wholesale prices in Rs/Quintal
                min_p = clean_numeric_price(row.get("Minimum Price(Rs./Quintal)"))
                max_p = clean_numeric_price(row.get("Maximum Price(Rs./Quintal)"))
                modal_p = clean_numeric_price(row.get("Modal Price(Rs./Quintal)"))

                # Sanity validation: check relationships
                is_anomalous = False
                if min_p and max_p and min_p > max_p:
                    is_anomalous = True
                if modal_p and min_p and modal_p < min_p:
                    is_anomalous = True
                if modal_p and max_p and modal_p > max_p:
                    is_anomalous = True

                if modal_p is not None or min_p is not None or max_p is not None:
                    wholesale_rows.append({
                        "observation_date": obs_date,
                        "commodity_name": canonical_comm,
                        "raw_commodity_name": raw_comm,
                        "variety": variety,
                        "state": state,
                        "district": district,
                        "market_name": raw_market.title(),
                        "min_price_rs_per_qtl": min_p,
                        "max_price_rs_per_qtl": max_p,
                        "modal_price_rs_per_qtl": modal_p,
                        "price_unit": "Rs/Quintal",
                        "price_type": "Wholesale",
                        "is_anomalous_flag": is_anomalous,
                        "data_source": "AGMARKNET Historical Mandi Archive"
                    })

                # Arrival quantity in Tonnes
                raw_arrival = row.get("Arrivals (Tonnes)")
                arrival_qty = clean_numeric_price(raw_arrival)
                if arrival_qty is not None:
                    arrival_rows.append({
                        "observation_date": obs_date,
                        "commodity_name": canonical_comm,
                        "raw_commodity_name": raw_comm,
                        "variety": variety,
                        "state": state,
                        "district": district,
                        "market_name": raw_market.title(),
                        "arrival_quantity_tonnes": arrival_qty,
                        "unit": "Tonnes",
                        "data_source": "AGMARKNET Mandi Arrivals"
                    })
        except Exception as e:
            logger.error(f"Error parsing wholesale file {mf}: {e}")

    # 2. Process data.gov.in AGMARKNET API JSON files
    data_gov_files = glob.glob("data/raw/wholesale/data_gov_mandi_*.json")
    for dgf in data_gov_files:
        try:
            with open(dgf, "r", encoding="utf-8") as f:
                content = json.load(f)
            records = content.get("records", [])
            for r in records:
                raw_market = str(r.get("market", "")).strip()
                if not raw_market or raw_market.lower() == "nan":
                    continue
                obs_date = parse_standard_date(r.get("arrival_date"))
                if not obs_date:
                    continue
                raw_comm = r.get("commodity", "")
                canonical_comm, _ = standardize_commodity_name(raw_comm)
                variety = str(r.get("variety", "Other")).strip()
                state = standardize_state_name(r.get("state", "Maharashtra"))
                raw_dist = r.get("district", "")
                district = standardize_district_name(raw_dist) if raw_dist else MH_MARKET_DISTRICTS.get(raw_market.lower(), raw_market.title())

                min_p = clean_numeric_price(r.get("min_price"))
                max_p = clean_numeric_price(r.get("max_price"))
                modal_p = clean_numeric_price(r.get("modal_price"))

                is_anomalous = False
                if min_p and max_p and min_p > max_p:
                    is_anomalous = True
                if modal_p and min_p and modal_p < min_p:
                    is_anomalous = True
                if modal_p and max_p and modal_p > max_p:
                    is_anomalous = True

                if modal_p is not None or min_p is not None or max_p is not None:
                    wholesale_rows.append({
                        "observation_date": obs_date,
                        "commodity_name": canonical_comm,
                        "raw_commodity_name": raw_comm,
                        "variety": variety,
                        "state": state,
                        "district": district,
                        "market_name": raw_market.title(),
                        "min_price_rs_per_qtl": min_p,
                        "max_price_rs_per_qtl": max_p,
                        "modal_price_rs_per_qtl": modal_p,
                        "price_unit": "Rs/Quintal",
                        "price_type": "Wholesale",
                        "is_anomalous_flag": is_anomalous,
                        "data_source": "data.gov.in AGMARKNET API (9ef84268-d588-465a-a308-a864a43d0070)"
                    })
        except Exception as e:
            logger.error(f"Error parsing data.gov.in file {dgf}: {e}")

    # Process wholesale dataframe
    wholesale_df = pd.DataFrame(wholesale_rows)
    if not wholesale_df.empty:
        before_cnt = len(wholesale_df)
        wholesale_df = wholesale_df.drop_duplicates(
            subset=["observation_date", "commodity_name", "market_name", "variety", "modal_price_rs_per_qtl"]
        )
        wholesale_df = wholesale_df.sort_values(by=["commodity_name", "observation_date", "market_name"])
        out_ws = PROCESSED_DIR / "wholesale_prices.csv"
        wholesale_df.to_csv(out_ws, index=False)
        logger.info(f"Cleaned {len(wholesale_df)} wholesale mandi records (from {before_cnt}). Saved to {out_ws}")
    else:
        logger.warning("No wholesale records found to process.")

    # Process arrival dataframe
    arrival_df = pd.DataFrame(arrival_rows)
    if not arrival_df.empty:
        before_cnt = len(arrival_df)
        arrival_df = arrival_df.drop_duplicates(
            subset=["observation_date", "commodity_name", "market_name", "variety", "arrival_quantity_tonnes"]
        )
        arrival_df = arrival_df.sort_values(by=["commodity_name", "observation_date", "market_name"])
        out_arr = PROCESSED_DIR / "market_arrivals.csv"
        arrival_df.to_csv(out_arr, index=False)
        logger.info(f"Cleaned {len(arrival_df)} market arrival records (from {before_cnt}). Saved to {out_arr}")
    else:
        logger.warning("No arrival records found to process.")

    return wholesale_df, arrival_df


def clean_cpi_data():
    """
    Cleans raw MoSPI CPI datasets from data/raw/cpi/:
    1. mospi_cpi_groups_*.json
    2. mospi_cpi_items_*.json
    Outputs data/processed/cpi_data.csv
    """
    logger.info("--- Cleaning MoSPI CPI Data ---")
    cpi_rows = []

    cpi_files = glob.glob("data/raw/cpi/mospi_cpi_*.json")
    for cf in cpi_files:
        try:
            with open(cf, "r", encoding="utf-8") as f:
                data = json.load(f)
            records = data.get("records", [])
            for r in records:
                year = r.get("year")
                month_name = str(r.get("month", "")).strip()
                if not year or not month_name:
                    continue
                # Convert month name to YYYY-MM-01
                date_str = f"{month_name} {year}"
                obs_date = parse_standard_date(date_str)
                if not obs_date:
                    continue

                index_val = clean_numeric_price(r.get("index"))
                inflation_val = clean_numeric_price(r.get("inflation"))
                state = standardize_state_name(r.get("state", "All India"))
                sector = str(r.get("sector", "Combined")).strip()

                # Check if group-level or item-level
                if "item" in r or "item_query_name" in r:
                    cat_type = "Item"
                    cat_name = r.get("item") or r.get("item_query_name")
                    subgroup = ""
                else:
                    cat_type = "Group / Subgroup"
                    cat_name = r.get("subgroup") or r.get("group", "Food and Beverages")
                    subgroup = r.get("subgroup", "")

                canonical_comm, raw_comm = standardize_commodity_name(cat_name)

                cpi_rows.append({
                    "observation_date": obs_date,
                    "year": int(year),
                    "month": month_name,
                    "geography": state,
                    "sector": sector,
                    "category_type": cat_type,
                    "category_name": canonical_comm,
                    "raw_category_name": cat_name,
                    "subgroup": subgroup,
                    "cpi_index": index_val,
                    "inflation_rate_yoy": inflation_val,
                    "base_year": r.get("baseyear", "2012"),
                    "data_source": "MoSPI Consumer Price Index API (api.mospi.gov.in)"
                })
        except Exception as e:
            logger.error(f"Error parsing CPI file {cf}: {e}")

    cpi_df = pd.DataFrame(cpi_rows)
    if not cpi_df.empty:
        before_cnt = len(cpi_df)
        cpi_df = cpi_df.drop_duplicates(
            subset=["observation_date", "geography", "sector", "category_name", "category_type"]
        )
        cpi_df = cpi_df.sort_values(by=["geography", "sector", "category_name", "observation_date"])
        out_cpi = PROCESSED_DIR / "cpi_data.csv"
        cpi_df.to_csv(out_cpi, index=False)
        logger.info(f"Cleaned {len(cpi_df)} CPI records (deduplicated from {before_cnt}). Saved to {out_cpi}")
    else:
        logger.warning("No CPI records found to process.")
    return cpi_df


def clean_household_expenditure_data():
    """
    Cleans MoSPI HCES datasets:
    1. hces_factsheet_extracted_statements.json (Statements 1, 4, 5, 8)
    2. mospi_hces_*.json (API records)
    Outputs data/processed/household_expenditure.csv
    """
    logger.info("--- Cleaning Household Expenditure Data ---")
    hces_rows = []

    # 1. Factsheet Statements
    fs_file = "data/raw/household_expenditure/hces_factsheet_extracted_statements.json"
    if os.path.exists(fs_file):
        try:
            with open(fs_file, "r", encoding="utf-8") as f:
                fs_data = json.load(f)

            # Statement 1: Food vs Non-Food shares
            st1 = fs_data.get("statement_1_food_share", [])
            for row in st1[3:]:
                clean_row = [str(c).replace("\n", " ").strip() for c in row if c]
                if len(clean_row) >= 5:
                    group_name = clean_row[0]
                    r_mpce = clean_numeric_price(clean_row[1])
                    r_share = clean_numeric_price(clean_row[2])
                    u_mpce = clean_numeric_price(clean_row[3])
                    u_share = clean_numeric_price(clean_row[4])
                    if r_mpce:
                        hces_rows.append({
                            "survey_year": "2022-23",
                            "geography": "All India",
                            "sector": "Rural",
                            "indicator_name": "Food vs Non-Food MPCE & Share",
                            "item_group": group_name,
                            "fractile_class": "All Classes",
                            "mpce_value_rs": r_mpce,
                            "share_percentage": r_share,
                            "data_source": "MoSPI NSSO HCES 2022-23 Factsheet Statement 1"
                        })
                    if u_mpce:
                        hces_rows.append({
                            "survey_year": "2022-23",
                            "geography": "All India",
                            "sector": "Urban",
                            "indicator_name": "Food vs Non-Food MPCE & Share",
                            "item_group": group_name,
                            "fractile_class": "All Classes",
                            "mpce_value_rs": u_mpce,
                            "share_percentage": u_share,
                            "data_source": "MoSPI NSSO HCES 2022-23 Factsheet Statement 1"
                        })

            # Statement 4: Fractile classes of MPCE
            st4 = fs_data.get("statement_4_fractile_classes", [])
            for row in st4[3:]:
                clean_row = [str(c).replace("\n", " ").strip() for c in row if c]
                if len(clean_row) >= 3:
                    f_class = clean_row[0]
                    r_val = clean_numeric_price(clean_row[1])
                    u_val = clean_numeric_price(clean_row[2])
                    if r_val:
                        hces_rows.append({
                            "survey_year": "2022-23",
                            "geography": "All India",
                            "sector": "Rural",
                            "indicator_name": "MPCE across Fractile Classes",
                            "item_group": "Total Consumption",
                            "fractile_class": f_class,
                            "mpce_value_rs": r_val,
                            "share_percentage": None,
                            "data_source": "MoSPI NSSO HCES 2022-23 Factsheet Statement 4"
                        })
                    if u_val:
                        hces_rows.append({
                            "survey_year": "2022-23",
                            "geography": "All India",
                            "sector": "Urban",
                            "indicator_name": "MPCE across Fractile Classes",
                            "item_group": "Total Consumption",
                            "fractile_class": f_class,
                            "mpce_value_rs": u_val,
                            "share_percentage": None,
                            "data_source": "MoSPI NSSO HCES 2022-23 Factsheet Statement 4"
                        })

            # Statement 5: Break-up of MPCE by food item groups
            st5 = fs_data.get("statement_5_item_groups", [])
            for row in st5[3:]:
                clean_row = [str(c).replace("\n", " ").strip() for c in row if c]
                if len(clean_row) >= 5:
                    item_grp = clean_row[0]
                    r_val = clean_numeric_price(clean_row[1])
                    r_pct = clean_numeric_price(clean_row[2])
                    u_val = clean_numeric_price(clean_row[3])
                    u_pct = clean_numeric_price(clean_row[4])
                    if r_val:
                        hces_rows.append({
                            "survey_year": "2022-23",
                            "geography": "All India",
                            "sector": "Rural",
                            "indicator_name": "MPCE and Share by Item Group",
                            "item_group": item_grp,
                            "fractile_class": "All Classes",
                            "mpce_value_rs": r_val,
                            "share_percentage": r_pct,
                            "data_source": "MoSPI NSSO HCES 2022-23 Factsheet Statement 5"
                        })
                    if u_val:
                        hces_rows.append({
                            "survey_year": "2022-23",
                            "geography": "All India",
                            "sector": "Urban",
                            "indicator_name": "MPCE and Share by Item Group",
                            "item_group": item_grp,
                            "fractile_class": "All Classes",
                            "mpce_value_rs": u_val,
                            "share_percentage": u_pct,
                            "data_source": "MoSPI NSSO HCES 2022-23 Factsheet Statement 5"
                        })

            # Statement 8: State-wise Average MPCE (including Maharashtra)
            st8 = fs_data.get("statement_8_state_mpce", [])
            for row in st8[3:]:
                clean_row = [str(c).replace("\n", " ").strip() for c in row if c]
                if len(clean_row) >= 3:
                    state_name = clean_row[0]
                    r_val = clean_numeric_price(clean_row[1])
                    u_val = clean_numeric_price(clean_row[2])
                    if r_val:
                        hces_rows.append({
                            "survey_year": "2022-23",
                            "geography": state_name,
                            "sector": "Rural",
                            "indicator_name": "State-wise Average MPCE",
                            "item_group": "Total Consumption",
                            "fractile_class": "All Classes",
                            "mpce_value_rs": r_val,
                            "share_percentage": None,
                            "data_source": "MoSPI NSSO HCES 2022-23 Factsheet Statement 8"
                        })
                    if u_val:
                        hces_rows.append({
                            "survey_year": "2022-23",
                            "geography": state_name,
                            "sector": "Urban",
                            "indicator_name": "State-wise Average MPCE",
                            "item_group": "Total Consumption",
                            "fractile_class": "All Classes",
                            "mpce_value_rs": u_val,
                            "share_percentage": None,
                            "data_source": "MoSPI NSSO HCES 2022-23 Factsheet Statement 8"
                        })
        except Exception as e:
            logger.error(f"Error parsing HCES factsheet statements: {e}")

    hces_df = pd.DataFrame(hces_rows)
    if not hces_df.empty:
        before_cnt = len(hces_df)
        hces_df = hces_df.drop_duplicates(
            subset=["survey_year", "geography", "sector", "indicator_name", "item_group", "fractile_class"]
        )
        out_hces = PROCESSED_DIR / "household_expenditure.csv"
        hces_df.to_csv(out_hces, index=False)
        logger.info(f"Cleaned {len(hces_df)} household expenditure records. Saved to {out_hces}")
    else:
        logger.warning("No HCES records found to process.")
    return hces_df


def main():
    logger.info("=== STARTING COMPLETE DATA CLEANING & STANDARDIZATION PIPELINE ===")
    clean_retail_data()
    clean_wholesale_and_arrivals_data()
    clean_cpi_data()
    clean_household_expenditure_data()
    logger.info("=== DATA CLEANING & STANDARDIZATION PIPELINE COMPLETED ===")

if __name__ == "__main__":
    main()
