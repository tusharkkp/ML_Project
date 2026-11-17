"""
Normalization and Standardization Mappings
Defines canonical names, units, conversion factors, and cleaning utilities
for agricultural commodities, geography, and price series.
"""

import re
from datetime import datetime

# Canonical Commodity Name Mappings
COMMODITY_MAPPINGS = {
    # Grains
    "rice": "Rice",
    "rice – pds": "Rice",
    "rice – other sources": "Rice",
    "rice (pds)": "Rice",
    "broken rice": "Rice (Broken)",
    "basmati rice": "Rice (Basmati)",
    "paddy": "Paddy",
    "wheat": "Wheat",
    "atta (wheat)": "Wheat (Atta)",
    "wheat/atta - pds": "Wheat",
    "wheat/atta - other sources": "Wheat",
    
    # Vegetables
    "tomato": "Tomato",
    "tomato local": "Tomato",
    "tomato hybrid": "Tomato",
    "onion": "Onion",
    "onion red": "Onion",
    "onion green": "Onion",
    "potato": "Potato",
    "potato jyoti": "Potato",
    "potato local": "Potato",
    
    # Pulses
    "tur/arhar dal": "Tur Dal",
    "arhar/tur dal": "Tur Dal",
    "tur dal": "Tur Dal",
    "arhar dal": "Tur Dal",
    "red gram split/arhar dal/tur dal": "Tur Dal",
    "pegeon pea(arhar fali)": "Tur Dal",
    "gram dal": "Gram Dal",
    "bengal gram dal(chana dal)": "Gram Dal",
    "bengal gram(gram)(whole)": "Gram Dal (Whole)",
    "moong dal": "Moong Dal",
    "green gram dal(moong dal)": "Moong Dal",
    "green gram(moong)(whole)": "Moong Dal (Whole)",
    "urad dal": "Urad Dal",
    "black gram dal(urd dal)": "Urad Dal",
    "black gram(urd beans)(whole)": "Urad Dal (Whole)",
    "masoor dal": "Masoor Dal",
    
    # Oils & Dairy
    "mustard oil": "Mustard Oil",
    "mustard oil (packed)": "Mustard Oil",
    "mustard": "Mustard Seed",
    "sunflower oil": "Sunflower Oil",
    "sunflower oil (packed)": "Sunflower Oil",
    "groundnut oil (packed)": "Groundnut Oil",
    "soya oil (packed)": "Soya Oil",
    "vanaspati (packed)": "Vanaspati",
    "palm oil (packed)": "Palm Oil",
    "milk": "Milk",
    "milk @": "Milk",
    "milk (liquid)": "Milk",
    "sugar": "Sugar",
    "gur": "Jaggery (Gur)",
    "tea loose": "Tea",
    "salt pack (iodised)": "Salt"
}

# Geography Mappings
STATE_MAPPINGS = {
    "maharashtra": "Maharashtra",
    "mh": "Maharashtra",
    "all india": "All India",
    "all-india": "All India",
    "national": "All India"
}

DISTRICT_MAPPINGS = {
    "pune": "Pune",
    "mumbai": "Mumbai",
    "nashik": "Nashik",
    "nasik": "Nashik",
    "nagpur": "Nagpur",
    "ahmednagar": "Ahmednagar",
    "ahmadnagar": "Ahmednagar",
    "ahilyanagar": "Ahmednagar",
    "aurangabad": "Chhatrapati Sambhajinagar",
    "chhatrapati sambhajinagar": "Chhatrapati Sambhajinagar",
    "osmanabad": "Dharashiv",
    "dharashiv": "Dharashiv",
    "solapur": "Solapur",
    "kolhapur": "Kolhapur",
    "jalgaon": "Jalgaon",
    "amravati": "Amravati",
    "satara": "Satara",
    "sangli": "Sangli"
}

def standardize_commodity_name(raw_name):
    """Returns (canonical_name, raw_name)."""
    if not raw_name or not str(raw_name).strip():
        return "Unknown", raw_name
    cleaned = str(raw_name).strip().lower()
    for key, canonical in COMMODITY_MAPPINGS.items():
        if cleaned == key or cleaned.startswith(key):
            return canonical, raw_name
    return str(raw_name).strip().title(), raw_name

def standardize_state_name(raw_state):
    """Standardizes state name."""
    if not raw_state:
        return "Unknown"
    cleaned = str(raw_state).strip().lower()
    return STATE_MAPPINGS.get(cleaned, str(raw_state).strip().title())

def standardize_district_name(raw_district):
    """Standardizes district name."""
    if not raw_district:
        return ""
    cleaned = str(raw_district).strip().lower()
    return DISTRICT_MAPPINGS.get(cleaned, str(raw_district).strip().title())

def parse_standard_date(raw_date):
    """
    Parses various date formats to ISO YYYY-MM-DD.
    Supports DD/MM/YYYY, YYYY-MM-DD, YYYY_MM_DD, DD-MM-YYYY, Month YYYY.
    """
    if not raw_date or str(raw_date).strip().lower() in ["none", "nan", "nat", ""]:
        return None
    s = str(raw_date).strip()

    formats = [
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%Y_%m_%d",
        "%d-%m-%Y",
        "%d/%m/%y",
        "%Y/%m/%d",
        "%d-%b-%Y",
        "%B %Y"
    ]
    for fmt in formats:
        try:
            dt = datetime.strptime(s, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Regex search for DD/MM/YYYY or YYYY-MM-DD inside strings
    m = re.search(r"(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})", s)
    if m:
        try:
            d, month, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return f"{y:04d}-{month:02d}-{d:02d}"
        except Exception:
            pass

    return None

def clean_numeric_price(val):
    """Converts price strings to clean float or None."""
    if val is None:
        return None
    s = str(val).replace(",", "").replace("--", "").strip()
    if not s or s.lower() in ["nr", "na", "null", "none", "-", ""]:
        return None
    m = re.search(r"[-+]?\d*\.?\d+", s)
    if m:
        try:
            num = float(m.group(0))
            return num if num > 0 else None
        except ValueError:
            return None
    return None
