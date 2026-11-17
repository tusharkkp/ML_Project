"""
Source Discovery and Probing Script
Investigates candidate data sources for the Food Inflation Project:
1. data.gov.in
2. AGMARKNET (agmarknet.gov.in)
3. CEDA (ceda.ashoka.edu.in)
4. DCA Price Monitoring (fcainfoweb.nic.in)
5. MoSPI (esankhyiki.mospi.gov.in, api.mospi.gov.in)
6. Additional economic/commodity sources
"""

import sys
import os
import json
import urllib.request
import urllib.error
import ssl
import requests
from bs4 import BeautifulSoup

# Disable SSL verification warnings if government portals have cert chain quirks
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 (Academic Research: Food Inflation Project)"
}

def probe_url(url, method="GET", params=None, timeout=15, verify_ssl=False):
    result = {
        "url": url,
        "status_code": None,
        "accessible": False,
        "content_type": None,
        "content_length": None,
        "redirect_url": None,
        "error": None,
        "snippet": ""
    }
    try:
        resp = requests.request(method, url, headers=HEADERS, params=params, timeout=timeout, verify=verify_ssl, allow_redirects=True)
        result["status_code"] = resp.status_code
        result["accessible"] = (200 <= resp.status_code < 400) or resp.status_code == 401 or resp.status_code == 403
        result["content_type"] = resp.headers.get("Content-Type", "")
        result["content_length"] = len(resp.content)
        result["redirect_url"] = resp.url if resp.url != url else None
        
        # Capture snippet
        text = resp.text[:1000].strip()
        result["snippet"] = text.replace("\n", " ").replace("\r", " ")[:300]
    except Exception as e:
        result["error"] = str(e)
    return result

def check_robots(base_url):
    robots_url = base_url.rstrip("/") + "/robots.txt"
    res = probe_url(robots_url, timeout=10)
    return {
        "url": robots_url,
        "status": res["status_code"],
        "content": res["snippet"] if res["accessible"] else "Not found/inaccessible"
    }

def main():
    print("=== STARTING SOURCE DISCOVERY PROBE ===\n")
    
    candidates = [
        # Source 1: data.gov.in
        ("data.gov.in API without key (resource 9ef84268-d588-465a-a308-a864a43d0070)", 
         "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070?format=json&limit=2"),
        ("data.gov.in catalog page", 
         "https://www.data.gov.in/catalog/current-daily-price-various-commodities-various-markets-mandi"),
        
        # Source 2: AGMARKNET
        ("AGMARKNET Portal Home", "https://agmarknet.gov.in/"),
        ("AGMARKNET PriceTrends", "https://agmarknet.gov.in/PriceTrends/SearchPriceTrends.aspx"),
        ("AGMARKNET robots.txt", "https://agmarknet.gov.in/robots.txt"),
        
        # Source 3: CEDA
        ("CEDA Data Portal", "https://ceda.ashoka.edu.in/data-portal/"),
        ("CEDA DCA Food Prices Portal", "https://dca.ceda.ashoka.edu.in/"),
        ("CEDA Agmarknet Portal", "https://agmarknet.ceda.ashoka.edu.in/"),
        ("CEDA API Documentation", "https://api.ceda.ashoka.edu.in/documentation/"),
        ("CEDA Agmarknet API endpoint test", "https://api.ceda.ashoka.edu.in/api/v1/agmarknet/"),
        
        # Source 4: Department of Consumer Affairs (DCA)
        ("DCA fcainfoweb Home", "https://fcainfoweb.nic.in/"),
        ("DCA DBprices Daily Report", "https://fcainfoweb.nic.in/Reports/DB/DBprices.aspx"),
        ("DCA Report Menu Web", "https://fcainfoweb.nic.in/reports/report_menu_web.aspx"),
        ("DCA Price Monitoring Division", "https://consumeraffairs.nic.in/price-monitoring-cell"),
        
        # Source 5: MoSPI
        ("MoSPI Home", "https://www.mospi.gov.in/"),
        ("MoSPI eSankhyiki CPI", "https://esankhyiki.mospi.gov.in/catalogue-main/catalogue?product=CPI"),
        ("MoSPI API Portal", "https://api.mospi.gov.in/"),
        
        # Source 6: World Bank & RBI
        ("World Bank Food Inflation Indicators", "https://microdata.worldbank.org/"),
        ("RBI DBIE Portal", "https://dbie.rbi.org.in/")
    ]
    
    findings = []
    for name, url in candidates:
        print(f"Probing: {name} ({url})...")
        res = probe_url(url)
        findings.append({"name": name, "res": res})
        print(f"  Status: {res['status_code']} | Content-Type: {res['content_type']} | Error: {res['error']}")
        if res['snippet']:
            print(f"  Snippet: {res['snippet'][:120]}...")
        print()

    # Save probe findings to metadata
    out_path = os.path.join("data", "metadata", "initial_probe_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(findings, f, indent=2)
    print(f"Saved initial probe findings to {out_path}")

if __name__ == "__main__":
    main()
