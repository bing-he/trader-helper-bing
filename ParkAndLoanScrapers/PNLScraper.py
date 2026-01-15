# This script uses the Selenium library to automate a web browser,
# navigate multiple websites, scrape data from various pages,
# and consolidate it into a single CSV file.

import time
import os
import csv
import random
from datetime import datetime
import re
import requests
import pdfplumber
from io import BytesIO

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException

# NEW: extra deps for TCE PAL
import io
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
import numpy as np
import openpyxl
import html
import pandas as pd
from typing import Optional, List, Dict, Any, Sequence, Tuple

# --- Helper Functions ---

def parse_date(date_str):
    """
    Parse a date string from multiple known formats into a datetime object.
    """
    for fmt in ["%B %d, %Y", "%d-%b-%y", "%m/%d/%Y"]:
        try:
            return datetime.strptime(str(date_str).strip(), fmt)
        except (ValueError, TypeError):
            pass
    raise ValueError(f"Date '{date_str}' could not be parsed.")

def calculate_months(start_date_str, end_date_str):
    """Calculate the number of months between two date strings (original format)."""
    try:
        start_date = datetime.strptime(start_date_str, "%B %d, %Y")
        end_date = datetime.strptime(end_date_str, "%B %d, %Y")
        return (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
    except (ValueError, TypeError):
        return 0

# --- Scraper Functions ---

def scrape_report_data_original(driver, url):
    """Scrape data from the original gasnom.com site format."""
    print(f"Scraping (Original Format): {url}")
    driver.get(url)
    def get_data_value(label_text):
        try:
            element = driver.find_element(By.XPATH, f"//td[normalize-space()='{label_text}']/following-sibling::td")
            return element.text.strip()
        except NoSuchElementException:
            return "N/A"

    rate_schedule = get_data_value("Rate Schedule")
    record_type = {"EPS": "Park", "ELS": "Loan"}.get(rate_schedule)
    if not record_type:
        print(f"   - Skipping: Rate Schedule '{rate_schedule}' is not EPS/ELS.")
        return None

    posting_date = get_data_value("Posting Date")
    end_date = get_data_value("Rate End Date")
    rate_charged_str = get_data_value("Rate Charged")

    true_rate = "N/A"
    if posting_date != "N/A" and end_date != "N/A":
        num_months = calculate_months(posting_date, end_date)
        try:
            rate_charged_float = float(rate_charged_str)
            true_rate = f"{(num_months * rate_charged_float):.4f}"
        except (ValueError, TypeError):
            pass

    quantity = get_data_value("Interruptible Quantity - Contract").replace(",", "")

    return {
        "Posting Date": posting_date,
        "Counter Party": get_data_value("Contract Holder Name"),
        "Type": record_type,
        "Quantity": quantity,
        "End Date": end_date,
        "True Rate": true_rate,
        "Location": url.split('/')[4],
        "Source URL": url
    }

def scrape_report_data_modern(driver, url, location_key):
    """Scrape data from the modern gasnom.com site format."""
    print(f"Scraping (Modern Format): {url}")
    driver.get(url)
    results = []
    try:
        posting_date = driver.find_element(
            By.XPATH, "//td[normalize-space()='Posting Date']/following-sibling::td").text.strip()
        counter_party = driver.find_element(
            By.XPATH, "//td[normalize-space()='Contract Holder Name']/following-sibling::td").text.strip()
        quantity = driver.find_element(
            By.XPATH, "//td[normalize-space()='Interruptible Quantity - Contract']/following-sibling::td"
        ).text.strip().replace(",", "")
    except NoSuchElementException as e:
        print(f"   - Could not find a required shared field. Skipping. Error: {e}")
        return []

    for record_type in ["Park", "Loan"]:
        try:
            legend = driver.find_element(By.XPATH, f"//legend[contains(text(), '{record_type}')]")
            fieldset = legend.find_element(By.XPATH, "./..")
            rate_charged = fieldset.find_element(
                By.XPATH, ".//td[contains(text(), 'Rate Charged - (All locations)')]/following-sibling::td"
            ).text.strip()
            results.append({
                "Posting Date": posting_date,
                "Counter Party": counter_party,
                "Type": record_type,
                "Quantity": quantity,
                "End Date": "N/A",
                "True Rate": rate_charged,
                "Location": location_key,
                "Source URL": url
            })
        except NoSuchElementException:
            continue
    return results

def scrape_report_data_enbridge(pdf_url, location_key):
    """Download and scrape data from an Enbridge PDF report."""
    print(f"Scraping (Enbridge PDF Format): {pdf_url}")
    results = []
    try:
        response = requests.get(pdf_url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        pdf_content = BytesIO(response.content)
        print("   - PDF downloaded successfully.")

        with pdfplumber.open(pdf_content) as pdf:
            first_page_text = pdf.pages[0].extract_text(x_tolerance=2) or ""
            post_date_match = re.search(r'Post Date:\s*([\d/]+)', first_page_text)
            if not post_date_match:
                print("   - WARNING: No Post Date found on first page.")
                return results
            post_date_obj = parse_date(post_date_match.group(1))
            post_date = post_date_obj.strftime("%B %d, %Y")
            full_text = "\n".join((p.extract_text(x_tolerance=2) or "") for p in pdf.pages)

        record_starts = [m.start() for m in re.finditer(r'K\s*Holder\s*Name\s*:', full_text)]

        for i, start in enumerate(record_starts):
            end = record_starts[i + 1] if i + 1 < len(record_starts) else len(full_text)
            record_text = full_text[start:end]
            record_flat = re.sub(r'\s+', ' ', record_text)

            if "Svc Req K:" not in record_text:
                continue
            rate_sch_match = re.search(r'Rate\s*Sch:\s*([A-Z]+)', record_text)
            if not rate_sch_match:
                continue
            rate_sch = rate_sch_match.group(1).strip()
            record_type = {"EPS": "Park", "ELS": "Loan"}.get(rate_sch)
            if not record_type:
                continue

            cp_match = re.search(r'K\s*Holder\s*Name\s*:\s*(.*?)\s*K\s*Holder\s*Prop', record_text, re.DOTALL)
            counter_party = cp_match.group(1).strip().replace('\n', ' ') if cp_match else "N/A"

            eq_match = re.search(
                r'K\s*Ent\s*Beg\s*Date\s*K\s*Ent\s*End\s*Date\s*IT\s*Qty\s*-\s*K.*?\b'
                r'(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+([\d,]+)',
                record_text, re.DOTALL
            )
            end_date = eq_match.group(2) if eq_match else "N/A"
            quantity = eq_match.group(3).replace(",", "") if eq_match else "N/A"

            rate_match = re.search(
                r'(?:Park|Loan)\s*Charge\s*-\s*(?:Balance|Over\s*Term|[A-Za-z/ ]+)?\s*NA\s*([0-9.]+)',
                record_flat, re.IGNORECASE
            )
            true_rate = rate_match.group(1) if rate_match else "N/A"

            if end_date == "N/A" or quantity == "N/A":
                continue

            results.append({
                "Posting Date": post_date,
                "Counter Party": counter_party,
                "Type": record_type,
                "Quantity": quantity,
                "End Date": end_date,
                "True Rate": true_rate,
                "Location": location_key,
                "Source URL": pdf_url,
            })
            print(f"   - SUCCESS: Found {record_type} record for {counter_party} @ {true_rate}")

    except Exception as e:
        print(f"   - ERROR: An unexpected error occurred. {e}")
    return results

# --- Link Discovery Functions ---

def scrape_gasnom_links(driver, direct_url):
    """Navigate to a gasnom.com reports page and extract all 'View' link URLs."""
    wait = WebDriverWait(driver, 30)
    facility_name = direct_url.split('/')[-2]
    print(f"--- Starting Link Extraction for {facility_name} ---")
    driver.get(direct_url)
    try:
        wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, "a.trview")))
        print("   - Data table loaded successfully.")
    except TimeoutException:
        print(f"   - Could not find report links for {facility_name}.")
        return []
    elements = driver.find_elements(By.CSS_SELECTOR, "a.trview")
    urls = [link.get_attribute('href') for link in elements if link.get_attribute('href')]
    print(f"   - Found {len(urls)} links for {facility_name}.")
    return urls

def scrape_enbridge_links(driver, direct_url, last_known_date=None):
    """Navigate to an Enbridge reports list page and extract PDF links."""
    location_name = direct_url.split('pipe=')[-1].split('&')[0]
    print(f"--- Starting Link Extraction for {location_name} ---")
    driver.get(direct_url)
    cutoff_date = last_known_date if last_known_date else datetime(2025, 4, 1)
    date_info = (f"newer than {cutoff_date.strftime('%Y-%m-%d')}"
                 if last_known_date else f"since {cutoff_date.strftime('%Y-%m-%d')}")
    print(f"   - Looking for reports {date_info}.")
    urls = []
    try:
        wait = WebDriverWait(driver, 20)
        rows = wait.until(EC.presence_of_all_elements_located((By.XPATH, "//table//tr[td]")))
        for row in rows:
            cells = row.find_elements(By.TAG_NAME, "td")
            if len(cells) == 2:
                date_str = cells[0].text
                link_element = cells[1].find_element(By.TAG_NAME, "a")
                if "[NO DATA]" in link_element.text:
                    continue
                report_date = parse_date(date_str)
                if report_date > cutoff_date:
                    full_url = link_element.get_attribute('href')
                    if full_url and not full_url.startswith('javascript'):
                        urls.append(full_url)
    except Exception as e:
        print(f"   - Could not find or process report links. Error: {e}")
    print(f"   - Found {len(urls)} new links for {location_name}.")
    return urls

# --- CSV and State Management ---

def get_max_ids_by_location(file_path):
    """Read the CSV and return a dictionary of location to its highest transaction ID."""
    max_ids = {}
    if not os.path.exists(file_path):
        return max_ids
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location = row.get('Location')
                source_url = row.get('Source URL')
                if location and source_url and 'id=' in source_url:
                    try:
                        url_id = int(source_url.split('id=')[-1])
                        if url_id > max_ids.get(location, 0):
                            max_ids[location] = url_id
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Warning: Could not read existing CSV. Error: {e}")
    return max_ids

def get_latest_date_for_location(file_path, location_name):
    """Find the most recent 'Posting Date' for a given location in the CSV."""
    latest_date = None
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('Location') == location_name:
                    try:
                        current_date = parse_date(row['Posting Date'])
                        if latest_date is None or current_date > latest_date:
                            latest_date = current_date
                    except (ValueError, KeyError):
                        continue
    except Exception as e:
        print(f"Warning: Could not read dates for {location_name}. Error: {e}")
    return latest_date

# --- CSV and State Management ---

def get_max_ids_by_location(file_path):
    """Read the CSV and return a dictionary of location to its highest transaction ID."""
    max_ids = {}
    if not os.path.exists(file_path):
        return max_ids
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location = row.get('Location')
                source_url = row.get('Source URL')
                if location and source_url and 'id=' in source_url:
                    try:
                        url_id = int(source_url.split('id=')[-1])
                        if url_id > max_ids.get(location, 0):
                            max_ids[location] = url_id
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        print(f"Warning: Could not read existing CSV. Error: {e}")
    return max_ids

def get_latest_date_for_location(file_path, location_name):
    """Find the most recent 'Posting Date' for a given location in the CSV."""
    latest_date = None
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row.get('Location') == location_name:
                    try:
                        current_date = parse_date(row['Posting Date'])
                        if latest_date is None or current_date > latest_date:
                            latest_date = current_date
                    except (ValueError, KeyError):
                        continue
    except Exception as e:
        print(f"Warning: Could not read dates for {location_name}. Error: {e}")
    return latest_date

def print_last_update_summary(file_path):
    """
    Read the final CSV and print a summary of the latest 'Park' or 'PAL'
    posting date per location, sorted with the newest date first.
    """
    latest_dates = {}
    if not os.path.exists(file_path):
        print("\n--- No CSV file found to generate a summary. ---")
        return
    try:
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                location = row.get("Location")
                date_str = row.get("Posting Date")
                record_type = row.get("Type", "").upper() # Get the type

                # --- MODIFIED CHECK ---
                # Only consider 'Park' or 'PAL' records for this summary
                if location and date_str and ("PARK" in record_type or "PAL" in record_type):
                # --- END MODIFIED CHECK ---
                    try:
                        current_date = parse_date(date_str)
                        if location not in latest_dates or current_date > latest_dates[location]:
                            latest_dates[location] = current_date
                    except ValueError:
                        continue
    except Exception as e:
        print(f"\nCould not generate summary. Error reading CSV: {e}")
        return

    print("\n--- Scrape Health Summary (Latest Park/PAL) ---") # Updated title
    print(f"Checked on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if not latest_dates:
        print("  No valid 'Park' or 'PAL' dates found in the file.") # Updated message
        return

    # --- THIS IS THE SORTING CHANGE ---
    # Sort by date (value, item[1]) in descending order
    sorted_summary = sorted(latest_dates.items(), key=lambda item: item[1], reverse=True)
    # --- END SORTING CHANGE ---

    print("  Pipelines ordered by most recent 'Park'/'PAL' posting:")
    for location, date_obj in sorted_summary:
        # Format date first (YYYY-MM-DD) for easier scanning
        print(f"    {date_obj.strftime('%Y-%m-%d')} : {location}")

# ===========================
# TCE PAL (TCO/Northern Border/Columbia Gulf/Millennium) Helpers
# ===========================
TCE_PIPELINES = [
    {"name": "TCO",             "url": "https://ebb.tceconnects.com/infopost/ReportViewer.aspx?/InfoPost/TransInterrupt&AssetNbr=51"},
    {"name": "Northern Border", "url": "https://ebb.tceconnects.com/infopost/ReportViewer.aspx?/InfoPost/TransInterrupt&AssetNbr=3029"},
    {"name": "Columbia Gulf",   "url": "https://ebb.tceconnects.com/infopost/ReportViewer.aspx?/InfoPost/TransInterrupt&AssetNbr=14"},
    {"name": "Millennium",      "url": "https://ebb.tceconnects.com/infopost/ReportViewer.aspx?/InfoPost/TransInterrupt&AssetNbr=26"},
]
_PIPELINE_LOCATION_PREF = {
    "Northern Border": "location name",
    "TCO":             "loc zn",
    "Columbia Gulf":   "loc zn",
    "Millennium":      "loc zn",
}

def _tce_log(msg): print(msg, flush=True)

def _tce_safe_to_dt(x):
    if isinstance(x, datetime): return x
    if x is None: return None
    if isinstance(x, float):
        try:
            if np.isnan(x) or np.isinf(x): return None
        except Exception:
            pass
    try:
        return pd.to_datetime(x, errors="raise")
    except Exception:
        return None

def _tce_extract_export_url_base(page_html: str, base_url: str) -> str:
    m = re.search(r'"ExportUrlBase"\s*:\s*"([^"]+)"', page_html) or \
        re.search(r"ExportUrlBase\s*[:=]\s*['\"]([^'\"]+)['\"]", page_html)
    if not m: raise ValueError("TCE: ExportUrlBase not found.")
    raw = m.group(1)
    unescaped = html.unescape(raw).replace("\\u0026", "&").replace("\\/", "/")
    abs_url = urljoin(base_url, unescaped)
    parsed = urlparse(abs_url)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))
    q.pop("Format", None); q.pop("rs:Format", None)
    q["ContentDisposition"] = "AlwaysAttachment"
    return urlunparse(parsed._replace(query=urlencode(q, doseq=True)))

def _tce_download_export(viewer_url: str) -> bytes:
    base_origin = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(viewer_url))
    with requests.Session() as s:
        s.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
        _tce_log(f"Fetching TCE page: {viewer_url}")
        r0 = s.get(viewer_url, timeout=60); r0.raise_for_status()
        base_url = _tce_extract_export_url_base(r0.text, base_origin)
        parsed = urlparse(base_url)
        base_q = dict(parse_qsl(parsed.query, keep_blank_values=True))
        base_q["ContentDisposition"] = "AlwaysAttachment"
        candidates = [{"rs:Format":"EXCELOPENXML"}, {"Format":"EXCELOPENXML"}, {"rs:Format":"CSV"}]
        last_err = None
        for fmt in candidates:
            q = {k:v for k,v in base_q.items() if k not in ("Format","rs:Format")}
            q.update(fmt)
            url = urlunparse(parsed._replace(query=urlencode(q, doseq=True)))
            _tce_log(f"  -> Exporting with {fmt} …")
            r = s.get(url, headers={"Referer": viewer_url}, timeout=120)
            try:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type","")
                is_xlsx = r.content.startswith(b"PK")
                is_csv  = ("csv" in ctype.lower()) or (not is_xlsx and r.text[:1] in ('"', ',', 'P', 'D'))
                if is_xlsx or is_csv: return r.content
                else: last_err = RuntimeError(f"Unexpected content-type {ctype}")
            except Exception as e:
                last_err = e
        raise RuntimeError(f"TCE export failed. Last error: {last_err}")

def _tce_rows_from_bytes(raw: bytes):
    if raw.startswith(b"PK"):
        wb = openpyxl.load_workbook(io.BytesIO(raw), data_only=True, read_only=True)
        ws = wb.active
        return [tuple(r) for r in ws.iter_rows(values_only=True)]
    try:
        txt = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        txt = raw.decode("latin-1", errors="replace")
    rdr = csv.reader(io.StringIO(txt))
    return [tuple(row) for row in rdr]

def _tce_find_main_header(rows):
    for i in range(0, min(len(rows), 120)):
        r = rows[i]
        if not r: continue
        lower = [str(c).strip().lower() if c is not None else "" for c in r]
        if any("posting date" in cell for cell in lower) and any("rate schedule" in cell for cell in lower):
            colmap = {}
            for ci, name in enumerate(r):
                if name is not None:
                    nm = str(name).strip().lower()
                    if nm: colmap[nm] = ci
            return i, colmap
    return None, {}

def _tce_find_rate_header(rows, start_idx, window=12):
    alt = {
        "beg":{"disc beg date","begin date","start date","beg date","discount begin date"},
        "end":{"disc end date","end date","discount end date","disc end"},
        "rate":{"rate chgd","rate charged","rate","daily rate"},
    }
    for j in range(start_idx, min(len(rows), start_idx+window)):
        r = rows[j]
        if not r: continue
        lower = [str(c).strip().lower() if c is not None else "" for c in r]
        ok_beg  = any(any(k in cell for k in alt["beg"])  for cell in lower)
        ok_end  = any(any(k in cell for k in alt["end"])  for cell in lower)
        ok_rate = any(any(k in cell for k in alt["rate"]) for cell in lower)
        if ok_beg and ok_end and ok_rate: return j
    return None

def _tce_map_rate_cols(header_row):
    idx = {"beg":0,"end":3,"rate":8}
    lower = [str(c).strip().lower() if c is not None else "" for c in header_row]
    for ci, txt in enumerate(lower):
        if any(k in txt for k in ["disc beg date","begin date","start date","beg date"]): idx["beg"]=ci
        if any(k in txt for k in ["disc end date","end date"]): idx["end"]=ci
        if any(k in txt for k in ["rate chgd","rate charged","daily rate"]) or txt=="rate": idx["rate"]=ci
    return idx

def _tce_walk_rate_block(rows, header_idx):
    start = header_idx+1
    cmap = _tce_map_rate_cols(rows[header_idx])
    k = start
    while k < len(rows):
        r = rows[k]
        if not r or (len(r)>0 and (r[0] is None or str(r[0]).strip()=="")): break
        k += 1
    return start, k, rows[start:k], cmap

def _tce_find_location_header(rows, after_idx, window=8):
    j = after_idx
    if j < len(rows):
        r = rows[j]
        if (not r) or (len(r)>0 and (r[0] is None or str(r[0]).strip()=="")): j += 1
    for k in range(j, min(len(rows), j+window)):
        r = rows[k]
        if not r: continue
        lower = [str(c).strip().lower() if c is not None else "" for c in r]
        if any(("loc zn" in cell) or ("location name" in cell) for cell in lower): return k
    return None

def _tce_locate_loc_cols(header_row):
    res = {}
    lower = [str(c).strip().lower() if c is not None else "" for c in header_row]
    for ci, txt in enumerate(lower):
        if "loc zn" in txt: res["loc zn"]=ci
        if "location name" in txt: res["location name"]=ci
    return res

def _tce_pick_location(loc_row, cols, pipeline_name):
    pref = _PIPELINE_LOCATION_PREF.get(pipeline_name, "loc zn")
    alt  = "location name" if pref=="loc zn" else "loc zn"
    if pref in cols and cols[pref] < len(loc_row):
        v = loc_row[cols[pref]]
        if v is not None and str(v).strip(): return str(v).strip()
    if alt in cols and cols[alt] < len(loc_row):
        v = loc_row[cols[alt]]
        if v is not None and str(v).strip(): return str(v).strip()
    return "N/A"

def tce_fetch_all_pal_rows(cutoff_date: Optional[datetime]) -> list[dict]:
    all_rows = []
    for p in TCE_PIPELINES:
        try:
            raw = _tce_download_export(p["url"])
            rows = _tce_rows_from_bytes(raw)
            _tce_log(f"[{p['name']}] rows: {len(rows)}")
            hdr_idx, cmap = _tce_find_main_header(rows)
            if hdr_idx is None:
                start_main = 12; posting_col=0; rate_sched_col=25; counterparty_col=9
            else:
                start_main = hdr_idx+1
                posting_col = cmap.get("posting date/time", cmap.get("posting date", 0))
                rate_sched_col = cmap.get("rate schedule", 25)
                counterparty_col = cmap.get("contract holder name", cmap.get("contract holder", 9))
            if p["name"] == "Millennium":
                rate_sched_col = 24  # Millennium override: Rate Schedule in column Y

            for i in range(start_main, len(rows)):
                row = rows[i]
                if not row: continue
                a = row[posting_col] if posting_col < len(row) else None
                sched = row[rate_sched_col] if rate_sched_col < len(row) else None
                post_dt = _tce_safe_to_dt(a)
                if post_dt is None: continue
                if str(sched).strip().upper() != "PAL": continue
                if cutoff_date and post_dt <= cutoff_date: continue

                cp = (str(row[counterparty_col]).strip()
                      if counterparty_col < len(row) and row[counterparty_col] is not None else "N/A")

                r_hdr = _tce_find_rate_header(rows, i+3, 12) or _tce_find_rate_header(rows, i+1, 16)
                if r_hdr is None: continue
                d_start, d_end, rate_rows, cmap_rate = _tce_walk_rate_block(rows, r_hdr)
                if not rate_rows: continue

                end_cell = rate_rows[-1][cmap_rate["end"]] if len(rate_rows[-1]) > cmap_rate["end"] else None
                end_dt = _tce_safe_to_dt(end_cell) or _tce_safe_to_dt(rate_rows[-1][1] if len(rate_rows[-1])>1 else None)
                if end_dt is None: continue

                true_rate = 0.0
                if len(rate_rows) >= 2:
                    tr = rate_rows[-2]
                    beg = _tce_safe_to_dt(tr[cmap_rate["beg"]] if len(tr)>cmap_rate["beg"] else None)
                    ed  = _tce_safe_to_dt(tr[cmap_rate["end"]] if len(tr)>cmap_rate["end"] else None)
                    rate_val = pd.to_numeric(tr[cmap_rate["rate"]] if len(tr)>cmap_rate["rate"] else None, errors="coerce")
                    if beg is not None and ed is not None and pd.notna(rate_val):
                        days = (ed.date()-beg.date()).days + 1
                        if days >= 0: true_rate = float(days)*float(rate_val)

                loc_hdr = _tce_find_location_header(rows, d_end)
                location = "N/A"
                if loc_hdr is not None and (loc_hdr+1) < len(rows):
                    cols = _tce_locate_loc_cols(rows[loc_hdr])
                    lrow = rows[loc_hdr+1]
                    if lrow: location = _tce_pick_location(lrow, cols, p["name"])

                all_rows.append({
                    "Posting Date": post_dt.strftime("%d-%b-%y"),
                    "Counter Party": cp,
                    "Type": "PAL",
                    "Quantity": "N/A",
                    "End Date": end_dt.strftime("%d-%b-%y"),
                    "True Rate": f"{true_rate:.4f}",
                    "Location": location,
                    "Source URL": p["url"],
                })
            _tce_log(f"[{p['name']}] cumulative: {len(all_rows)} rows")
        except Exception as e:
            _tce_log(f"[{p['name']}] ERROR: {e}")
    return all_rows

# --- Main Execution Block ---

def main():
    """
    Main function to orchestrate the scraping process.
    """
    SITES = {
        "1": {"name": "Pine Prairie", "url": "https://www.gasnom.com/ip/PinePrairie/tr_list.cfm?type=2", "location_key": "PinePrairie", "scraper_type": "original"},
        "2": {"name": "Southern Pines", "url": "https://www.gasnom.com/ip/SOUTHERNPINES/tr_list.cfm?type=2", "location_key": "SOUTHERNPINES", "scraper_type": "original"},
        "3": {"name": "Mississippi Hub", "url": "https://www.gasnom.com/ip/mississippihub/tr_list.cfm?type=2", "location_key": "MissHub", "scraper_type": "modern"},
        "4": {"name": "Golden Triangle", "url": "https://www.gasnom.com/ip/goldentriangle/tr_list.cfm?type=2", "location_key": "GoldTri", "scraper_type": "modern"},
        "5": {"name": "Caledonia", "url": "http://www.gasnom.com/ip/caledonia/tr_list.cfm?type=2", "location_key": "Caledonia", "scraper_type": "modern"},
        "6": {"name": "Bobcat", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=BGS&type=TRI", "location_key": "Bobcat", "scraper_type": "enbridge_pdf"},
        "7": {"name": "Algonquin", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=AG&type=TRI", "location_key": "Algonquin", "scraper_type": "enbridge_pdf"},
        "8": {"name": "BigSandy", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=BSP&type=TRI", "location_key": "BigSandy", "scraper_type": "enbridge_pdf"},
        "9": {"name": "Egan", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=EG&type=TRI", "location_key": "Egan", "scraper_type": "enbridge_pdf"},
        "10": {"name": "EastTenn", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=ET&type=TRI", "location_key": "EastTenn", "scraper_type": "enbridge_pdf"},
        "11": {"name": "NEXUS", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=NXUS&type=TRI", "location_key": "NEXUS", "scraper_type": "enbridge_pdf"},
        "12": {"name": "SESH", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=SESH&type=TRI", "location_key": "SESH", "scraper_type": "enbridge_pdf"},
        "13": {"name": "Saltville", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=SG&type=TRI", "location_key": "Saltville", "scraper_type": "enbridge_pdf"},
        "14": {"name": "Steckman", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=SR&type=TRI", "location_key": "Steckman", "scraper_type": "enbridge_pdf"},
        "15": {"name": "Sabal", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=STT&type=TRI", "location_key": "Sabal", "scraper_type": "enbridge_pdf"},
        "16": {"name": "TresPalacios", "url": "https://infopost.enbridge.com/infopost/TransRptList.asp?pipe=TPGS&type=TRI", "location_key": "TresPalacios", "scraper_type": "enbridge_pdf"},
        "17": {"name": "TCE PAL (TCO/Northern Border/Columbia Gulf/Millennium)", "url": "TCE_PAL", "location_key": "TCE_PAL", "scraper_type": "tce_pal"},
    }
    
    print("Please choose which site(s) to scrape:")
    for key, site in SITES.items():
        print(f"  {key}: {site['name']}")
    print("  A: All Sites")

    choice_input = input("Enter your choice(s), separated by commas (e.g., 1,3,6): ").strip().upper()

    sites_to_process = []
    if choice_input == 'A':
        sites_to_process = list(SITES.values())
    else:
        choices = [c.strip() for c in choice_input.split(',')]
        for choice in choices:
            if choice in SITES:
                sites_to_process.append(SITES[choice])
            else:
                print(f"Warning: '{choice}' is not a valid selection and will be ignored.")

    if not sites_to_process:
        print("No valid sites selected. Exiting.")
        return
    
    # --- Relative Pathing Setup ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(script_dir, "INFOParkAndLoan")
    OUTPUT_FILE = os.path.join(OUTPUT_DIR, "Parkand Loans.csv")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nOutput will be saved to: {OUTPUT_FILE}")
    
    driver = webdriver.Chrome()
    newly_scraped_data = []

    # --- TCE PAL fast path (no link discovery needed) ---
    tce_selected = [s for s in sites_to_process if s.get('scraper_type') == 'tce_pal']
    if tce_selected:
        print('\n--- Running TCE PAL (all pipelines) ---')
        last_known = get_latest_date_for_location(OUTPUT_FILE, 'TCE_PAL')
        tce_rows = tce_fetch_all_pal_rows(last_known)
        if tce_rows:
            newly_scraped_data.extend(tce_rows)
        sites_to_process = [s for s in sites_to_process if s.get('scraper_type') != 'tce_pal']

    try:
        all_new_links = []
        for site in sites_to_process:
            scraper_type = site['scraper_type']
            location_key = site['location_key']
            print(f"\n--- Starting link discovery for {site['name']} ---")

            if scraper_type == "enbridge_pdf":
                last_date = get_latest_date_for_location(OUTPUT_FILE, location_key)
                links = scrape_enbridge_links(driver, site['url'], last_date)
                for link in links:
                    all_new_links.append({'url': link, 'location_key': location_key, 'scraper_type': scraper_type})
            else:
                max_ids = get_max_ids_by_location(OUTPUT_FILE)
                print(f"   - Highest existing ID for {location_key}: {max_ids.get(location_key, 0)}")
                links = scrape_gasnom_links(driver, site['url'])
                max_id_for_loc = max_ids.get(location_key, 0)
                for link in links:
                    try:
                        transaction_id = int(link.split('id=')[-1])
                        if transaction_id > max_id_for_loc:
                            all_new_links.append({'url': link, 'location_key': location_key, 'scraper_type': scraper_type})
                    except (ValueError, IndexError):
                        print(f"Could not parse ID from URL: {link}")

        print(f"\n>>> Found {len(all_new_links)} total new records to process.")

        for i, link_info in enumerate(all_new_links):
            url = link_info['url']
            location_key = link_info['location_key']
            scraper_type = link_info['scraper_type']
            print(f"\nProcessing link {i+1} of {len(all_new_links)} for '{location_key}'...")

            data_list = []
            if scraper_type == "enbridge_pdf":
                data_list = scrape_report_data_enbridge(url, location_key)
            elif scraper_type == 'modern':
                data_list = scrape_report_data_modern(driver, url, location_key)
            else:  # 'original'
                data_item = scrape_report_data_original(driver, url)
                if data_item:
                    data_list.append(data_item)
            if data_list:
                newly_scraped_data.extend(data_list)

            sleep_time = random.uniform(3, 6)
            print(f"--> Pausing for {sleep_time:.2f} seconds...")
            time.sleep(sleep_time)

    except Exception as e:
        print(f"\n--- A CRITICAL ERROR OCCURRED: {e} ---")
    finally:
        if 'driver' in locals() and driver.service.is_connectable():
            print("\nClosing the browser.")
            driver.quit()

    if not newly_scraped_data:
        print("\nNo new data was scraped. The existing CSV file remains unchanged.")
        print_last_update_summary(OUTPUT_FILE)
        return

    # --- Combine, de-dup, sort, write ---
    existing_data = []
    fieldnames = ["Posting Date", "Counter Party", "Type", "Quantity", "End Date", "True Rate", "Location", "Source URL"]
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, 'r', newline='', encoding='utf-8') as f_read:
                reader = csv.DictReader(f_read)
                existing_data.extend(list(reader))
        except Exception as e:
            print(f"Warning: Could not read existing CSV. Error: {e}")
    combined_data = existing_data + newly_scraped_data

    # de-dupe using (Source URL, Counter Party)
    unique_data = {(row['Source URL'], row.get('Counter Party', '')): row for row in combined_data}
    final_data = list(unique_data.values())

    try:
        _df = pd.DataFrame(final_data)
        _df["_pd"] = pd.to_datetime(_df["Posting Date"], errors="coerce")
        _df.sort_values(by=["Counter Party", "_pd"], ascending=[True, False], inplace=True)
        _df.drop(columns=["_pd"], inplace=True)
        final_data = _df.to_dict(orient="records")
    except Exception as e:
        print(f"\nERROR during final sort: {e}. Data will be written unsorted.")

    if final_data:
        print(f"\n--- Writing {len(final_data)} total records to CSV ---")
        try:
            with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(final_data)
            print(f"Successfully saved all data to {OUTPUT_FILE}")
        except Exception as e:
            print(f"Error writing to CSV file: {e}")
    else:
        print("\nNo data to write to the file.")

    print_last_update_summary(OUTPUT_FILE)

if __name__ == "__main__":
    main()
