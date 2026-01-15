#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-pipeline PAL scraper for TCE Connects SSRS pages
Pipelines: TCO, Northern Border, Columbia Gulf, Millennium

- Robust SSRS export (avoids 500 by removing conflicting/empty format keys).
- Tries rs:Format=EXCELOPENXML, then Format=EXCELOPENXML, then CSV fallback.
- MAIN TABLE header discovery to find 'Posting Date'/'Posting Date/Time' and 'Rate Schedule'.
- Millennium-specific override: Rate Schedule is column Y (index 24).
- Rate sub-table parsing (header discovery) for End Date & True Rate.
- Location selection per pipeline: Northern Border -> 'Location Name', others -> 'Loc Zn' (with fallback).
- De-dupes on [Posting Date, Counter Party, Location, End Date, Pipeline].

Usage:
  python TCOScraper.py
  python TCOScraper.py --only "Millennium"
  python TCOScraper.py --only "TCO" --local-xlsx "C:\\path\\TransInterrupt.xlsx"
"""

import os
import re
import io
import sys
import html
import csv
import argparse
import requests
import pandas as pd
import numpy as np
import openpyxl
from datetime import datetime, timedelta
from urllib.parse import urljoin, urlparse, parse_qsl, urlencode, urlunparse
from typing import Optional, Dict, List, Tuple, Any, Sequence

# ---------- Configuration ----------
OUTPUT_DIR = r"C:\Users\patri\OneDrive\Desktop\Coding\TraderHelper\ParkAndLoanScrapers\INFOParkAndLoan"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "TCEConnects_PAL.csv")

PIPELINES = [
    {"name": "TCO",             "url": "https://ebb.tceconnects.com/infopost/ReportViewer.aspx?/InfoPost/TransInterrupt&AssetNbr=51"},
    {"name": "Northern Border", "url": "https://ebb.tceconnects.com/infopost/ReportViewer.aspx?/InfoPost/TransInterrupt&AssetNbr=3029"},
    {"name": "Columbia Gulf",   "url": "https://ebb.tceconnects.com/infopost/ReportViewer.aspx?/InfoPost/TransInterrupt&AssetNbr=14"},
    {"name": "Millennium",      "url": "https://ebb.tceconnects.com/infopost/ReportViewer.aspx?/InfoPost/TransInterrupt&AssetNbr=26"},
]

# Preferred location column by pipeline
PIPELINE_LOCATION_PREF = {
    "Northern Border": "location name",
    "TCO":             "loc zn",
    "Columbia Gulf":   "loc zn",
    "Millennium":      "loc zn",
}

CSV_HEADERS_BASE = [
    "Posting Date", "Counter Party", "Type", "Quantity",
    "End Date", "True Rate", "Location", "SourceURL"
]
CSV_HEADERS = CSV_HEADERS_BASE + ["Pipeline"]

# ---------- Logging ----------
def log(msg: str) -> None:
    print(msg, flush=True)

def _now() -> datetime:
    return datetime.now()

# ---------- Date helpers ----------
def safe_to_datetime(x: Any) -> Optional[datetime]:
    if isinstance(x, datetime):
        return x
    if x is None:
        return None
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    try:
        return pd.to_datetime(x, errors="raise")
    except Exception:
        return None

# ---------- Existing CSV cutoff ----------
def get_cutoff_date(file_path: str) -> datetime:
    default_cutoff = _now() - timedelta(days=10)
    if not os.path.exists(file_path):
        log("No existing CSV found. Will process data from the last 10 days.")
        return default_cutoff
    try:
        df = pd.read_csv(file_path, usecols=["Posting Date"])
        if df.empty:
            return default_cutoff
        dt = pd.to_datetime(df["Posting Date"], errors="coerce", format="%d-%b-%y")
        if dt.isna().all():
            dt = pd.to_datetime(df["Posting Date"], errors="coerce")
        latest = dt.max()
        if pd.isna(latest):
            log("Could not parse any dates in existing CSV. Defaulting to last 10 days.")
            return default_cutoff
        log(f"Last known record is from: {latest.strftime('%Y-%m-%d')}")
        return latest
    except Exception as e:
        log(f"Warning: Could not read existing CSV. Error: {e}. Defaulting to last 10 days.")
        return default_cutoff

# ---------- SSRS helpers ----------
def extract_export_url_base(page_html: str, base_url: str) -> str:
    m = re.search(r'"ExportUrlBase"\s*:\s*"([^"]+)"', page_html)
    if not m:
        m = re.search(r"ExportUrlBase\s*[:=]\s*['\"]([^'\"]+)['\"]", page_html)
    if not m:
        raise ValueError("Could not find ExportUrlBase in page source.")

    raw = m.group(1)
    unescaped = html.unescape(raw).replace("\\u0026", "&").replace("\\/", "/")
    abs_url = urljoin(base_url, unescaped)

    parsed = urlparse(abs_url)
    q = dict(parse_qsl(parsed.query, keep_blank_values=True))

    # Remove conflicting/empty format keys; we’ll add our own
    q.pop("Format", None)
    q.pop("rs:Format", None)
    q["ContentDisposition"] = "AlwaysAttachment"

    new_query = urlencode(q, doseq=True)
    return urlunparse(parsed._replace(query=new_query))

def download_export_bytes(viewer_url: str) -> bytes:
    base_origin = "{uri.scheme}://{uri.netloc}".format(uri=urlparse(viewer_url))
    with requests.Session() as s:
        s.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "*/*"})
        log(f"Fetching page content and session cookies from: {viewer_url}")
        r0 = s.get(viewer_url, timeout=60)
        r0.raise_for_status()
        page = r0.text
        log("Session established.")

        base_url = extract_export_url_base(page, base_origin)
        parsed = urlparse(base_url)
        base_q = dict(parse_qsl(parsed.query, keep_blank_values=True))
        base_q["ContentDisposition"] = "AlwaysAttachment"

        candidates = [
            {"rs:Format": "EXCELOPENXML"},
            {"Format": "EXCELOPENXML"},
            {"rs:Format": "CSV"},
        ]

        last_error = None
        for fmt in candidates:
            q = {k: v for k, v in base_q.items() if k not in ("Format", "rs:Format")}
            q.update(fmt)
            url = urlunparse(parsed._replace(query=urlencode(q, doseq=True)))
            log(f"Attempting export with params {fmt} …")
            r = s.get(url, headers={"Referer": viewer_url}, timeout=120)
            try:
                r.raise_for_status()
                ctype = r.headers.get("Content-Type", "")
                is_xlsx = r.content.startswith(b"PK")
                is_csv_like = ("csv" in ctype.lower()) or (not is_xlsx and r.text[:1] in ('"', ',', 'P', 'D'))
                if is_xlsx or is_csv_like:
                    log("Download complete.")
                    return r.content
                else:
                    snippet = r.text[:200] if "text" in ctype.lower() else str(r.content[:100])
                    raise RuntimeError(f"Unexpected content-type '{ctype}'. Snippet: {snippet!r}")
            except Exception as e:
                last_error = e
                log(f"  -> Export attempt with {fmt} failed: {e}")

        raise RuntimeError(f"All SSRS export attempts failed. Last error: {last_error}")

# ---------- row loaders ----------
def rows_from_xlsx_bytes(xbytes: bytes) -> List[Tuple[Any, ...]]:
    wb = openpyxl.load_workbook(io.BytesIO(xbytes), data_only=True, read_only=True)
    ws = wb.active
    return [tuple(r) for r in ws.iter_rows(values_only=True)]

def rows_from_csv_bytes(cbytes: bytes) -> List[Tuple[Any, ...]]:
    try:
        text = cbytes.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        text = cbytes.decode("latin-1", errors="replace")
    rdr = csv.reader(io.StringIO(text))
    return [tuple(row) for row in rdr]

def load_rows_from_bytes(raw: bytes) -> List[Tuple[Any, ...]]:
    if raw.startswith(b"PK"):
        return rows_from_xlsx_bytes(raw)
    return rows_from_csv_bytes(raw)

# ---------- header discovery for MAIN posting table ----------
def find_main_header_and_map(rows: List[Tuple[Any, ...]]) -> Tuple[Optional[int], Dict[str, int]]:
    """
    Find the main posting table header row with 'Posting Date'/'Posting Date/Time' and 'Rate Schedule'.
    Returns (header_index, {lower_header_name: col_index}).
    """
    header_idx = None
    colmap: Dict[str, int] = {}
    for i in range(0, min(len(rows), 120)):
        r = rows[i]
        if not r:
            continue
        lower = [str(c).strip().lower() if c is not None else "" for c in r]
        if any("posting date" in cell for cell in lower) and any("rate schedule" in cell for cell in lower):
            header_idx = i
            for ci, name in enumerate(r):
                if name is not None:
                    nm = str(name).strip().lower()
                    if nm:
                        colmap[nm] = ci
            break
    return header_idx, colmap

# ---------- parsing helpers for Rate/Location blocks ----------
def find_rate_header_index(rows: List[Tuple], start_idx: int, search_window: int = 12) -> Optional[int]:
    alt_targets = {
        "beg": {"disc beg date", "begin date", "start date", "beg date", "discount begin date"},
        "end": {"disc end date", "end date", "discount end date", "disc end"},
        "rate": {"rate chgd", "rate charged", "rate", "daily rate"},
    }
    for j in range(start_idx, min(len(rows), start_idx + search_window)):
        r = rows[j]
        if not r:
            continue
        lower = [str(c).strip().lower() if c is not None else "" for c in r]
        ok_beg  = any(any(k in cell for k in alt_targets["beg"])  for cell in lower)
        ok_end  = any(any(k in cell for k in alt_targets["end"])  for cell in lower)
        ok_rate = any(any(k in cell for k in alt_targets["rate"]) for cell in lower)
        if ok_beg and ok_end and ok_rate:
            return j
    return None

def map_rate_header_indices(header_row: Sequence[Any]) -> Dict[str, int]:
    idx = {"beg": 0, "end": 3, "rate": 8}  # defaults
    lower = [str(c).strip().lower() if c is not None else "" for c in header_row]
    for ci, txt in enumerate(lower):
        if "disc beg date" in txt or "begin date" in txt or "start date" in txt or "beg date" in txt:
            idx["beg"] = ci
        if "disc end date" in txt or "end date" in txt:
            idx["end"] = ci
        if "rate chgd" in txt or "rate charged" in txt or (txt == "rate") or "daily rate" in txt:
            idx["rate"] = ci
    return idx

def walk_rate_block(rows: List[Tuple], header_idx: int) -> Tuple[int, int, List[Tuple], Dict[str, int]]:
    data_start = header_idx + 1
    col_map = map_rate_header_indices(rows[header_idx])
    k = data_start
    while k < len(rows):
        r = rows[k]
        if not r or (len(r) > 0 and (r[0] is None or str(r[0]).strip() == "")):
            break
        k += 1
    return data_start, k, rows[data_start:k], col_map

def find_location_header_index(rows: List[Tuple], after_idx: int, search_window: int = 8) -> Optional[int]:
    j = after_idx
    if j < len(rows):
        r = rows[j]
        if (not r) or (len(r) > 0 and (r[0] is None or str(r[0]).strip() == "")):
            j += 1
    for k in range(j, min(len(rows), j + search_window)):
        r = rows[k]
        if not r:
            continue
        lower = [str(c).strip().lower() if c is not None else "" for c in r]
        if any(("loc zn" in cell) or ("location name" in cell) for cell in lower):
            return k
    return None

def locate_location_columns(header_row: Sequence[Any]) -> Dict[str, int]:
    res: Dict[str, int] = {}
    lower = [str(c).strip().lower() if c is not None else "" for c in header_row]
    for ci, txt in enumerate(lower):
        if "loc zn" in txt:
            res["loc zn"] = ci
        if "location name" in txt:
            res["location name"] = ci
    return res

def pick_location_value(loc_row: Sequence[Any], cols: Dict[str, int], pipeline_name: str) -> str:
    pref_key = PIPELINE_LOCATION_PREF.get(pipeline_name, "loc zn")
    alt_key = "location name" if pref_key == "loc zn" else "loc zn"

    if pref_key in cols and cols[pref_key] < len(loc_row):
        val = loc_row[cols[pref_key]]
        if val is not None and str(val).strip():
            return str(val).strip()

    if alt_key in cols and cols[alt_key] < len(loc_row):
        val = loc_row[cols[alt_key]]
        if val is not None and str(val).strip():
            return str(val).strip()

    return "N/A"

# ---------- MAIN parse ----------
def parse_transinterrupt_rows(rows: List[Tuple[Any, ...]], cutoff_date: Optional[datetime],
                              source_url: str, pipeline_name: str) -> pd.DataFrame:
    log(f"[{pipeline_name}] Loaded {len(rows)} rows from export.")
    out: List[Dict[str, str]] = []

    # (A) Discover the main header row and map columns
    hdr_idx, cmap = find_main_header_and_map(rows)
    if hdr_idx is None:
        start_main = 12
        posting_col = 0
        rate_sched_col = 25
        counterparty_col = 9
    else:
        start_main = hdr_idx + 1
        posting_col = cmap.get("posting date/time", cmap.get("posting date", 0))
        rate_sched_col = cmap.get("rate schedule", 25)
        counterparty_col = cmap.get("contract holder name", cmap.get("contract holder", 9))

    # (B) Millennium override: Rate Schedule is Column Y (index 24)
    if pipeline_name == "Millennium":
        rate_sched_col = 24

    # (C) Iterate main rows to find PAL
    for i in range(start_main, len(rows)):
        row = rows[i]
        if not row:
            continue

        a = row[posting_col] if posting_col < len(row) else None
        sched = row[rate_sched_col] if rate_sched_col < len(row) else None
        posting_dt = safe_to_datetime(a)
        if posting_dt is None:
            continue
        if str(sched).strip().upper() != "PAL":
            continue
        if cutoff_date and posting_dt <= cutoff_date:
            continue

        counterparty = (str(row[counterparty_col]).strip()
                        if counterparty_col < len(row) and row[counterparty_col] is not None else "N/A")

        # (D) Rate block: search for header near the trigger row
        header_guess = i + 3
        rate_hdr_idx = find_rate_header_index(rows, header_guess, search_window=12)
        if rate_hdr_idx is None:
            rate_hdr_idx = find_rate_header_index(rows, i + 1, search_window=16)
        if rate_hdr_idx is None:
            continue

        data_start, data_end, rate_rows, cmap_rate = walk_rate_block(rows, rate_hdr_idx)
        if not rate_rows:
            continue

        # End Date from last rate row
        end_cell = rate_rows[-1][cmap_rate["end"]] if len(rate_rows[-1]) > cmap_rate["end"] else None
        end_dt = safe_to_datetime(end_cell)
        if end_dt is None:
            end_cell_b = rate_rows[-1][1] if len(rate_rows[-1]) > 1 else None
            end_dt = safe_to_datetime(end_cell_b)
        if end_dt is None:
            continue
        end_date_str = end_dt.strftime("%d-%b-%y")

        # True Rate from penultimate row
        true_rate = 0.0
        if len(rate_rows) >= 2:
            tr = rate_rows[-2]
            beg = safe_to_datetime(tr[cmap_rate["beg"]] if len(tr) > cmap_rate["beg"] else None)
            end_ = safe_to_datetime(tr[cmap_rate["end"]] if len(tr) > cmap_rate["end"] else None)
            rate_chgd = tr[cmap_rate["rate"]] if len(tr) > cmap_rate["rate"] else None
            rate_chgd_f = pd.to_numeric(rate_chgd, errors="coerce")
            if beg is not None and end_ is not None and pd.notna(rate_chgd_f):
                days = (end_.date() - beg.date()).days + 1
                if days >= 0:
                    true_rate = float(days) * float(rate_chgd_f)

        # (E) Location block
        loc_hdr_idx = find_location_header_index(rows, data_end)
        location = "N/A"
        if loc_hdr_idx is not None and (loc_hdr_idx + 1) < len(rows):
            cols = locate_location_columns(rows[loc_hdr_idx])
            loc_row = rows[loc_hdr_idx + 1]
            if loc_row:
                location = pick_location_value(loc_row, cols, pipeline_name)

        out.append({
            "Posting Date": posting_dt.strftime("%d-%b-%y"),
            "Counter Party": counterparty,
            "Type": "PAL",
            "Quantity": "N/A",
            "End Date": end_date_str,
            "True Rate": f"{true_rate:.4f}",
            "Location": location,
            "SourceURL": source_url,
            "Pipeline": pipeline_name,
        })

    return pd.DataFrame(out)

# ---------- write ----------
def append_and_write(df_new: pd.DataFrame, out_csv: str) -> int:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    if os.path.exists(out_csv):
        try:
            df_old = pd.read_csv(out_csv)
        except Exception:
            log("Existing CSV unreadable; starting fresh.")
            df_old = pd.DataFrame(columns=CSV_HEADERS)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.drop_duplicates(
        subset=["Posting Date", "Counter Party", "Location", "End Date", "Pipeline"],
        keep="last",
        inplace=True,
    )
    # Sort desc by Posting Date; then by Pipeline
    dt_series = pd.to_datetime(df_all["Posting Date"], errors="coerce", format="%d-%b-%y")
    df_all["_pd"] = dt_series
    df_all.sort_values(by=["_pd", "Pipeline"], ascending=[False, True], inplace=True)
    df_all.drop(columns=["_pd"], inplace=True)

    # Ensure column order
    for col in CSV_HEADERS:
        if col not in df_all.columns:
            df_all[col] = ""
    df_all = df_all[CSV_HEADERS]

    df_all.to_csv(out_csv, index=False)
    return len(df_all)

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Scrape & parse multiple PAL pipelines into one CSV.")
    ap.add_argument("--local-xlsx", help="Parse a local Excel file instead of downloading (applies only to --only pipeline).", default=None)
    ap.add_argument("--only", help="Limit to a single pipeline name (e.g., 'TCO', 'Millennium').", default=None)
    args = ap.parse_args()

    cutoff = get_cutoff_date(OUTPUT_FILE)
    log(f"--- Starting: Looking for records newer than {cutoff.strftime('%Y-%m-%d')} ---")

    targets = PIPELINES
    if args.only:
        name = args.only.strip().lower()
        targets = [p for p in PIPELINES if p["name"].lower() == name]
        if not targets:
            log(f"No pipeline found matching --only {args.only}. Available: {[p['name'] for p in PIPELINES]}")
            sys.exit(1)

    dfs = []
    for p in targets:
        try:
            if args.local_xlsx and len(targets) == 1:
                log(f"[{p['name']}] Parsing local Excel: {args.local_xlsx}")
                with open(args.local_xlsx, "rb") as f:
                    raw = f.read()
            else:
                raw = download_export_bytes(p["url"])

            rows = load_rows_from_bytes(raw)
            dfp = parse_transinterrupt_rows(rows, cutoff_date=cutoff, source_url=p["url"], pipeline_name=p["name"])
            if dfp.empty:
                log(f"[{p['name']}] No new PAL records.")
            else:
                log(f"[{p['name']}] Parsed {len(dfp)} new PAL records.")
                dfs.append(dfp)
        except Exception as e:
            log(f"[{p['name']}] ERROR: {e}")

    if not dfs:
        log("No new PAL records across selected pipelines.")
        return

    df_new = pd.concat(dfs, ignore_index=True)
    n_total = append_and_write(df_new, OUTPUT_FILE)
    log(f"Wrote {len(df_new)} new records across {len(dfs)} pipelines. Total rows now: {n_total}")
    log(f"CSV: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
