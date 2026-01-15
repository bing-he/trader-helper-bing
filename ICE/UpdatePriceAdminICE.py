import icepython as ice
import pandas as pd
import re
import time
from pathlib import Path
import logging
import numpy as np

from common.pathing import ROOT
from common.logs import get_file_logger

INFO_DIR = ROOT / "INFO"
ADMIN_FILE = INFO_DIR / "PriceAdmin.csv"
logging.basicConfig(level=logging.INFO)
logger = get_file_logger(Path(__file__).stem)

def discover_and_categorize_symbols():
    """
    A complete, one-step tool that first discovers all daily physical gas
    market codes from the ICE API and then compares them against the
    PriceAdmin.csv file, appending any new items to the 'Uncategorized' section.
    """
    print("--- 🕵️  PriceAdmin Discovery & Auto-Categorizer ---")

    # --- Configuration ---
    # Corrected path to point to the INFO directory
    TARGET_SUFFIX = 'IPG' 
    SEARCH_KEYWORD = 'NEXT DAY GAS'
    logger.info("Discovery start | ADMIN_FILE=%s | TARGET_SUFFIX=%s", ADMIN_FILE, TARGET_SUFFIX)
    
    # --- Step 1: Discover Symbols from API ---
    print("\n[1] Discovering symbols from ICE API...")
    
    # --- Sub-step 1a: Initialize Connection ---
    print(" -> Initializing and stabilizing the API connection...")
    try:
        ice.start_publisher()
        ice.get_quotes(symbols=['IBM'], fields=['last'], subscribe=True)
        print("    -> Connection primed successfully.")
        time.sleep(3)
    except Exception as e:
        print(f"    -> Warning: Connection priming failed, but continuing anyway. Error: {e}")

    # --- Sub-step 1b: Find the Filter ID for the Target Suffix ---
    print(f" -> Finding the Filter ID for Suffix '{TARGET_SUFFIX}'...")
    target_filter_id = None
    try:
        facets = ice.get_search_facets()
        suffix_facet = next((f for f in facets if str(f).lower() == 'suffix'), None)
        if not suffix_facet:
            print("    -> ERROR: Could not find 'Suffix' in the list of available facets.")
            return

        suffix_filters = ice.get_search_filters(suffix_facet)
        suffix_filter_item = next((f for f in suffix_filters if f[1].strip() == TARGET_SUFFIX), None)
        if suffix_filter_item:
            target_filter_id = suffix_filter_item[0]
            print(f"    -> Success! Found Filter ID: {target_filter_id}")
        else:
            print(f"    -> ERROR: Could not find the '{TARGET_SUFFIX}' suffix filter.")
            return
    except Exception as e:
        print(f"    -> ERROR: An error occurred during filter discovery: {e}")
        return

    # --- Sub-step 1c: Perform a Single, Highly Targeted Search ---
    print(f" -> Searching for markets with keyword '{SEARCH_KEYWORD}' and suffix '{TARGET_SUFFIX}'...")
    try:
        search_results = ice.get_search(SEARCH_KEYWORD, rows=9999, filters=[target_filter_id])
        if not search_results or 'Error' in str(search_results):
            print("    -> ERROR: The search failed or returned no results.")
            return
        print(f"    -> Success! Found {len(search_results)} potential markets.")
    except Exception as e:
        print(f"    -> ERROR: An error occurred during the filtered search: {e}")
        return

    # --- Sub-step 1d: Parse and Filter Results ---
    print(" -> Parsing and filtering results...")
    df_results = pd.DataFrame(list(search_results), columns=['Symbol', 'Description', 'Exchange', 'Asset Type'])
    discovered_df = df_results[
        df_results['Symbol'].str.endswith("D1-IPG", na=False) &
        df_results['Description'].str.startswith("NG FIRM PHYS, FP -", na=False) &
        df_results['Description'].str.endswith("- NEXT DAY GAS", na=False)
    ]
    print(f" -> Found {len(discovered_df)} symbols matching the specific description criteria.")

    # --- Step 2: Load PriceAdmin.csv and Compare ---
    print(f"\n[2] Loading '{ADMIN_FILE}' and comparing against discovered symbols...")
    try:
        admin_df = pd.read_csv(ADMIN_FILE, header=None, keep_default_na=False)
    except FileNotFoundError:
        print(f" -> ERROR: Could not find '{ADMIN_FILE}'.")
        return
    
    new_items_to_add = []
    match_count = 0

    for index, row in discovered_df.iterrows():
        symbol = row['Symbol']
        description = row['Description']

        code_match = re.search(r'^([A-Z0-9]{3})', str(symbol))
        if not code_match: continue
        code = code_match.group(1)

        stripped_desc = description.replace("NG FIRM PHYS, FP - ", "").replace(" - NEXT DAY GAS", "")

        # CORRECTED: Convert both sides to uppercase for a case-insensitive match.
        is_match = admin_df[
            (admin_df[2].astype(str).str.upper() == code.upper()) & 
            (admin_df[1].astype(str).str.upper() == stripped_desc.upper())
        ].any().any()

        if is_match:
            match_count += 1
        else:
            new_row = [''] * 16 
            new_row[1] = stripped_desc # Column B
            new_row[2] = code          # Column C
            new_items_to_add.append(new_row)

    print(f" -> Comparison complete. Found {match_count} existing matches.")

    # --- Step 3: Append New Items to PriceAdmin.csv ---
    if not new_items_to_add:
        print("\n[3] No new uncategorized items to add. PriceAdmin.csv is already up to date!")
        return
        
    print(f"\n[3] Found {len(new_items_to_add)} new items to add.")
    
    try:
        uncategorized_header_index = admin_df[admin_df[0] == 'Uncategorized:'].index
        
        # Create a DataFrame for the new items
        new_items_df = pd.DataFrame(new_items_to_add)

        if uncategorized_header_index.empty:
            print(" -> 'Uncategorized:' section not found. Appending new section to the end of the file.")
            
            # Create 5 blank rows
            blank_rows = pd.DataFrame([[''] * 16] * 5) 
            
            # Create the 'Uncategorized:' header row
            uncategorized_header = pd.DataFrame([['Uncategorized:'] + [''] * 15])
            
            # Combine the original data, blank space, header, and new items
            final_admin_df = pd.concat([admin_df, blank_rows, uncategorized_header, new_items_df], ignore_index=True)

        else:
            print(" -> Adding new items under the existing 'Uncategorized:' section.")
            insert_index = uncategorized_header_index[0] + 1
            
            top_part = admin_df.iloc[:insert_index]
            bottom_part = admin_df.iloc[insert_index:]
            
            # Combine the parts: top, new items, bottom
            final_admin_df = pd.concat([top_part, new_items_df, bottom_part], ignore_index=True)
        
        # Save the updated DataFrame back to the CSV file without headers or index
        final_admin_df.to_csv(ADMIN_FILE, index=False, header=False)
        print(f" -> Successfully updated '{ADMIN_FILE}' with the new items.")

    except Exception as e:
        print(f" -> ERROR: An unexpected error occurred while updating the PriceAdmin file: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="resolve key paths then exit")
    args, _ = parser.parse_known_args()
    logger.info(
        "DRY_RUN=%s | FILE=%s | ROOT=%s | INFO_DIR=%s | ADMIN_FILE=%s",
        args.dry_run,
        __file__,
        ROOT,
        INFO_DIR,
        ADMIN_FILE,
    )
    print(f"[dry] ROOT={ROOT}")
    print(f"[dry] INFO_DIR={INFO_DIR}")
    print(f"[dry] ADMIN_FILE={ADMIN_FILE}")
    if not args.dry_run:
        discover_and_categorize_symbols()
