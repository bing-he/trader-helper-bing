import pandas as pd
import numpy as np
import time
from datetime import datetime
from UpdateHistoricalPrices import ConfigManager, DataFetcher, CONFIG_FILE, ice, ICE_SYMBOL_SUFFIX

def smart_fetch_ice(symbol_map, start_date, label):
    """
    Robust fetcher that:
    1. Normalizes column names (strips whitespace).
    2. Prioritizes 'VWAP Close', then 'Settle', etc.
    3. IMPORTANT: Skips columns that are entirely 0.0 (fixes the GDA issue).
    4. Fallback: Uses the first numeric column with data if named fields fail.
    """
    if ice is None:
        print("ICEPython not found.")
        return pd.DataFrame()

    print(f"Fetching {label} from {start_date.date()}...")
    frames = []
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = datetime.now().strftime('%Y-%m-%d')
    
    # Priority: VWAP is usually best for Fixed. Settle is usually best for GDA (Basis).
    fetch_fields = ['VWAP Close', 'Settle', 'Wtd Avg Price', 'Close']
    
    for market, code in symbol_map.items():
        full_symbol = f"{code}{ICE_SYMBOL_SUFFIX}"
        try:
            time.sleep(0.2)
            # Request all relevant fields
            data = ice.get_timeseries(
                symbols=[full_symbol], 
                start_date=start_str, 
                end_date=end_str, 
                fields=fetch_fields, 
                granularity='D'
            )
            
            if not data or len(data) < 2 or 'Error' in str(data[0]):
                # print(f"  No data for {market} ({full_symbol})")
                continue

            # Convert to DataFrame
            df = pd.DataFrame(list(data[1:]), columns=list(data[0]))
            
            # CLEANUP: Fix column names (strip whitespace) and Date
            df.columns = [str(c).strip() for c in df.columns] 
            
            if 'Time' in df.columns: df.rename(columns={'Time': 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # --- SMART SELECTION LOGIC ---
            chosen_col = None
            used_field = "None"
            
            # Strategy 1: Look for specific fields with NON-ZERO data
            for field in fetch_fields:
                if field in df.columns:
                    series = pd.to_numeric(df[field], errors='coerce')
                    # Check: Not empty AND has at least one non-zero value
                    if not series.dropna().empty and (series != 0).any():
                        chosen_col = series
                        used_field = field
                        break
            
            # Strategy 2: Fallback to ANY numeric column with non-zero data
            # (Fixes cases where headers might change or contain typos)
            if chosen_col is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    series = pd.to_numeric(df[col], errors='coerce')
                    if not series.dropna().empty and (series != 0).any():
                        chosen_col = series
                        used_field = f"Fallback: {col}"
                        break

            if chosen_col is not None:
                chosen_col.name = market
                frames.append(chosen_col)
                # print(f"  {market}: Found data in '{used_field}'")
            else:
                print(f"  {market}: fetched columns {df.columns.tolist()} but all were 0.0 or NaN.")

        except Exception as e:
            print(f"  Error fetching {market}: {e}")
            pass

    print(f"Fetching {label}: Complete.")
    if frames:
        return pd.concat(frames, axis=1).sort_index()
    return pd.DataFrame()

def main():
    print("--- Custom Data Pull: Bennington Fix ---")
    
    # 1. Load Config
    config = ConfigManager(CONFIG_FILE)
    config.load()
    
    # 2. Define Targets
    target_hubs = [
        'Henry', 
        'CG-Mainline', 
        'TCO', 
        'Bennington', 
        'NGPL-TXOK East', 
        'TETCO-WLA'
    ]
    
    # 3. Filter Mappings
    platts_subset = {k: v for k, v in config.platts_map.items() if k in target_hubs}
    ice_fixed_subset = {k: v for k, v in config.ice_fixed_map.items() if k in target_hubs}
    ice_gda_subset = {k: v for k, v in config.ice_gda_map.items() if k in target_hubs}

    start_date = datetime(2025, 11, 20)

    # 4. Fetch and Display
    
    # Platts
    print("\n" + "="*40)
    print("   PLATTS SETTLES (Flow Date)")
    print("="*40)
    if platts_subset:
        p_df = DataFetcher.fetch_platts(platts_subset, start_date)
        print(p_df.round(4) if not p_df.empty else "No Platts data.")

    # ICE Fixed
    print("\n" + "="*40)
    print("   ICE FIXED (Trade Date)")
    print("="*40)
    if ice_fixed_subset:
        i_df = smart_fetch_ice(ice_fixed_subset, start_date, "ICE Fixed")
        print(i_df.round(4) if not i_df.empty else "No ICE Fixed data.")

    # ICE GDA
    print("\n" + "="*40)
    print("   ICE GDA (Trade Date)")
    print("="*40)
    if ice_gda_subset:
        g_df = smart_fetch_ice(ice_gda_subset, start_date, "ICE GDA")
        print(g_df.round(4) if not g_df.empty else "No ICE GDA data.")

if __name__ == "__main__":
    main()