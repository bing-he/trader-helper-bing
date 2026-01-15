"""
UpdateHistoricalPrices.py

A unified program to fetch, align, and calculate natural gas prices.
Strict adherence to User Constraints:
1. No fuzzy matching on market component names.
2. Explicit Date Mapping: Flow Dates <-mapped via Price-> Trade Dates.
3. Explicit Roll Forward logic using the Date Map.

Path: C:/Users/patri/OneDrive/Desktop/Coding/TraderHelper/ICE/UpdateHistoricalPrices.py
"""

import os
import sys
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import numpy as np
import requests
from dotenv import load_dotenv

# Try importing icepython
try:
    import icepython as ice
except ImportError:
    ice = None

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# Directories
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DIR = PROJECT_ROOT / "INFO"
OUTPUT_DIR = SCRIPT_DIR / "Outputs"

# Files
CONFIG_FILE = INPUT_DIR / "PriceAdmin.csv"
OUTPUT_FILE = OUTPUT_DIR / "PRICES.csv"
ENV_FILE = PROJECT_ROOT / "Platts" / ".env"

# Parameters
DAYS_TO_REFRESH = 15
PLATTS_API_URL = "https://api.platts.com/market-data/v3/value/current/symbol"

# Alignment Settings
ALIGNMENT_TOLERANCE = 0.004 
ALIGNMENT_WINDOW_DAYS = 10 
ICE_SYMBOL_SUFFIX = ' D1-IPG' 

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Load Environment Variables
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    logger.warning(f".env file NOT found at expected path: {ENV_FILE}")


class ConfigManager:
    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.platts_map: Dict[str, str] = {}
        self.ice_fixed_map: Dict[str, str] = {}
        self.ice_gda_map: Dict[str, str] = {}
        self.ref_map: Dict[str, str] = {}

    def load(self) -> None:
        if not self.filepath.exists():
            logger.error(f"Configuration file not found: {self.filepath}")
            sys.exit(1)

        try:
            df = pd.read_csv(self.filepath)
            # Basic cleanup: Strip whitespace from headers and content
            df.columns = df.columns.str.strip()
            df['Market Component'] = df['Market Component'].str.strip()

            # 1. Platts Map
            if 'PlattsCodePlatts' in df.columns:
                self.platts_map = df.dropna(subset=['PlattsCodePlatts']).set_index('Market Component')['PlattsCodePlatts'].to_dict()

            # 2. ICE Fixed Map
            if 'Daily Code' in df.columns:
                self.ice_fixed_map = df.dropna(subset=['Daily Code']).set_index('Market Component')['Daily Code'].to_dict()

            # 3. ICE GDA Map (Basis Adders)
            gda_col = 'GasDailyCode'
            if 'GasDailyCode' in df.columns: gda_col = 'GasDailyCode'
            elif 'Gas Daily Code' in df.columns: gda_col = 'Gas Daily Code'
            
            if gda_col in df.columns:
                self.ice_gda_map = df.dropna(subset=[gda_col]).set_index('Market Component')[gda_col].to_dict()

            # 4. Reference Map (Strict string match required by user)
            ref_col = 'GasDaily'
            if 'GasDaily' in df.columns: ref_col = 'GasDaily'
            elif 'Gas Daily' in df.columns: ref_col = 'Gas Daily'

            if ref_col in df.columns:
                self.ref_map = df.dropna(subset=[ref_col]).set_index('Market Component')[ref_col].to_dict()
            
            logger.info(f"Config Loaded: {len(self.platts_map)} Platts, {len(self.ice_fixed_map)} Fixed, {len(self.ice_gda_map)} GDA.")

        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)


class DataFetcher:
    @staticmethod
    def get_platts_token() -> Optional[str]:
        api_key = os.getenv("PLATTS_API_KEY") or os.getenv("PLATTS_CLIENT_ID")
        username = os.getenv("PLATTS_USERNAME")
        password = os.getenv("PLATTS_PASSWORD")

        if not username or not password:
            logger.warning("Platts credentials missing.")
            return None

        auth_url = "https://api.ci.spglobal.com/auth/api"
        payload = {"username": username, "password": password}
        
        if api_key:
             auth_url = "https://auth.platts.com/oauth/token"
             payload = {
                "grant_type": "password", "username": username, "password": password,
                "client_id": api_key, "scope": "implicit"
            }
        
        try:
            if "auth.platts.com" in auth_url:
                 response = requests.post(auth_url, data=payload, timeout=10)
            else:
                 headers = {"Content-Type": "application/x-www-form-urlencoded"}
                 response = requests.post(auth_url, headers=headers, data=payload, timeout=10)
            response.raise_for_status()
            return response.json().get("access_token")
        except Exception as e:
            logger.error(f"Platts Auth Failed: {e}")
            return None

    @staticmethod
    def fetch_platts(symbol_map: Dict[str, str], start_date: datetime) -> pd.DataFrame:
        token = DataFetcher.get_platts_token()
        if not token or not symbol_map: return pd.DataFrame()

        logger.info(f"Fetching Platts data from {start_date.date()}...")
        history_url = "https://api.ci.spglobal.com/market-data/v3/value/history/symbol"
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        all_records = []

        for market_name, symbol_code in symbol_map.items():
            filter_str = f'symbol IN ("{symbol_code}") AND bate:"U" AND assessDate>="{start_date.strftime("%Y-%m-%d")}"'
            page = 1
            while True:
                try:
                    resp = requests.get(history_url, headers=headers, params={"PageSize": 1000, "Page": page, "Filter": filter_str}, timeout=10)
                    if resp.status_code != 200: break
                    data = resp.json().get("results", [])
                    if not data or not data[0].get('data'): break
                    
                    for item in data[0]['data']:
                        if item.get("value") is not None:
                            all_records.append({'Date': item.get("assessDate"), 'Market Component': market_name, 'Price': item.get("value")})
                    
                    if len(data[0]['data']) < 1000: break
                    page += 1
                    time.sleep(0.01)
                except: break

        if not all_records: return pd.DataFrame()
        
        df = pd.DataFrame(all_records)
        df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
        return df.pivot(index='Date', columns='Market Component', values='Price').apply(pd.to_numeric, errors='coerce')

    @staticmethod
    def fetch_ice(symbol_map: Dict[str, str], start_date: datetime, label: str) -> pd.DataFrame:
        if ice is None: return pd.DataFrame()
        
        logger.info(f"Fetching {label} from {start_date.date()}...")
        frames = []
        start_str, end_str = start_date.strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d')
        
        # SMART FIELD SELECTION: Order of priority
        fetch_fields = ['VWAP Close', 'Settle', 'Wtd Avg Price', 'Close']
        
        total = len(symbol_map)
        
        for i, (market, code) in enumerate(symbol_map.items()):
            full_symbol = f"{code}{ICE_SYMBOL_SUFFIX}"
            try:
                time.sleep(0.2)
                data = ice.get_timeseries(
                    symbols=[full_symbol], 
                    start_date=start_str, 
                    end_date=end_str, 
                    fields=fetch_fields, 
                    granularity='D'
                )
                
                if not data or len(data) < 2 or 'Error' in str(data[0]): continue

                df_pd = pd.DataFrame(list(data[1:]), columns=list(data[0]))
                
                # Cleanup Headers (Strip whitespace)
                df_pd.columns = [str(c).strip() for c in df_pd.columns]
                
                if 'Time' in df_pd.columns: df_pd.rename(columns={'Time': 'Date'}, inplace=True)
                df_pd['Date'] = pd.to_datetime(df_pd['Date'])
                df_pd.set_index('Date', inplace=True)
                
                # --- SMART SELECTION LOGIC ---
                # Loop fields, pick first one that has data (and isn't just 0.0)
                chosen_series = None
                
                # Check named fields first
                for field in fetch_fields:
                    if field in df_pd.columns:
                        vals = pd.to_numeric(df_pd[field], errors='coerce')
                        if not vals.dropna().empty and (vals != 0).any():
                            chosen_series = vals
                            break
                
                # Check any numeric field if named ones failed
                if chosen_series is None:
                    numeric_cols = df_pd.select_dtypes(include=[np.number]).columns
                    for col in numeric_cols:
                        vals = pd.to_numeric(df_pd[col], errors='coerce')
                        if not vals.dropna().empty and (vals != 0).any():
                            chosen_series = vals
                            break

                if chosen_series is not None:
                    chosen_series.name = market
                    frames.append(chosen_series)
                
            except Exception: pass
            if i % 10 == 0: print(f"Fetching {label}: {i}/{total}...", end='\r')
            
        print(f"Fetching {label}: Complete.          ")
        return pd.concat(frames, axis=1) if frames else pd.DataFrame()


class PriceProcessor:
    @staticmethod
    def process(platts_df: pd.DataFrame, ice_fixed_df: pd.DataFrame, 
                ice_gda_df: pd.DataFrame, ref_map: Dict[str, str]) -> pd.DataFrame:
        
        if platts_df.empty and ice_fixed_df.empty: return pd.DataFrame()

        # --- STEP 1: INITIAL MERGE ---
        # We start with Platts (Flow Date). 
        # We assume standard T+1 for initial visual, but we will calculate strictly below.
        ice_shifted = ice_fixed_df.copy()
        ice_shifted.index = ice_shifted.index + timedelta(days=1)
        merged_df = platts_df.combine_first(ice_shifted)
        
        # --- STEP 2: CREATE DATE ALIGNMENT MAP ---
        # Goal: Create a Series 'date_map' where index=FlowDate and value=TradeDate
        
        logger.info("Step 2: Aligning ICE Trade Dates to Platts Flow Dates...")
        anchors = [c for c in ['Henry', 'CG-Mainline', 'TCO', 'Chicago'] if c in platts_df.columns and c in ice_fixed_df.columns]
        
        if not anchors:
            logger.warning("CRITICAL: No anchors (Henry, TCO) found. Cannot align dates.")
            return merged_df

        logger.info(f"Using Anchors: {anchors}")
        
        # Initialize map series
        date_map = pd.Series(index=merged_df.index, dtype='datetime64[ns]')
        date_map[:] = pd.NaT

        platts_dict = platts_df[anchors].to_dict('index')
        ice_dict = ice_fixed_df[anchors].to_dict('index')
        
        matches_found = 0

        # Loop Flow Dates (Platts)
        for flow_date in platts_dict.keys():
            flow_prices = platts_dict[flow_date]
            
            # Find matching Trade Date (ICE)
            best_trade_date = None
            min_avg_diff = float('inf')
            
            # Look back ALIGNMENT_WINDOW_DAYS
            potential_trades = [d for d in ice_dict.keys() if 0 <= (flow_date - d).days <= ALIGNMENT_WINDOW_DAYS]
            
            for trade_date in potential_trades:
                trade_prices = ice_dict[trade_date]
                
                diffs = []
                for a in anchors:
                    if pd.notna(flow_prices.get(a)) and pd.notna(trade_prices.get(a)):
                        diffs.append(abs(flow_prices[a] - trade_prices[a]))
                
                if diffs:
                    avg_diff = sum(diffs) / len(diffs)
                    if avg_diff < min_avg_diff:
                        min_avg_diff = avg_diff
                        best_trade_date = trade_date
            
            # If match is within tolerance, record it in the map
            if best_trade_date and min_avg_diff <= ALIGNMENT_TOLERANCE:
                date_map[flow_date] = best_trade_date
                matches_found += 1
                
                # Diagnostic log for large gaps (strips)
                lag = (flow_date - best_trade_date).days
                if lag > 4:
                    logger.info(f"  Map: Flow {flow_date.date()} <- Trade {best_trade_date.date()} (Lag: {lag}d, Err: {min_avg_diff:.4f})")

        logger.info(f"Aligned {matches_found} specific dates using price matching.")

        # --- STEP 3: ROLL FORWARD (The "Hold on" Logic) ---
        # If a Flow Date (e.g., Saturday) has no direct match, it uses the last valid Trade Date.
        # We perform forward fill on the map.
        date_map = date_map.sort_index().ffill()
        
        # --- STEP 4: APPLY GDA PRICES USING THE MAP ---
        logger.info("Step 4: Applying ICE GDA prices using the Date Map...")
        
        # Create a new DataFrame for GDA data aligned to Flow Date
        # reindex(date_map) effectively pulls the row from ice_gda_df corresponding to the Trade Date
        # and places it at the Flow Date index.
        valid_map = date_map.dropna()
        
        if not valid_map.empty:
            # Reindex ICE GDA to match the mapped trade dates
            # .loc[valid_map] pulls the rows. .set_index(valid_map.index) aligns them to Flow Dates.
            aligned_gda = ice_gda_df.loc[valid_map].set_index(valid_map.index)
            
            # Reindex to full merged_df to handle any missing rows (though ffill handles most)
            aligned_gda = aligned_gda.reindex(merged_df.index)
        else:
            logger.warning("Date Map is empty. Cannot align GDA prices.")
            aligned_gda = pd.DataFrame(index=merged_df.index)

        # --- STEP 5: CALCULATION (Exact Name Matching Only) ---
        logger.info("Step 5: Calculating derived prices (Ref + GDA)...")
        calc_count = 0
        
        for target, ref in ref_map.items():
            # STRICT NAME MATCHING: No fuzzy logic.
            if ref not in merged_df.columns: 
                # logger.debug(f"Skipping {target}: Ref '{ref}' missing.")
                continue
                
            if target not in merged_df.columns: merged_df[target] = np.nan
            if target not in aligned_gda.columns: 
                # logger.debug(f"Skipping {target}: Basis '{target}' missing in GDA data.")
                continue
            
            # Logic: Price = Reference (Flow Date) + Basis (Mapped Trade Date)
            calculated = merged_df[ref] + aligned_gda[target]
            
            # Update only if missing
            mask = merged_df[target].isna() & calculated.notna()
            if mask.any():
                merged_df.loc[mask, target] = calculated[mask]
                calc_count += mask.sum()
                
        logger.info(f"  -> Filled {calc_count} missing points.")
        return merged_df


def main():
    logger.info("--- Starting Price Update ---")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = ConfigManager(CONFIG_FILE)
    config.load()
    
    # Defaults
    platts_start = datetime(2000, 1, 1)
    ice_start = datetime(2014, 1, 1)
    history_df = pd.DataFrame()
    
    # Load History
    if OUTPUT_FILE.exists():
        try:
            history_df = pd.read_csv(OUTPUT_FILE, index_col=0, parse_dates=True)
            if not history_df.empty:
                cutoff = history_df.index.max() - timedelta(days=DAYS_TO_REFRESH)
                history_df = history_df[history_df.index < cutoff]
                platts_start = cutoff
                ice_start = cutoff - timedelta(days=ALIGNMENT_WINDOW_DAYS + 2) 
                logger.info(f"Update Mode: Fetching from {cutoff.date()}")
        except Exception as e:
            logger.warning(f"Error reading history: {e}")

    # Fetch
    platts_df = DataFetcher.fetch_platts(config.platts_map, platts_start)
    ice_fixed_df = DataFetcher.fetch_ice(config.ice_fixed_map, ice_start, "ICE Fixed")
    ice_gda_df = DataFetcher.fetch_ice(config.ice_gda_map, ice_start, "ICE GDA")

    # Process
    new_data = PriceProcessor.process(platts_df, ice_fixed_df, ice_gda_df, config.ref_map)
    
    if new_data.empty and history_df.empty: return

    # Merge & Save
    logger.info("Saving to disk...")
    final_df = pd.concat([history_df, new_data])
    final_df = final_df[~final_df.index.duplicated(keep='last')]
    final_df.sort_index(ascending=False, inplace=True)
    
    final_df.columns = final_df.columns.astype(str)
    final_df = final_df.reindex(sorted(final_df.columns), axis=1)
    final_df = final_df.round(4)
    final_df.dropna(how='all', inplace=True)

    try:
        final_df.to_csv(OUTPUT_FILE)
        logger.info(f"Success: {OUTPUT_FILE}")
    except PermissionError:
        logger.error(f"Permission denied: {OUTPUT_FILE} is open.")

if __name__ == "__main__":
    main()