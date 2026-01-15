# --- AUTO-PATCHED: Enforce venv + preflight ---
import sys, subprocess, time, pathlib

# Make sure project root and Scripts/ are importable before pulling common.pathing
_ROOT = pathlib.Path(__file__).resolve().parent.parent
_SCRIPTS = _ROOT / "Scripts"
for _p in (_ROOT, _SCRIPTS):
    _p_str = str(_p)
    if _p_str not in sys.path:
        sys.path.insert(0, _p_str)

from common.pathing import ROOT as _ROOT  # now import resolves via updated sys.path

try:
    import tools.ensure_venv as _ensure_venv  # exits if wrong interpreter / missing deps
except Exception as _e:
    print("Preflight load failed:", _e)
    raise SystemExit(1)
PY = sys.executable

"""
Orchestrates the execution of all Criterion data update scripts in sequence.

This script runs a predefined list of Python scripts, ensuring they use the
same Python environment. It captures the output of each script and will halt
the entire process if any single script fails.
"""

import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# ==============================================================================
#  CONFIGURATION & CONSTANTS
# ==============================================================================

# --- Path Configuration ---
# All scripts are expected to be in the same directory as this orchestrator.
ORCHESTRATOR_DIR = _ROOT / "Criterion"

# --- Script Execution Order ---
# Define the sequence in which the scripts should be run.
SCRIPTS_TO_RUN = [
    "UpdateAndForecastFundy.py",
    "UpdateCriterionHenryFlows.py",
    "UpdateCriterionLNG.py",
    "UpdateCriterionNuclear.py",
    "UpdateCriterionStorage.py",
]

# ==============================================================================
#  CORE FUNCTIONS
# ==============================================================================

def setup_logging():
    """Configures a basic logger to show timestamped messages."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

def _child_env() -> dict:
    """Return env with repo root and Scripts added to PYTHONPATH for children."""
    env = os.environ.copy()
    extra = [str(_ROOT), str(_SCRIPTS)]
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = os.pathsep.join(extra + [existing]) if existing else os.pathsep.join(extra)
    return env


def run_script(script_path: Path) -> bool:
    """
    Runs a Python script and logs its output.

    Args:
        script_path: The full Path object for the script to be executed.

    Returns:
        True if the script ran successfully, False otherwise.
    """
    script_name = script_path.name
    logging.info(f"========== Running: {script_name} ==========")

    if not script_path.exists():
        logging.warning(f"--- SKIPPING: Script not found at {script_path} ---")
        return True  # Treat as success to not halt the chain for a missing file

    try:
        # Use sys.executable to ensure the same virtual environment is used.
        # The 'check=True' flag will raise CalledProcessError for non-zero exit codes.
        process = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8',
            errors='replace', # Avoid errors with special characters in output
            env=_child_env(),
        )
        
        # Log the standard output from the script
        logging.info(f"--- Output from {script_name} ---\n{process.stdout.strip()}")
        
        # Log any standard errors (often used for warnings)
        if process.stderr:
            logging.warning(f"--- Warnings/Errors from {script_name} ---\n{process.stderr.strip()}")
            
        logging.info(f"========== Finished: {script_name} ==========")
        return True

    except subprocess.CalledProcessError as e:
        logging.critical(f"!!!!!!!!!! ERROR running {script_name} !!!!!!!!!!")
        logging.critical(f"Return Code: {e.returncode}")
        # Log the full output to help with debugging the failed script
        logging.error(f"--- STDOUT ---\n{e.stdout.strip()}")
        logging.error(f"--- STDERR ---\n{e.stderr.strip()}")
        logging.critical(f"!!!!!!!!!! Script {script_name} failed. !!!!!!!!!!")
        return False
        
    except Exception as e:
        logging.critical(f"An unexpected error occurred while trying to run {script_name}: {e}")
        return False

def main():
    """
    Main function to execute all data update scripts in the defined order.
    """
    setup_logging()
    start_time = time.time()
    logging.info("<<<<< Starting Criterion Data Orchestration >>>>>")

    for script_name in SCRIPTS_TO_RUN:
        script_path = ORCHESTRATOR_DIR / script_name
        success = run_script(script_path)
        
        # Halt the entire orchestration if any script fails
        if not success:
            logging.critical("<<<<< Orchestration halted due to a script failure. >>>>>")
            sys.exit(1)

    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"<<<<< All scripts completed successfully in {total_time:.2f} seconds. >>>>>")

if __name__ == '__main__':
    main()


def run_step(name: str, script_path: str):
    script_path = str(pathlib.Path(script_path))
    print(f"\n[{time.strftime('%H:%M:%S')}] RUNNING: {name}...")
    print(f"   - Path: {script_path}")
    result = subprocess.run([PY, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print("=" * 80)
        print(f"   - FATAL ERROR: The script '{name}' failed to execute.")
        print(f"   - Return Code: {result.returncode}")
        print(f"   - STDOUT (Output from the script):\n{result.stdout}")
        print(f"   - STDERR (Error message from the script):\n{result.stderr}")
        print("=" * 80)
        raise SystemExit(1)
    print(f"   - SUCCESS: Finished '{name}' successfully.")
